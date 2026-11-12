"""Paramètres par défaut du pipeline et réglage de la machine."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict

import cv2

log = logging.getLogger("pivot")

ROI_ALIGN = 64      # les zones projetées sont alignées sur 64 px (permet 6 niveaux de pyramide)
WIDTH_ALIGN = 128   # la largeur du panorama est un multiple de 128 (hauteur multiple de 64)


@dataclass
class PivotConfig:
    input: str = ""
    output: str = "pano.jpg"
    width: int = 4096               # largeur de sortie ; la hauteur vaut width / 2
    # --- prétraitement
    work_megapix: float = 1.0       # taille de la copie de travail (détection + estimation)
    # --- points d'intérêt et appariement
    detector: str = "sift"          # "sift" | "orb"
    max_features: int = 4000
    ratio: float = 0.75             # test de ratio de Lowe
    ordered: bool = True            # images triées dans l'ordre de la rotation
    neighbor_range: int = 2         # en mode ordonné : compare i avec i+1 .. i+neighbor_range (boucle fermée)
    # --- géométrie
    ransac_thresh: float = 3.0      # px, à l'échelle de travail
    min_inliers: int = 20
    min_confidence: float = 1.0     # seuil du critère de Brown & Lowe
    focal: float | None = None      # focale imposée (px pleine résolution), sinon EXIF puis estimation
    ba_max_points: int = 200        # points par paire utilisés dans l'ajustement de faisceaux
    distortion: bool = True         # estime un coefficient de distorsion radiale dans l'ajustement
    wave_correction: bool = True
    # --- capteurs du téléphone (capture.json de Pivot Capture, lu dans le dossier d'entrée)
    sensors: bool = True            # utilise les orientations mesurées si capture.json est présent
    sensor_sigma_deg: float = 0.5   # incertitude d'une orientation mesurée (a priori de l'ajustement)
    sensor_max_gap_deg: float = 5.0 # une paire qui contredit le gyroscope de plus que cela est rejetée
    # --- rendu
    expo: bool = True               # compensation de gain
    seam: str = "voronoi"           # "voronoi" | "graphcut"
    blend: str = "multiband"        # "multiband" | "linear"
    bands: int = 5
    fill: str = "inpaint"           # "none" | "inpaint" | "color" | "gradient"
    fill_color: tuple = (0, 0, 0)   # BGR
    jpeg_quality: int = 95
    # --- exécution
    workers: int = 0                # 0 = tous les cœurs disponibles
    baseline: bool = False
    debug: bool = False
    debug_dir: str = "debug"

    def __post_init__(self):
        w = max(WIDTH_ALIGN, int(round(self.width / WIDTH_ALIGN)) * WIDTH_ALIGN)
        if w != self.width:
            log.warning("Largeur %d arrondie à %d (multiple de %d).", self.width, w, WIDTH_ALIGN)
            self.width = w
        self.bands = int(min(max(self.bands, 1), 6))
        if self.workers <= 0:
            self.workers = available_cores()

    @property
    def height(self) -> int:
        return self.width // 2

    def to_dict(self) -> dict:
        return asdict(self)


def available_cores() -> int:
    """Cœurs réellement utilisables (respecte l'affinité et la limite cgroup d'un conteneur)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:          # macOS / Windows
        n = os.cpu_count() or 1
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def configure_runtime(cfg: PivotConfig) -> dict:
    """Règle OpenCV pour exploiter la machine et renvoie ce qui a été détecté.

    OpenCV parallélise en interne (remap, pyramides, SIFT) et relâche le GIL :
    on combine donc ses fils internes avec un pool de threads Python par image.
    """
    cv2.setUseOptimized(True)
    cv2.setNumThreads(cfg.workers)
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        ram_gb = 0.0
    try:
        cuda = cv2.cuda.getCudaEnabledDeviceCount()
    except (cv2.error, AttributeError):
        cuda = 0
    info = {
        "coeurs": cfg.workers,
        "ram_go": round(ram_gb, 1),
        "opencv": cv2.__version__,
        "opencv_optimise": bool(cv2.useOptimized()),
        "opencv_cuda": int(cuda),
        "opencl": bool(cv2.ocl.haveOpenCL()),
    }
    log.info("Machine : %(coeurs)d cœurs, %(ram_go)s Go RAM, OpenCV %(opencv)s, "
             "CUDA=%(opencv_cuda)d, OpenCL=%(opencl)s", info)
    return info


def config_from_args(args) -> PivotConfig:
    """Construit la configuration depuis les arguments de la ligne de commande."""
    return PivotConfig(
        input=args.input, output=args.output, width=args.width,
        work_megapix=args.work_mp, detector=args.detector, ratio=args.ratio,
        ordered=not args.unordered, neighbor_range=args.neighbor_range,
        focal=args.focal, wave_correction=not args.no_wave, distortion=not args.no_distortion,
        expo=not args.no_expo, seam=args.seam, blend=args.blend, bands=args.bands,
        fill=args.fill, fill_color=tuple(args.fill_color), jpeg_quality=args.quality,
        workers=args.workers, baseline=args.baseline,
        debug=args.debug, debug_dir=args.debug_dir, sensors=not args.no_sensors,
    )
