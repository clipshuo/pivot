"""Orchestrateur : enchaîne les étapes 1 à 10.

`run_pipeline(cfg)` est le seul point d'entrée métier. La ligne de commande
(cli.py) l'appelle, et une future route d'API pourra l'appeler telle quelle.
"""
from __future__ import annotations

import logging

import numpy as np

from . import blending
from . import features
from . import fill
from . import geometry
from . import io_utils
from . import projection
from .baseline import run_baseline
from .config import PivotConfig, configure_runtime
from .debug import DebugWriter, timed
from .models import Camera, StitchReport, StitchResult

log = logging.getLogger("pivot")


def _check_sharpness(images, report) -> None:
    """Signale les images nettement plus floues que les autres."""
    scores = np.array([im.blur for im in images])
    report.metrics["nettete"] = {im.name: round(im.blur, 1) for im in images}
    limit = 0.35 * float(np.median(scores))
    blurry = [im.name for im in images if im.blur < limit]
    if blurry:
        report.warn(f"Images probablement floues (netteté < 35 % de la médiane) : {', '.join(blurry)}. "
                    "Elles donnent peu de points fiables ; filmer plus lentement.")


def _pair_metrics(matches) -> list[dict]:
    return [{"i": m.i, "j": m.j, "appariements": m.n_matches, "inliers_ransac": m.n_ransac,
             "inliers_rotation": m.n_inliers if m.R is not None else 0,
             "confiance": round(m.confidence, 2), "rms_rotation_px": round(m.rms, 2), "valide": bool(m.valid)}
            for m in matches]


def _coverage_stats(coverage: np.ndarray) -> dict:
    """Couverture angulaire du panorama final."""
    H, W = coverage.shape
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    rows = np.flatnonzero(coverage.any(axis=1))
    area = float((coverage.mean(axis=1) * np.cos(lat)).sum() / np.cos(lat).sum())
    return {
        "horizontale_deg": round(float(coverage.any(axis=0).mean()) * 360.0, 1),
        "latitude_max_deg": round(float(np.degrees(lat[rows[0]])), 1) if len(rows) else 0.0,
        "latitude_min_deg": round(float(np.degrees(lat[rows[-1]])), 1) if len(rows) else 0.0,
        "sphere_couverte_pct": round(100.0 * area, 1),
    }


def run_pipeline(cfg: PivotConfig) -> StitchResult:
    """Assemble le dossier cfg.input en panorama équirectangulaire et l'écrit dans cfg.output."""
    report = StitchReport()
    dbg = DebugWriter(cfg.debug, cfg.debug_dir)
    report.metrics["machine"] = configure_runtime(cfg)
    report.metrics["config"] = cfg.to_dict()
    try:
        return _run(cfg, report, dbg)
    finally:
        dbg.metrics(report)                               # le rapport est écrit même en cas d'échec


def _run(cfg, report, dbg) -> StitchResult:
    W, H = cfg.width, cfg.height

    # 1. Chargement : pleine résolution pour le rendu, copie ~1 MP pour l'estimation.
    with timed(report, "1 chargement"):
        images = io_utils.load_images(cfg.input, cfg.work_megapix, cfg.workers)
    n = len(images)
    wh, ww = images[0].work.shape[:2]
    center = ((ww - 1) / 2.0, (wh - 1) / 2.0)
    report.metrics["images"] = {"nombre": n, "taille": list(images[0].full.shape[1::-1]),
                                "taille_travail": [ww, wh]}
    log.info("%d images %dx%d, copie de travail %dx%d", n, *images[0].full.shape[1::-1], ww, wh)
    _check_sharpness(images, report)

    if cfg.baseline:
        with timed(report, "baseline cv2.Stitcher"):
            pano = run_baseline(images, cfg)
        io_utils.save_panorama(cfg.output, pano, cfg.jpeg_quality, add_xmp=False)
        report.metrics["sortie"] = {"fichier": cfg.output, "taille": list(pano.shape[1::-1])}
        return StitchResult(pano, np.ones(pano.shape[:2], bool), {}, report)

    # 2. Points d'intérêt.
    with timed(report, "2 detection"):
        feats = features.detect_features(images, cfg)
    report.metrics["points_par_image"] = [len(f.pts) for f in feats]
    poor = [images[k].name for k, f in enumerate(feats) if len(f.pts) < 100]
    if poor:
        report.warn(f"Moins de 100 points d'intérêt sur : {', '.join(poor)} (image floue ou sans texture).")
    dbg.keypoints(images, feats)

    # 3. Appariement + RANSAC par paire.
    with timed(report, "3 appariement"):
        pairs = features.candidate_pairs(n, cfg.ordered, cfg.neighbor_range)
        matches = features.match_all(feats, pairs, cfg, center)

    # 4. Focale initiale, rotations relatives, graphe de connexion.
    with timed(report, "4 geometrie par paire"):
        focal0 = geometry.estimate_initial_focal(matches, images, cfg, report)
        geometry.refine_pairs(matches, focal0, center, cfg)
        graph = geometry.build_graph(matches, n)
        nodes = geometry.largest_component(graph)
        ref = nodes[0]
        R0 = geometry.initial_rotations(graph, nodes, ref)
        rnorm = float(np.hypot(ww, wh) / 2.0)
        cameras = {k: Camera(R0[k], focal0, center[0], center[1], 0.0, rnorm) for k in nodes}
    report.metrics["paires"] = _pair_metrics(matches)
    report.metrics["focale_initiale_px_travail"] = round(focal0, 1)
    dbg.matches(images, matches, n)
    dbg.graph(matches, n, nodes)

    dropped = [images[k].name for k in range(n) if k not in cameras]
    if dropped:
        report.warn(f"{len(dropped)} image(s) non reliée(s) au reste et écartée(s) : {', '.join(dropped)}. "
                    "Recouvrement insuffisant avec les voisines.")
    weak = [(m.i, m.j) for m in matches if not m.valid and min((m.j - m.i) % n, (m.i - m.j) % n) == 1]
    if cfg.ordered and weak:
        report.warn(f"Paires voisines sans recouvrement fiable : {weak}.")
    loop_closed = any(m.valid and m.i in cameras and m.j in cameras and (m.j - m.i) > n / 2 for m in matches)
    report.metrics["boucle_fermee"] = bool(loop_closed)

    # 5. Ajustement de faisceaux.
    err0 = geometry.reprojection_errors(cameras, matches, center)
    gap0 = geometry.edge_consistency_deg(cameras, matches)
    with timed(report, "5 ajustement de faisceaux"):
        # La distorsion n'est estimée que si la boucle est fermée : sans la contrainte des 360°,
        # focale et distorsion se compensent et l'optimisation dérive vers une focale fausse.
        fit_k = cfg.distortion and loop_closed
        if cfg.distortion and not loop_closed:
            log.info("Boucle non fermée : distorsion radiale non estimée (focale trop peu contrainte).")
        cameras, ba_info = geometry.bundle_adjust(cameras, matches, ref, center, cfg, fit_distortion=fit_k)
    err1 = geometry.reprojection_errors(cameras, matches, center)
    report.metrics["paires"] = _pair_metrics(matches)     # état des paires avant mise à jour (focale initiale)
    geometry.update_pair_rotations(matches, cameras[ref], center)
    gap1 = geometry.edge_consistency_deg(cameras, matches)
    focal = cameras[ref].focal
    report.metrics["ajustement_de_faisceaux"] = {
        **ba_info,
        "erreur_moyenne_avant_px": round(float(err0.mean()), 3),
        "erreur_moyenne_apres_px": round(float(err1.mean()), 3),
        "erreur_rms_apres_px": round(float(np.sqrt((err1 ** 2).mean())), 3),
        "ecart_boucle_max_avant_deg": round(float(gap0.max()), 3),
        "ecart_boucle_max_apres_deg": round(float(gap1.max()), 3),
    }
    report.metrics["focale"] = {
        "px_travail": round(focal, 1), "px_pleine_resolution": round(focal / images[0].scale, 1),
        "champ_horizontal_deg": round(float(np.degrees(2 * np.arctan(ww / (2 * focal)))), 1),
    }
    report.metrics["distorsion_k1"] = round(cameras[ref].k1, 5)
    log.info("Focale %.1f px (pleine résolution), champ horizontal %.1f°, erreur de reprojection "
             "%.2f -> %.2f px, écart de boucle %.2f° -> %.2f°",
             focal / images[0].scale, report.metrics["focale"]["champ_horizontal_deg"],
             err0.mean(), err1.mean(), gap0.max(), gap1.max())
    if err1.mean() > 2.0:
        report.warn(f"Erreur de reprojection élevée après ajustement ({err1.mean():.2f} px à l'échelle de travail) : "
                    "parallaxe probable (la caméra ne tourne pas autour de son centre optique) ou forte "
                    "distorsion de l'objectif. Attendez-vous à des fantômes ou des décalages.")
    if cfg.ordered and not loop_closed:
        report.warn("Boucle non fermée : aucune correspondance fiable entre la fin et le début de la séquence. "
                    "Le tour est incomplet ou le recouvrement final est insuffisant ; le panorama sera partiel "
                    "et la dérive n'est pas corrigée.")

    # 6. Correction d'horizon.
    if cfg.wave_correction:
        cameras = geometry.wave_correct(cameras, ref)

    # 7. Projection équirectangulaire de chaque image dans sa zone du panorama.
    with timed(report, "7 projection"):
        warped = projection.warp_all(images, cameras, cfg)

    # 8. Exposition, coutures, fusion.
    with timed(report, "8 exposition"):
        if cfg.expo:
            gains = blending.compute_gains(warped, W, H)
            blending.apply_gains(warped, gains)
            report.metrics["gains"] = [round(float(g), 3) for g in gains]
    with timed(report, "8 coutures"):
        seams = blending.find_seams(warped, W, H, cfg.seam)
    with timed(report, "8 fusion"):
        pano, coverage = blending.blend(warped, seams, cfg)
    dbg.masks(warped, seams, coverage)

    stats = _coverage_stats(coverage)
    report.metrics["couverture"] = stats
    log.info("Couverture : %(horizontale_deg)s° en longitude, latitudes %(latitude_min_deg)s° à "
             "%(latitude_max_deg)s°, %(sphere_couverte_pct)s %% de la sphère", stats)
    if stats["horizontale_deg"] < 359.0:
        report.warn(f"Tour incomplet : {stats['horizontale_deg']}° couverts en longitude sur 360°.")

    # 9. Zones non couvertes (zénith, nadir).
    with timed(report, "9 remplissage"):
        pano = fill.fill_uncovered(pano, coverage, cfg.fill, cfg.fill_color)

    # 10. Sauvegarde JPEG + métadonnées GPano.
    with timed(report, "10 sauvegarde"):
        io_utils.save_panorama(cfg.output, pano, cfg.jpeg_quality)
    report.metrics["sortie"] = {"fichier": cfg.output, "taille": [W, H]}
    report.timings["total"] = round(sum(report.timings.values()), 3)
    log.info("Panorama écrit : %s (%dx%d) en %.1f s", cfg.output, W, H, report.timings["total"])
    return StitchResult(pano, coverage, cameras, report)
