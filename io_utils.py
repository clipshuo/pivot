"""Étapes 1 et 10 : chargement des images, lecture EXIF, sauvegarde JPEG + métadonnées XMP GPano."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from models import ImageData, NotEnoughImagesError, PivotError

log = logging.getLogger("pivot")

EXTENSIONS = (".jpg", ".jpeg", ".png")
XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def list_images(folder: str) -> list[str]:
    """Chemins des images du dossier, triés par nom (l'ordre de la rotation)."""
    if not os.path.isdir(folder):
        raise PivotError(f"Dossier introuvable : {folder}")
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(EXTENSIONS))
    return [os.path.join(folder, n) for n in names]


def read_exif_focal_px(path: str, image_width: int, image_height: int) -> float | None:
    """Focale en pixels (pleine résolution) depuis l'EXIF, ou None si elle n'est pas calculable.

    La focale en millimètres seule ne suffit pas : il faut soit l'équivalent 35 mm,
    soit la densité du capteur (FocalPlaneXResolution).
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769) if exif else {}
    except Exception:
        return None
    long_side = max(image_width, image_height)
    f35 = sub.get(0xA405)                       # FocalLengthIn35mmFilm
    if f35:
        return float(f35) / 36.0 * long_side    # 36 mm = grand côté du plein format
    f_mm, res, unit = sub.get(0x920A), sub.get(0xA20E), sub.get(0xA210, 2)
    if f_mm and res:
        mm_per_unit = {2: 25.4, 3: 10.0}.get(int(unit), 25.4)
        return float(f_mm) * float(res) / mm_per_unit
    return None


def blur_score(gray: np.ndarray) -> float:
    """Netteté : variance du laplacien. Une valeur faible signale une image floue."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _load_one(path: str, work_megapix: float) -> ImageData:
    full = cv2.imread(path, cv2.IMREAD_COLOR)   # applique l'orientation EXIF
    if full is None:
        raise PivotError(f"Image illisible : {path}")
    h, w = full.shape[:2]
    scale = min(1.0, (work_megapix * 1e6 / (w * h)) ** 0.5)
    work = full if scale == 1.0 else cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    scale = work.shape[1] / w
    return ImageData(os.path.basename(path), path, full, work, scale,
                     read_exif_focal_px(path, w, h),
                     blur_score(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)))


def load_images(folder: str, work_megapix: float, workers: int = 4) -> list[ImageData]:
    """Charge toutes les images en parallèle et fabrique les copies de travail."""
    paths = list_images(folder)
    if len(paths) < 2:
        raise NotEnoughImagesError(f"Il faut au moins 2 images, {len(paths)} trouvée(s) dans {folder}.")
    with ThreadPoolExecutor(workers) as ex:
        images = list(ex.map(lambda p: _load_one(p, work_megapix), paths))
    sizes = {im.full.shape[:2] for im in images}
    if len(sizes) > 1:
        raise PivotError(f"Toutes les images doivent avoir la même taille (trouvé : {sorted(sizes)}). "
                         "Pivot suppose une seule caméra sans zoom.")
    return images


def build_gpano_xmp(width: int, height: int) -> bytes:
    """Paquet XMP GPano déclarant une sphère équirectangulaire complète."""
    fields = {
        "UsePanoramaViewer": "True",
        "ProjectionType": "equirectangular",
        "StitchingSoftware": "Pivot",
        "FullPanoWidthPixels": width,
        "FullPanoHeightPixels": height,
        "CroppedAreaImageWidthPixels": width,
        "CroppedAreaImageHeightPixels": height,
        "CroppedAreaLeftPixels": 0,
        "CroppedAreaTopPixels": 0,
    }
    attrs = "\n".join(f'      GPano:{k}="{v}"' for k, v in fields.items())
    xmp = ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
           '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
           '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
           '    <rdf:Description rdf:about=""\n'
           '      xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"\n'
           f'{attrs}/>\n'
           '  </rdf:RDF>\n'
           '</x:xmpmeta>\n'
           '<?xpacket end="w"?>')
    return xmp.encode("utf-8")


def insert_xmp(jpeg: bytes, xmp: bytes) -> bytes:
    """Insère le paquet XMP dans un segment APP1, juste après l'en-tête JPEG (et l'APP0 éventuel)."""
    if jpeg[:2] != b"\xff\xd8":
        raise PivotError("Flux JPEG invalide.")
    payload = XMP_HEADER + xmp
    if len(payload) + 2 > 0xFFFF:
        raise PivotError("Paquet XMP trop long pour un segment APP1.")
    segment = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    pos = 2
    if jpeg[2:4] == b"\xff\xe0":                # APP0 (JFIF) : le XMP se place après
        pos = 4 + int.from_bytes(jpeg[4:6], "big")
    return jpeg[:pos] + segment + jpeg[pos:]


def save_panorama(path: str, pano: np.ndarray, quality: int = 95, add_xmp: bool = True) -> None:
    """Écrit le panorama. En JPEG, ajoute les métadonnées GPano si l'image est bien 2:1."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".jpg", ".jpeg"):
        if not cv2.imwrite(path, pano):
            raise PivotError(f"Écriture impossible : {path}")
        if add_xmp:
            log.warning("Métadonnées GPano non écrites : elles ne sont gérées que pour le JPEG.")
        return
    ok, buf = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise PivotError("Encodage JPEG impossible.")
    data = buf.tobytes()
    h, w = pano.shape[:2]
    if add_xmp and w == 2 * h:
        data = insert_xmp(data, build_gpano_xmp(w, h))
    with open(path, "wb") as f:
        f.write(data)
