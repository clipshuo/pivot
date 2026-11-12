"""Mode de référence (--baseline) : assemblage par cv2.Stitcher.

Sert à vérifier qu'un jeu d'images est assemblable et à comparer visuellement
avec le pipeline de Pivot. cv2.Stitcher en mode PANORAMA utilise en interne le
pipeline cv2.detail avec projection sphérique. Sa sortie est recadrée sur la
zone couverte : ce n'est pas une image équirectangulaire 2:1 complète.
"""
from __future__ import annotations

import cv2
import numpy as np

from models import ImageData, PivotError

_STATUS = {
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "pas assez d'images reliées entre elles (recouvrement insuffisant)",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "échec de l'estimation des homographies",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "échec de l'ajustement des paramètres de caméra",
}


def run_baseline(images: list[ImageData], cfg) -> np.ndarray:
    """Assemble les images avec cv2.Stitcher et renvoie le panorama sphérique recadré."""
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    stitcher.setRegistrationResol(min(1.0, cfg.work_megapix))
    stitcher.setWaveCorrection(cfg.wave_correction)
    # Résolution de composition (en mégapixels par image) comparable à celle du rendu de Pivot.
    h, w = images[0].full.shape[:2]
    stitcher.setCompositingResol(min(w * h / 1e6, 1.0 * (cfg.width / 4096) ** 2))
    status, pano = stitcher.stitch([im.full for im in images])
    if status != cv2.Stitcher_OK:
        raise PivotError(f"cv2.Stitcher a échoué : {_STATUS.get(status, f'code {status}')}.")
    return pano
