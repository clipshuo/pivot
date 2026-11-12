"""Orientations mesurées par le téléphone (fichier capture.json de l'app Pivot Capture).

Le gyroscope place chaque image à 1 ou 2 degrés près, y compris les images sans texture
(mur blanc, ciel) que la vision seule doit écarter. Le pipeline s'en sert pour :
- ne comparer que les paires d'images qui se recouvrent vraiment ;
- rejeter les appariements incompatibles avec l'orientation mesurée ;
- initialiser toutes les rotations, et les retenir près de la mesure pendant l'ajustement ;
- connaître la verticale, donc l'horizon, sans la déduire des images.

Conventions de capture.json : quaternion (w, x, y, z) appareil -> monde Android (z vers le haut) ;
appareil : x à droite de l'écran, y vers le haut de l'écran, z sortant de l'écran ; caméra arrière
en portrait (repère OpenCV) = diag(1, -1, -1) · appareil.
Monde de Pivot : y vers le bas. On pose x_pivot = x_android, y_pivot = -z_android, z_pivot = y_android.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

log = logging.getLogger("pivot")

FILENAME = "capture.json"
_CAM_FROM_DEVICE = np.diag([1.0, -1.0, -1.0])
_PIVOT_FROM_ANDROID = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


@dataclass
class SensorPoses:
    rotations: dict[str, np.ndarray]      # nom de fichier -> R monde (Pivot) -> caméra
    focal_px: float | None                # focale pleine résolution annoncée par l'appareil
    hfov_deg: float | None
    vfov_deg: float | None


def rotation_from_quaternion(q_wxyz) -> np.ndarray:
    """R monde Pivot -> caméra, depuis le quaternion appareil -> monde Android."""
    w, x, y, z = (float(v) for v in q_wxyz)
    R_dw = Rotation.from_quat([x, y, z, w]).as_matrix()
    return _CAM_FROM_DEVICE @ R_dw.T @ _PIVOT_FROM_ANDROID.T


def quaternion_from_rotation(R: np.ndarray) -> list[float]:
    """Inverse de rotation_from_quaternion (sert aux tests et aux données synthétiques)."""
    R_dw = (_CAM_FROM_DEVICE.T @ R @ _PIVOT_FROM_ANDROID).T
    x, y, z, w = Rotation.from_matrix(R_dw).as_quat()
    return [float(w), float(x), float(y), float(z)]


def load_poses(folder: str) -> SensorPoses | None:
    """Lit <dossier>/capture.json ; None s'il est absent ou inexploitable."""
    path = os.path.join(folder, FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rotations = {s["file"]: rotation_from_quaternion(s["quaternion_wxyz"]) for s in data["shots"]}
        cam = data.get("camera", {})
        return SensorPoses(rotations, cam.get("focal_px") or None, cam.get("hfov_deg"), cam.get("vfov_deg"))
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.warning("capture.json illisible (%s) : orientations du téléphone ignorées.", e)
        return None


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle de la rotation qui sépare deux orientations."""
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1.0) / 2.0, -1.0, 1.0))))


def overlapping_pairs(R: list[np.ndarray], max_angle_deg: float) -> list[tuple[int, int]]:
    """Paires dont les directions de visée sont assez proches pour que les images se recouvrent."""
    fwd = np.stack([r[2] for r in R])                 # axe z de chaque caméra, dans le monde
    ang = np.degrees(np.arccos(np.clip(fwd @ fwd.T, -1.0, 1.0)))
    n = len(R)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if ang[i, j] < max_angle_deg]
