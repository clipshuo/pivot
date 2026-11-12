"""Structures de données partagées entre tous les modules de Pivot.

Conventions géométriques (valables dans tout le projet) :
- Repère caméra : x à droite, y vers le bas, z vers l'avant (convention OpenCV).
- R_i va du monde vers la caméra i : un rayon monde X se projette en x ~ K · R_i · X.
- Les coordonnées pixel entières désignent le CENTRE du pixel ; le point principal
  d'une image de largeur w est donc (w - 1) / 2.
- Dans le repère monde, l'axe y pointe vers le bas : le « haut » est -y.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("pivot")


# --------------------------------------------------------------------------- erreurs
class PivotError(Exception):
    """Erreur bloquante du pipeline (code de sortie 2 en ligne de commande)."""


class NotEnoughImagesError(PivotError):
    """Moins de deux images exploitables."""


class InsufficientOverlapError(PivotError):
    """Aucune paire d'images n'a assez de correspondances fiables."""


class DisconnectedGraphError(PivotError):
    """Le graphe de connexion ne relie pas assez d'images entre elles."""


class FocalEstimationError(PivotError):
    """Impossible d'obtenir une focale plausible."""


class BundleAdjustmentError(PivotError):
    """L'ajustement de faisceaux a divergé."""


# --------------------------------------------------------------------------- données
@dataclass
class ImageData:
    """Une image d'entrée : pleine résolution pour le rendu, copie réduite pour l'estimation."""
    name: str
    path: str
    full: np.ndarray                  # BGR, pleine résolution
    work: np.ndarray                  # BGR, ~1 mégapixel
    scale: float                      # taille(work) / taille(full)
    exif_focal_px: float | None = None  # focale en pixels pleine résolution, si l'EXIF le permet
    blur: float = 0.0                 # variance du laplacien (netteté)


@dataclass
class Features:
    """Points d'intérêt d'une image, en coordonnées pixel de l'image de travail."""
    pts: np.ndarray                   # (N, 2) float32
    desc: np.ndarray                  # (N, D)


@dataclass
class PairMatch:
    """Correspondances entre les images i et j, et géométrie estimée pour la paire."""
    i: int
    j: int
    pts_i: np.ndarray                 # (M, 2) appariements après test de ratio
    pts_j: np.ndarray
    inliers: np.ndarray               # (M,) bool, après RANSAC puis modèle de rotation
    H: np.ndarray | None = None       # homographie i -> j en coordonnées centrées (sert au tri et à la focale)
    confidence: float = 0.0           # critère de Brown & Lowe : n_inliers / (8 + 0.3 · n_matches)
    valid: bool = False
    R: np.ndarray | None = None       # rotation relative R_j · R_iᵀ
    rms: float = 0.0                  # résidu RMS (px) du modèle de rotation pure sur les inliers
    n_ransac: int = 0                 # inliers de l'homographie, avant le filtre par la rotation

    @property
    def n_matches(self) -> int:
        return int(len(self.pts_i))

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


@dataclass
class Camera:
    """Caméra en rotation pure : orientation R (monde -> caméra) et intrinsèques simples."""
    R: np.ndarray
    focal: float
    cx: float
    cy: float
    k1: float = 0.0                   # distorsion radiale : p_idéal = p_observé · (1 + k1 · |p|² / rnorm²)
    rnorm: float = 1.0                # rayon de normalisation (demi-diagonale de l'image), en pixels

    def K(self) -> np.ndarray:
        return np.array([[self.focal, 0.0, self.cx],
                         [0.0, self.focal, self.cy],
                         [0.0, 0.0, 1.0]])

    def scaled(self, s: float) -> "Camera":
        """Même caméra pour une image redimensionnée d'un facteur s."""
        return Camera(self.R.copy(), self.focal * s,
                      (self.cx + 0.5) * s - 0.5, (self.cy + 0.5) * s - 0.5, self.k1, self.rnorm * s)

    def undistort(self, x, y):
        """Coordonnées centrées observées -> coordonnées centrées idéales (sténopé)."""
        g = 1.0 + self.k1 * (x * x + y * y) / self.rnorm ** 2
        return x * g, y * g

    def distort(self, x, y):
        """Inverse de undistort, par itération de point fixe (converge vite : k1 est petit)."""
        if self.k1 == 0.0:
            return x, y
        xd, yd = x, y
        for _ in range(8):
            g = 1.0 + self.k1 * (xd * xd + yd * yd) / self.rnorm ** 2
            xd, yd = x / g, y / g
        return xd, yd


@dataclass
class WarpedImage:
    """Image projetée dans sa zone utile (ROI) du panorama équirectangulaire.

    La ROI commence à la colonne x0 et peut dépasser le bord droit : les colonnes
    réelles sont (x0 + k) modulo W, ce qui gère le raccord à ±180°.
    """
    index: int
    image: np.ndarray                 # (h, w, 3) uint8
    mask: np.ndarray                  # (h, w) uint8, 255 = pixel couvert
    weight: np.ndarray                # (h, w) float32, distance normalisée au bord de l'image source
    x0: int
    y0: int

    @property
    def h(self) -> int:
        return self.image.shape[0]

    @property
    def w(self) -> int:
        return self.image.shape[1]

    def wraps(self, W: int) -> bool:
        return self.x0 + self.w > W

    def cols(self, W: int) -> np.ndarray:
        return (self.x0 + np.arange(self.w)) % W


@dataclass
class StitchReport:
    """Métriques, temps et avertissements collectés pendant le pipeline."""
    metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning(msg)

    def to_dict(self) -> dict:
        return {"metrics": self.metrics, "timings_s": self.timings, "warnings": self.warnings}


@dataclass
class StitchResult:
    panorama: np.ndarray
    coverage: np.ndarray              # (H, W) bool : pixels réellement couverts par une image
    cameras: dict                     # index d'image -> Camera (échelle de travail)
    report: StitchReport
