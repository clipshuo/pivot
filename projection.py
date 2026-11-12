"""Étape 7 : projection équirectangulaire. C'est le cœur de Pivot.

Principe (mapping inverse) : pour chaque pixel (u, v) du panorama, on calcule
la direction 3D qu'il représente, on la projette dans la caméra par K · R, et on
va lire la couleur à cet endroit de l'image source avec cv2.remap.

    longitude θ = ((u + 0.5) / W - 0.5) · 2π      θ ∈ [-π, π], 0 au centre de l'image
    latitude  φ = (0.5 - (v + 0.5) / H) · π       φ ∈ [-π/2, π/2], +π/2 en haut (zénith)
    rayon     d = (cos φ · sin θ, -sin φ, cos φ · cos θ)     (y vers le bas)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from config import ROI_ALIGN
from models import Camera, WarpedImage


def equirect_rays(width: int, height: int, roi=None) -> np.ndarray:
    """Rayons unitaires (h, w, 3) des pixels du panorama, pour toute l'image ou une zone (x0, y0, w, h).

    x0 + w peut dépasser la largeur : la longitude est périodique, le raccord à
    ±180° est donc géré sans cas particulier.
    """
    x0, y0, w, h = roi if roi is not None else (0, 0, width, height)
    theta = ((x0 + np.arange(w) + 0.5) / width - 0.5) * 2.0 * np.pi
    phi = (0.5 - (y0 + np.arange(h) + 0.5) / height) * np.pi
    cos_phi = np.cos(phi)[:, None]
    rays = np.empty((h, w, 3))
    rays[..., 0] = cos_phi * np.sin(theta)[None, :]
    rays[..., 1] = -np.sin(phi)[:, None]
    rays[..., 2] = cos_phi * np.cos(theta)[None, :]
    return rays


def rays_to_pixels(rays: np.ndarray, cam: Camera):
    """Projette des rayons monde dans une caméra : x ~ K · R · d.

    Renvoie (map_x, map_y, devant) : coordonnées pixel dans l'image source et
    masque des rayons situés devant la caméra (z > 0). Un rayon derrière la
    caméra se projetterait aussi dans l'image, à tort : d'où ce masque.
    """
    Xc = rays @ cam.R.T                                   # monde -> caméra
    z = Xc[..., 2]
    front = z > 1e-6
    zs = np.where(front, z, 1.0)
    x, y = cam.focal * Xc[..., 0] / zs, cam.focal * Xc[..., 1] / zs    # sténopé idéal, centré
    # Loin de l'axe la distorsion n'a plus de sens : on ne garde que le voisinage de l'image.
    front &= (np.abs(x) < 4 * cam.rnorm) & (np.abs(y) < 4 * cam.rnorm) if cam.k1 else True
    xi, yi = np.where(front, x, 0.0), np.where(front, y, 0.0)
    x, y = cam.distort(xi, yi)                            # vers l'image réelle
    if cam.k1:
        # Avec k1 < 0 le modèle se replie loin de l'axe : un rayon hors champ pourrait retomber
        # dans l'image. On ne garde que les points dont l'aller-retour est cohérent.
        xb, yb = cam.undistort(x, y)
        front &= (np.abs(xb - xi) < 0.5) & (np.abs(yb - yi) < 0.5)
    return x + cam.cx, y + cam.cy, front


def pixel_to_ray(u, v, cam: Camera) -> np.ndarray:
    """Inverse de la projection : rayon monde unitaire vu par le pixel (u, v) de la caméra."""
    x, y = cam.undistort(np.asarray(u, float) - cam.cx, np.asarray(v, float) - cam.cy)
    d = np.stack([x / cam.focal, y / cam.focal, np.ones_like(x)], axis=-1)
    d = d @ cam.R                                         # caméra -> monde (Rᵀ appliqué à droite)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def ray_to_equirect(ray, width: int, height: int):
    """Coordonnées (u, v) du panorama pour un rayon monde (indices pixel, centre = entier)."""
    ray = np.asarray(ray, float)
    ray = ray / np.linalg.norm(ray, axis=-1, keepdims=True)
    theta = np.arctan2(ray[..., 0], ray[..., 2])
    phi = np.arcsin(np.clip(-ray[..., 1], -1.0, 1.0))
    return (theta / (2 * np.pi) + 0.5) * width - 0.5, (0.5 - phi / np.pi) * height - 0.5


def _inside(map_x, map_y, front, src_w, src_h):
    return front & (map_x >= 0) & (map_x <= src_w - 1) & (map_y >= 0) & (map_y <= src_h - 1)


def camera_roi(cam: Camera, src_w: int, src_h: int, width: int, height: int, margin: int = 8):
    """Zone du panorama couverte par une caméra : (x0, y0, w, h), alignée sur ROI_ALIGN.

    On projette une version grossière de la sphère entière et on regarde où la
    caméra voit quelque chose. En colonnes, la zone est le plus petit arc
    circulaire contenant les colonnes couvertes, ce qui traite naturellement une
    image à cheval sur ±180° ou qui voit un pôle (zone sur toute la largeur).
    Renvoie None si la caméra ne couvre rien.
    """
    cw, ch = 1024, 512
    mx, my, front = rays_to_pixels(equirect_rays(cw, ch), cam)
    valid = _inside(mx, my, front, src_w, src_h)
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    if len(rows) == 0:
        return None
    a = ROI_ALIGN
    y0 = int(np.floor((rows[0] / ch * height - margin) / a)) * a
    y1 = int(np.ceil(((rows[-1] + 1) / ch * height + margin) / a)) * a
    y0, y1 = max(0, y0), min(height, y1)
    # plus grand trou circulaire entre colonnes couvertes : la zone est son complément
    gaps = np.diff(np.concatenate([cols, [cols[0] + cw]]))
    k = int(np.argmax(gaps))
    start, end = cols[(k + 1) % len(cols)], cols[k]
    length = (end - start) % cw + 1
    x0 = int(np.floor((start / cw * width - margin) / a)) * a
    x1 = int(np.ceil(((start + length) / cw * width + margin) / a)) * a
    if gaps[k] <= 2 or x1 - x0 >= width:                  # tour complet (pôle visible)
        return 0, y0, width, y1 - y0
    return x0 % width, y0, x1 - x0, y1 - y0


def warp_image(img: np.ndarray, cam: Camera, width: int, height: int, index: int = 0) -> WarpedImage | None:
    """Projette une image dans sa zone du panorama.

    Renvoie l'image projetée, son masque de validité, et un poids par pixel : la
    distance au bord le plus proche de l'image SOURCE (1 au centre, 0 au bord).
    Ce poids sert au fondu linéaire et au tracé des coutures.
    """
    src_h, src_w = img.shape[:2]
    roi = camera_roi(cam, src_w, src_h, width, height)
    if roi is None:
        return None
    map_x, map_y, front = rays_to_pixels(equirect_rays(width, height, roi), cam)
    valid = _inside(map_x, map_y, front, src_w, src_h)
    mx, my = map_x.astype(np.float32), map_y.astype(np.float32)
    mx[~valid] = -1.0                                     # hors image : remap y met du noir
    my[~valid] = -1.0
    warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    border = np.minimum(np.minimum(map_x, src_w - 1 - map_x), np.minimum(map_y, src_h - 1 - map_y))
    weight = np.where(valid, border / (0.5 * min(src_w, src_h)), 0.0).clip(0.0, 1.0).astype(np.float32)
    return WarpedImage(index, warped, valid.astype(np.uint8) * 255, weight, roi[0], roi[1])


def render_scale(focal_full: float, width: int, oversample: float = 1.25) -> float:
    """Facteur de réduction des images sources avant projection.

    Le panorama a W / 2π pixels par radian ; une image source en a `focale`. Si
    la source est bien plus fine que la sortie, l'échantillonner directement
    crée du repliement (aliasing) : on la réduit d'abord proprement.
    """
    return float(min(1.0, oversample * (width / (2.0 * np.pi)) / focal_full))


def warp_all(images, cameras: dict, cfg) -> list[WarpedImage]:
    """Projette toutes les images retenues, en parallèle.

    `cameras` est à l'échelle de travail ; on la convertit à l'échelle de rendu.
    """
    any_cam = next(iter(cameras.values()))
    s_work = images[0].scale
    s_render = render_scale(any_cam.focal / s_work, cfg.width)

    def one(idx):
        full = images[idx].full
        src = full if s_render >= 1.0 else cv2.resize(full, None, fx=s_render, fy=s_render,
                                                      interpolation=cv2.INTER_AREA)
        cam = cameras[idx].scaled(src.shape[1] / images[idx].work.shape[1])
        return warp_image(src, cam, cfg.width, cfg.height, idx)

    with ThreadPoolExecutor(cfg.workers) as ex:
        return [w for w in ex.map(one, sorted(cameras)) if w is not None]
