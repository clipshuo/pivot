"""Étape 9 : gestion des zones sans image source (zénith, nadir, tour incomplet).

Le panorama reste toujours une image 2:1 valide ; seul le contenu des zones non
couvertes change selon la méthode choisie.
"""
from __future__ import annotations

import cv2
import numpy as np


def _feather(pano, filled, coverage, radius: int):
    """Compose `filled` dans les trous avec une transition douce de quelques pixels côté trou."""
    hole = (~coverage).astype(np.uint8)
    dist = cv2.distanceTransform(hole, cv2.DIST_L2, 3)
    alpha = np.clip(dist / max(radius, 1), 0.0, 1.0)[..., None]
    edge = cv2.blur(pano, (2 * radius + 1, 2 * radius + 1)).astype(np.float32)
    cnt = cv2.blur(coverage.astype(np.float32), (2 * radius + 1, 2 * radius + 1))[..., None]
    near = edge / np.maximum(cnt, 1e-3)                   # couleur moyenne des pixels couverts voisins
    out = pano.copy()
    mix = near * (1 - alpha) + filled.astype(np.float32) * alpha
    mix = np.where(cnt > 1e-3, mix, filled)
    out[~coverage] = np.clip(mix, 0, 255).astype(np.uint8)[~coverage]
    return out


def _fill_inpaint(pano, coverage):
    """Inpainting (Telea) en basse résolution, circulaire en longitude, puis agrandi.

    Les trous couvrent souvent plus de la moitié de la sphère : à pleine
    résolution l'inpainting prendrait des minutes pour un résultat de toute
    façon lisse. On travaille donc sur une image d'environ 512 px de large.
    """
    H, W = coverage.shape
    s = max(1, W // 512)
    small = cv2.resize(pano, (W // s, H // s), interpolation=cv2.INTER_AREA)
    cov_s = cv2.resize(coverage.astype(np.uint8), (W // s, H // s), interpolation=cv2.INTER_AREA) > 0
    cov_s = cv2.erode(cov_s.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not cov_s.any():
        return pano
    pad = small.shape[1] // 4
    wrap = lambda a: np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
    res = cv2.inpaint(wrap(small), wrap((~cov_s).astype(np.uint8) * 255), 5, cv2.INPAINT_TELEA)
    # Telea laisse des traînées quand le trou est grand : plus on s'éloigne de la zone
    # couverte, plus on remplace son résultat par une version très lissée.
    sig = small.shape[1] / 24.0
    smooth = cv2.GaussianBlur(res, (0, 0), sig)
    dist = cv2.distanceTransform(wrap((~cov_s).astype(np.uint8)), cv2.DIST_L2, 3)
    a = np.clip(dist / (2.0 * sig), 0.0, 1.0)[..., None]
    res = (res * (1 - a) + smooth * a).astype(np.uint8)
    res = res[:, pad:pad + small.shape[1]]
    filled = cv2.resize(res, (W, H), interpolation=cv2.INTER_CUBIC)
    return _feather(pano, filled, coverage, max(4, W // 512))


def _fill_gradient(pano, coverage):
    """Dégradé vertical : chaque colonne fond de la couleur du bord couvert vers la couleur moyenne du pôle."""
    H, W = coverage.shape
    out = pano.astype(np.float32)
    anycol = coverage.any(axis=0)
    if not anycol.any():
        return pano
    top = np.where(anycol, coverage.argmax(axis=0), 0)
    bot = np.where(anycol, H - 1 - coverage[::-1].argmax(axis=0), H - 1)
    k = max(3, (W // 64) | 1)

    def edge_colors(rows):
        """Couleur du bord par colonne, lissée circulairement ; colonnes vides interpolées."""
        col = pano[np.clip(rows, 0, H - 1), np.arange(W)].astype(np.float32)
        wgt = anycol.astype(np.float32)[:, None]
        num = np.concatenate([col * wgt] * 3, axis=0)[None]
        den = np.concatenate([wgt] * 3, axis=0)[None]
        kk = k
        while True:                                       # élargit le lissage jusqu'à couvrir les colonnes vides
            n_ = cv2.blur(num, (kk, 1))[0, W:2 * W]
            d_ = cv2.blur(den, (kk, 1))[0, W:2 * W]
            if (d_ > 1e-6).all() or kk > 2 * W:
                break
            kk = kk * 2 + 1
        return n_ / np.maximum(d_.reshape(W, 1), 1e-6)

    margin = max(2, H // 256)
    ct, cb = edge_colors(top + margin), edge_colors(bot - margin)
    y = np.arange(H, dtype=np.float32)[:, None]
    for colors, pole_side in ((ct, True), (cb, False)):
        pole = colors.mean(axis=0)
        if pole_side:
            a = np.clip(y / np.maximum(top[None, :], 1), 0, 1)             # 0 au pôle, 1 au bord couvert
            region = (y < top[None, :]) | ~anycol[None, :] & (y < H / 2)
        else:
            a = np.clip((H - 1 - y) / np.maximum(H - 1 - bot[None, :], 1), 0, 1)
            region = (y > bot[None, :]) | ~anycol[None, :] & (y >= H / 2)
        a = (a ** 1.5)[..., None]
        grad = colors[None, :, :] * a + pole[None, None, :] * (1 - a)
        region &= ~coverage
        out[region] = grad[region]
    out = np.clip(out, 0, 255).astype(np.uint8)
    remaining = ~coverage & ~((y < top[None, :]) | (y > bot[None, :]) | ~anycol[None, :])
    if remaining.any():                                   # trous internes : inpainting
        out = _fill_inpaint(out, ~remaining)
    return out


def fill_uncovered(pano: np.ndarray, coverage: np.ndarray, method: str = "inpaint",
                   color=(0, 0, 0)) -> np.ndarray:
    """Remplit les pixels sans image source.

    - "none"     : noir.
    - "color"    : couleur unie (BGR).
    - "gradient" : dégradé vers les pôles à partir des couleurs du bord couvert.
    - "inpaint"  : prolongement des couleurs voisines par cv2.inpaint.
    """
    if coverage.all() or method == "none":
        return pano
    if method == "color":
        out = pano.copy()
        out[~coverage] = np.uint8(color)
        return out
    if method == "gradient":
        return _fill_gradient(pano, coverage)
    if method == "inpaint":
        return _fill_inpaint(pano, coverage)
    raise ValueError(f"Méthode de remplissage inconnue : {method}")
