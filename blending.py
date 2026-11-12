"""Étape 8 : compensation d'exposition, recherche de coutures et fusion.

Toutes les images arrivent déjà projetées dans leur zone (ROI) du panorama.
Une ROI peut être à cheval sur le bord droit : ses colonnes réelles sont
(x0 + k) modulo W. Les fonctions _get / _put masquent ce détail.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from models import WarpedImage

log = logging.getLogger("pivot")


# ------------------------------------------------------------------ accès aux zones avec repli à ±180°
def _get(canvas: np.ndarray, w: WarpedImage) -> np.ndarray:
    """Copie (ou vue) de la zone de `canvas` occupée par l'image projetée w."""
    W = canvas.shape[1]
    rows = slice(w.y0, w.y0 + w.h)
    if not w.wraps(W):
        return canvas[rows, w.x0:w.x0 + w.w]
    return canvas[rows][:, w.cols(W)]


def _put(canvas: np.ndarray, w: WarpedImage, values: np.ndarray) -> None:
    W = canvas.shape[1]
    rows = slice(w.y0, w.y0 + w.h)
    if not w.wraps(W):
        canvas[rows, w.x0:w.x0 + w.w] = values
    else:
        canvas[rows, w.cols(W)] = values


def _lowres_step(W: int) -> int:
    """Pas de sous-échantillonnage (puissance de 2) donnant un panorama d'environ 1024 px de large."""
    s = 1
    while W // (2 * s) >= 1024 and s < 64:
        s *= 2
    return s


def coverage_mask(warped: list[WarpedImage], W: int, H: int) -> np.ndarray:
    """Masque (H, W) des pixels couverts par au moins une image."""
    cov = np.zeros((H, W), bool)
    for w in warped:
        _put(cov, w, _get(cov, w) | (w.mask > 0))
    return cov


# ------------------------------------------------------------------ compensation d'exposition
def compute_gains(warped: list[WarpedImage], W: int, H: int,
                  sigma_n: float = 10.0, sigma_g: float = 0.1) -> np.ndarray:
    """Un gain par image, par moindres carrés (Brown & Lowe).

    On minimise Σ_ij N_ij [ (g_i·I_ij - g_j·I_ji)² / σn² + (1 - g_i)² / σg² ]
    où I_ij est l'intensité moyenne de l'image i dans sa zone commune avec j et
    N_ij le nombre de pixels communs. Le second terme retient les gains près de
    1, sinon la solution triviale g = 0 gagnerait. Le système est linéaire.
    """
    n = len(warped)
    s = _lowres_step(W)
    Ws, Hs = W // s, H // s
    lum = np.zeros((n, Hs, Ws), np.float32)
    msk = np.zeros((n, Hs, Ws), bool)
    for k, w in enumerate(warped):
        g = cv2.cvtColor(w.image[::s, ::s], cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = w.mask[::s, ::s] > 0
        rows = slice(w.y0 // s, w.y0 // s + g.shape[0])
        cols = (w.x0 // s + np.arange(g.shape[1])) % Ws
        lum[k][rows, cols] = g
        msk[k][rows, cols] = m
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            ov = msk[i] & msk[j]
            N = int(ov.sum())
            if N < 50:
                continue
            Iij, Iji = float(lum[i][ov].mean()), float(lum[j][ov].mean())
            A[i, i] += N * (Iij ** 2 / sigma_n ** 2 + 1 / sigma_g ** 2)
            A[j, j] += N * (Iji ** 2 / sigma_n ** 2 + 1 / sigma_g ** 2)
            A[i, j] -= N * Iij * Iji / sigma_n ** 2
            A[j, i] -= N * Iij * Iji / sigma_n ** 2
            b[i] += N / sigma_g ** 2
            b[j] += N / sigma_g ** 2
    gains = np.ones(n)
    used = np.diag(A) > 0
    if used.any():
        gains[used] = np.linalg.solve(A[np.ix_(used, used)], b[used])
    return gains


def apply_gains(warped: list[WarpedImage], gains: np.ndarray) -> None:
    for w, g in zip(warped, gains):
        if abs(g - 1.0) > 1e-3:
            w.image = cv2.convertScaleAbs(w.image, alpha=float(g))


# ------------------------------------------------------------------ coutures
def _partition(warped, W, H, candidates) -> np.ndarray:
    """Attribue chaque pixel à l'image candidate de plus fort poids. Renvoie l'index d'image (H, W), -1 = aucune."""
    best_w = np.full((H, W), -1.0, np.float32)
    best_i = np.full((H, W), -1, np.int16)
    for k, (w, cand) in enumerate(zip(warped, candidates)):
        cw = np.where(cand, w.weight + 1e-6, -1.0).astype(np.float32)
        cur_w, cur_i = _get(best_w, w), _get(best_i, w)
        upd = cw > cur_w
        _put(best_w, w, np.where(upd, cw, cur_w))
        _put(best_i, w, np.where(upd, k, cur_i).astype(np.int16))
    return best_i


def _graphcut_candidates(warped, W) -> list[np.ndarray]:
    """Coutures par coupe de graphe (cv2.detail), calculées en basse résolution.

    La coupe évite de traverser les zones où deux images diffèrent (objets en
    mouvement, parallaxe). cv2.detail ne connaît pas le repli à ±180° : les
    paires à cheval sur ce bord sont départagées ensuite par _partition.
    """
    s = _lowres_step(W)
    imgs = [np.ascontiguousarray(w.image[::s, ::s]).astype(np.float32) for w in warped]
    masks = [np.ascontiguousarray(w.mask[::s, ::s]) for w in warped]
    corners = [(w.x0 // s, w.y0 // s) for w in warped]
    finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
    out = finder.find(imgs, corners, masks)
    cands = []
    for w, m in zip(warped, out):
        m = m.get() if isinstance(m, cv2.UMat) else m
        m = cv2.resize(m, (w.w, w.h), interpolation=cv2.INTER_NEAREST)
        cands.append((m > 0) & (w.mask > 0))
    return cands


def find_seams(warped: list[WarpedImage], W: int, H: int, method: str = "voronoi") -> list[np.ndarray]:
    """Découpe le panorama en régions, une par image : chaque pixel a une seule image source.

    - "voronoi" : le pixel va à l'image dont il est le plus loin du bord. Simple,
      rapide, coutures au milieu des zones de recouvrement.
    - "graphcut" : coupe de graphe guidée par la différence de couleur, puis
      repli sur la règle précédente pour les pixels restés sans propriétaire.
    Renvoie un masque booléen par image, de la taille de sa ROI.
    """
    valid = [w.mask > 0 for w in warped]
    owner = _partition(warped, W, H, valid)
    if method == "graphcut":
        try:
            gc = _partition(warped, W, H, _graphcut_candidates(warped, W))
            owner = np.where(gc >= 0, gc, owner)
        except cv2.error as e:
            log.warning("Coupe de graphe indisponible (%s) : coutures Voronoï utilisées.", e)
    return [_get(owner, w) == k for k, w in enumerate(warped)]


# ------------------------------------------------------------------ fondu linéaire
def blend_linear(warped, seams, W: int, H: int):
    """Fondu simple : moyenne des images pondérée par la distance au bord de chaque image source."""
    num = np.zeros((H, W, 3), np.float32)
    den = np.zeros((H, W), np.float32)
    for w in warped:
        wt = w.weight + 1e-6 * (w.mask > 0)
        _put(num, w, _get(num, w) + w.image.astype(np.float32) * wt[..., None])
        _put(den, w, _get(den, w) + wt)
    cov = den > 0
    pano = np.zeros((H, W, 3), np.uint8)
    pano[cov] = np.clip(num[cov] / den[cov][:, None], 0, 255).astype(np.uint8)
    return pano, cov


# ------------------------------------------------------------------ fusion multi-bandes
def _extend_edges(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Prolonge les couleurs de l'image dans sa zone invalide (méthode « push-pull »).

    Sans cela, le noir autour de l'image se mélangerait aux basses fréquences de
    la pyramide et assombrirait les bords du panorama.
    """
    c = img * valid[..., None]
    wt = valid.astype(np.float32)
    levels = [(c, wt)]
    while min(wt.shape) > 2 and len(levels) < 12 and (levels[-1][1] < 1e-3).any():
        c, wt = cv2.pyrDown(c), cv2.pyrDown(wt)
        levels.append((c, wt))
    c, wt = levels[-1]
    for c_lvl, w_lvl in reversed(levels[:-1]):
        size = (w_lvl.shape[1], w_lvl.shape[0])
        cu, wu = cv2.pyrUp(c, dstsize=size), cv2.pyrUp(wt, dstsize=size)
        hole = w_lvl < 1e-3
        c = np.where(hole[..., None], cu, c_lvl)
        wt = np.where(hole, wu, w_lvl)
    out = c / np.maximum(wt, 1e-6)[..., None]
    out[valid] = img[valid]
    return out


def _pyramids(img: np.ndarray, weight: np.ndarray, bands: int):
    """Pyramide laplacienne de l'image et pyramide gaussienne du poids."""
    g = [img]
    wp = [weight]
    for _ in range(bands):
        g.append(cv2.pyrDown(g[-1]))
        wp.append(cv2.pyrDown(wp[-1]))
    lap = [g[k] - cv2.pyrUp(g[k + 1], dstsize=(g[k].shape[1], g[k].shape[0])) for k in range(bands)]
    lap.append(g[bands])
    return lap, wp


def blend_multiband(warped, seams, W: int, H: int, bands: int = 5, workers: int = 4):
    """Fusion multi-bandes (Burt & Adelson).

    Chaque image est décomposée en bandes de fréquences (pyramide laplacienne).
    Les basses fréquences sont mélangées sur une large zone (pas de saut de
    luminosité visible), les hautes fréquences sur une zone étroite (pas de
    flou ni de fantôme). La largeur de transition vient du masque de couture,
    lissé à chaque niveau (pyramide gaussienne).

    Le traitement est circulaire en longitude : la colonne W-1 est voisine de la
    colonne 0, sinon une ligne verticale apparaîtrait dans le viewer à ±180°.
    """
    while bands > 1 and (W % (1 << bands) or H % (1 << bands)):
        bands -= 1
    step = 1 << bands
    P = 4 * step                                          # marge autour de chaque zone, multiple de 2^bands
    num = [np.zeros((H >> k, W >> k, 3), np.float32) for k in range(bands + 1)]
    den = [np.zeros((H >> k, W >> k), np.float32) for k in range(bands + 1)]

    def prepare(args):
        w, seam = args
        ys, ye = max(0, w.y0 - P), min(H, w.y0 + w.h + P)
        full = w.w + 2 * P >= W                           # l'image fait (presque) le tour complet
        if full:
            x_start, wl = 0, W
            img = np.zeros((ye - ys, W, 3), np.float32)
            sm = np.zeros((ye - ys, W), np.float32)
            va = np.zeros((ye - ys, W), bool)
            r, c = slice(w.y0 - ys, w.y0 - ys + w.h), w.cols(W)
            img[r, c], sm[r, c], va[r, c] = w.image, seam, w.mask > 0
            img = _extend_edges(img, va)
            wrap = lambda a: np.concatenate([a[:, -P:], a, a[:, :P]], axis=1)   # marge circulaire
            img, sm = wrap(img), wrap(sm)
        else:
            x_start, wl = w.x0 - P, w.w + 2 * P
            img = np.zeros((ye - ys, wl, 3), np.float32)
            sm = np.zeros((ye - ys, wl), np.float32)
            va = np.zeros((ye - ys, wl), bool)
            r, c = slice(w.y0 - ys, w.y0 - ys + w.h), slice(P, P + w.w)
            img[r, c], sm[r, c], va[r, c] = w.image, seam, w.mask > 0
            img = _extend_edges(img, va)
        lap, wp = _pyramids(img, sm, bands)
        if full:                                          # retire la marge circulaire à chaque niveau
            lap = [a[:, P >> k:a.shape[1] - (P >> k)] for k, a in enumerate(lap)]
            wp = [a[:, P >> k:a.shape[1] - (P >> k)] for k, a in enumerate(wp)]
        return ys, x_start, lap, wp

    with ThreadPoolExecutor(workers) as ex:
        for ys, x_start, lap, wp in ex.map(prepare, zip(warped, seams)):
            for k in range(bands + 1):
                rows = slice(ys >> k, (ys >> k) + lap[k].shape[0])
                cols = ((x_start >> k) + np.arange(lap[k].shape[1])) % (W >> k)
                num[k][rows, cols] += lap[k] * wp[k][..., None]
                den[k][rows, cols] += wp[k]

    # Reconstruction : on normalise chaque niveau puis on remonte la pyramide, toujours en circulaire.
    levels = []
    for k in range(bands + 1):
        lvl = num[k] / np.maximum(den[k], 1e-6)[..., None]
        p = P >> k
        levels.append(np.concatenate([lvl[:, -p:], lvl, lvl[:, :p]], axis=1))
    res = levels[bands]
    for k in range(bands - 1, -1, -1):
        res = cv2.pyrUp(res, dstsize=(levels[k].shape[1], levels[k].shape[0])) + levels[k]
    res = res[:, P:P + W]
    cov = coverage_mask(warped, W, H)
    pano = np.clip(res, 0, 255).astype(np.uint8)
    pano[~cov] = 0
    return pano, cov


def blend(warped, seams, cfg):
    """Fusionne les images projetées selon la méthode configurée. Renvoie (panorama, couverture)."""
    if cfg.blend == "linear":
        return blend_linear(warped, seams, cfg.width, cfg.height)
    return blend_multiband(warped, seams, cfg.width, cfg.height, cfg.bands, cfg.workers)
