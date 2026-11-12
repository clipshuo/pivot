"""Étapes 4 à 6 : géométrie par paire, focale, ajustement de faisceaux, correction d'horizon.

Modèle : rotation pure autour du centre optique. Entre deux images,
H_ij = K · R_j · R_iᵀ · K⁻¹. L'homographie libre (8 degrés de liberté) ne sert
qu'à trier les inliers et à initialiser la focale ; le modèle final n'utilise
que des rotations et une focale partagée.

Dans ce module, les points sont en coordonnées CENTRÉES (origine au point principal).
"""
from __future__ import annotations

import heapq
import logging

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from .models import (BundleAdjustmentError, Camera, DisconnectedGraphError,
                    InsufficientOverlapError, PairMatch)

log = logging.getLogger("pivot")


# ------------------------------------------------------------------ étape 4 : par paire
def ransac_homography(pts_i: np.ndarray, pts_j: np.ndarray, thresh: float):
    """Homographie i -> j par RANSAC. Renvoie (H, masque booléen des inliers) ou (None, None)."""
    H, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, thresh, maxIters=2000, confidence=0.999)
    if H is None or not np.isfinite(H).all():
        return None, None
    return H, mask.ravel().astype(bool)


def focal_from_homography(H: np.ndarray) -> float | None:
    """Focale déduite d'une homographie de rotation pure (Szeliski & Shum, repris par Brown & Lowe).

    Si H = K₁ · R · K₀⁻¹ avec K = diag(f, f, 1), l'orthonormalité des lignes et des
    colonnes de R donne deux estimations de f² pour chaque caméra. On renvoie la
    moyenne géométrique des deux focales, ou None si H n'est pas compatible.
    """
    h = (H / H[2, 2]).ravel()

    def pick(v1, v2, d1, d2):
        if v1 < v2:
            v1, v2, d1, d2 = v2, v1, d2, d1
        if v1 > 0 and v2 > 0:
            return np.sqrt(v1 if abs(d1) > abs(d2) else v2)
        return np.sqrt(v1) if v1 > 0 else None

    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = h[6] * h[7], (h[7] - h[6]) * (h[7] + h[6])
        f1 = pick(-(h[0] * h[1] + h[3] * h[4]) / d1,
                  (h[0] ** 2 + h[3] ** 2 - h[1] ** 2 - h[4] ** 2) / d2, d1, d2)
        d1, d2 = h[0] * h[3] + h[1] * h[4], h[0] ** 2 + h[1] ** 2 - h[3] ** 2 - h[4] ** 2
        f0 = pick(-h[2] * h[5] / d1, (h[5] ** 2 - h[2] ** 2) / d2, d1, d2)
    if f0 is None or f1 is None or not np.isfinite([f0, f1]).all():
        return None
    return float(np.sqrt(f0 * f1))


def estimate_initial_focal(matches: list[PairMatch], images, cfg, report) -> float:
    """Focale initiale à l'échelle de travail : imposée, sinon EXIF, sinon médiane des homographies."""
    scale = images[0].scale
    width = images[0].work.shape[1]
    if cfg.focal:
        report.metrics["focale_source"] = "imposée"
        return cfg.focal * scale
    exif = [im.exif_focal_px for im in images if im.exif_focal_px]
    if exif:
        report.metrics["focale_source"] = "exif"
        return float(np.median(exif)) * scale
    focals = [f for f in (focal_from_homography(m.H) for m in matches if m.valid)
              if f is not None and 0.3 * width < f < 15 * width]
    if focals:
        report.metrics["focale_source"] = f"homographies ({len(focals)} paires)"
        return float(np.median(focals))
    report.metrics["focale_source"] = "défaut"
    report.warn("Focale non estimable depuis les homographies : valeur par défaut (largeur de l'image), "
                "l'ajustement de faisceaux devra la corriger. Résultat à vérifier.")
    return float(width)


def rotation_from_matches(pts_i: np.ndarray, pts_j: np.ndarray, focal: float) -> np.ndarray:
    """Rotation relative R_ij = R_j · R_iᵀ par l'algorithme de Kabsch sur les rayons.

    Chaque point devient un rayon unitaire (x, y, f) ; on cherche la rotation qui
    aligne au mieux les rayons de i sur ceux de j (SVD de la matrice de corrélation).
    """
    def rays(p):
        r = np.column_stack([p, np.full(len(p), focal)]).astype(np.float64)
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    a, b = rays(pts_i), rays(pts_j)
    U, _, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])   # impose det = +1
    return U @ D @ Vt


def transfer_error(R: np.ndarray, focal: float, pts_i: np.ndarray, pts_j: np.ndarray) -> np.ndarray:
    """Erreur (px) entre les points de j et ceux de i reprojetés par la rotation R."""
    q = np.column_stack([pts_i, np.full(len(pts_i), focal)]) @ R.T
    z = np.where(np.abs(q[:, 2:3]) < 1e-9, 1e-9, q[:, 2:3])
    return np.linalg.norm(focal * q[:, :2] / z - pts_j, axis=1)


def refine_pairs(matches: list[PairMatch], focal: float, center, cfg) -> None:
    """Remplace l'homographie de chaque paire valide par une vraie rotation.

    Les inliers de l'homographie incompatibles avec la rotation pure sont rejetés :
    c'est là que la parallaxe (translation de la caméra) devient visible.
    """
    c = np.float64(center)
    for m in matches:
        if not m.valid:
            continue
        pi, pj = m.pts_i - c, m.pts_j - c
        inl = m.inliers.copy()
        for _ in range(2):
            R = rotation_from_matches(pi[inl], pj[inl], focal)
            err = transfer_error(R, focal, pi, pj)
            new = m.inliers & (err < 2.0 * cfg.ransac_thresh)
            if new.sum() < 8:
                break
            inl = new
        m.R = rotation_from_matches(pi[inl], pj[inl], focal)
        m.inliers = inl
        m.rms = float(np.sqrt(np.mean(transfer_error(m.R, focal, pi[inl], pj[inl]) ** 2)))
        m.valid = m.n_inliers >= cfg.min_inliers


def update_pair_rotations(matches: list[PairMatch], cam: Camera, center) -> None:
    """Ré-estime la rotation de chaque paire valide avec la focale et la distorsion finales.

    Les rotations par paire de l'étape 4 sont calculées avec la focale initiale. Si
    l'ajustement de faisceaux la modifie, elles sont biaisées d'autant : pour mesurer
    l'écart de boucle après ajustement, il faut les recalculer avec le modèle final.
    """
    c = np.float64(center)
    for m in matches:
        if not m.valid:
            continue
        pi = np.column_stack(cam.undistort(*(m.pts_i[m.inliers] - c).T))
        pj = np.column_stack(cam.undistort(*(m.pts_j[m.inliers] - c).T))
        m.R = rotation_from_matches(pi, pj, cam.focal)
        m.rms = float(np.sqrt(np.mean(transfer_error(m.R, cam.focal, pi, pj) ** 2)))


# ------------------------------------------------------------------ graphe de connexion
def build_graph(matches: list[PairMatch], n: int) -> dict[int, dict[int, PairMatch]]:
    """Graphe non orienté : un sommet par image, une arête par paire valide."""
    graph = {i: {} for i in range(n)}
    for m in matches:
        if m.valid:
            graph[m.i][m.j] = m
            graph[m.j][m.i] = m
    return graph


def largest_component(graph) -> list[int]:
    """Plus grand ensemble d'images reliées entre elles."""
    seen, best = set(), []
    for start in graph:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in graph[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(comp) > len(best):
            best = comp
    if len(best) < 2:
        raise InsufficientOverlapError(
            "Aucune paire d'images n'a assez de correspondances fiables : recouvrement insuffisant, "
            "images floues ou scène sans texture.")
    return sorted(best)


def initial_rotations(graph, nodes: list[int], ref: int) -> dict[int, np.ndarray]:
    """Rotations initiales en chaînant les rotations relatives le long d'un arbre couvrant maximal.

    L'arbre (algorithme de Prim, poids = nombre d'inliers) part de l'image de
    référence, dont la rotation est l'identité. Les arêtes hors de l'arbre (dont la
    fermeture de boucle) ne sont pas utilisées ici : la dérive accumulée est
    corrigée ensuite par l'ajustement de faisceaux.
    """
    allowed = set(nodes)
    R = {ref: np.eye(3)}
    heap = [(-m.n_inliers, ref, v) for v, m in graph[ref].items() if v in allowed]
    heapq.heapify(heap)
    while heap and len(R) < len(nodes):
        _, u, v = heapq.heappop(heap)
        if v in R:
            continue
        m = graph[u][v]
        R[v] = m.R @ R[u] if (m.i, m.j) == (u, v) else m.R.T @ R[u]
        for w, mw in graph[v].items():
            if w in allowed and w not in R:
                heapq.heappush(heap, (-mw.n_inliers, v, w))
    if len(R) < len(nodes):
        raise DisconnectedGraphError("Graphe de connexion incohérent : images non reliées à la référence.")
    return R


# ------------------------------------------------------------------ métriques
def _observations(cameras, matches, center, max_points=None):
    """Aplatit les inliers de toutes les paires valides en tableaux (indices, points centrés)."""
    c = np.float64(center)
    rng = np.random.default_rng(0)
    ii, jj, pi, pj = [], [], [], []
    for m in matches:
        if not m.valid or m.i not in cameras or m.j not in cameras:
            continue
        idx = np.flatnonzero(m.inliers)
        if max_points and len(idx) > max_points:
            idx = rng.choice(idx, max_points, replace=False)
        ii.append(np.full(len(idx), m.i))
        jj.append(np.full(len(idx), m.j))
        pi.append(m.pts_i[idx] - c)
        pj.append(m.pts_j[idx] - c)
    return np.concatenate(ii), np.concatenate(jj), np.vstack(pi), np.vstack(pj)


def reprojection_errors(cameras: dict, matches, center) -> np.ndarray:
    """Erreur de reprojection (px) de chaque inlier, pour l'état courant des caméras."""
    ii, jj, pi, pj = _observations(cameras, matches, center)
    cam = next(iter(cameras.values()))
    Ri = np.stack([cameras[k].R for k in ii])
    Rj = np.stack([cameras[k].R for k in jj])
    r = _residuals_and_jacobian(cam.focal, cam.k1, cam.rnorm, Ri, Rj, pi, pj)[0]
    return np.linalg.norm(r[:, :2], axis=1)


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def edge_consistency_deg(cameras: dict, matches) -> np.ndarray:
    """Écart angulaire, par arête, entre la rotation mesurée sur la paire et celle des caméras.

    Avant l'ajustement de faisceaux, le maximum donne l'erreur de fermeture de
    boucle (la dérive accumulée sur le tour).
    """
    out = [rotation_angle_deg(m.R.T @ cameras[m.j].R @ cameras[m.i].R.T)
           for m in matches if m.valid and m.i in cameras and m.j in cameras]
    return np.array(out)


# ------------------------------------------------------------------ étape 5 : ajustement de faisceaux
def _skew(v: np.ndarray) -> np.ndarray:
    """Matrices antisymétriques [v]× pour un lot de vecteurs (M, 3) : [v]× w = v × w."""
    S = np.zeros((len(v), 3, 3))
    S[:, 0, 1], S[:, 0, 2] = -v[:, 2], v[:, 1]
    S[:, 1, 0], S[:, 1, 2] = v[:, 2], -v[:, 0]
    S[:, 2, 0], S[:, 2, 1] = -v[:, 1], v[:, 0]
    return S


def _residuals_and_jacobian(f, k, norm, Ri, Rj, pi, pj):
    """Résidus (M, 4) et leurs dérivées par rapport à f, k, à la caméra i et à la caméra j.

    Distorsion radiale : un point observé p (centré) correspond au point idéal
    p · (1 + k · s), avec s = |p|² / norm². Les résidus sont mesurés entre points idéaux.
    Une rotation est perturbée à gauche : R <- exp([ω]×) · R, avec ω petit.
    Pour r1 = f · π(q) - p_j idéal, où q = R_j · R_iᵀ · a et a = (x_i, y_i, f) idéal :
        dq/dω_j = -[q]×      dq/dω_i = R_j R_iᵀ [a]×      dq/df = 3e colonne de R_j R_iᵀ
    r2 est le cas symétrique (j -> i). π est la division par z.
    """
    M = len(pi)
    Rrel = Rj @ Ri.transpose(0, 2, 1)
    fcol = np.full((M, 1), f)
    si = (pi ** 2).sum(axis=1, keepdims=True) / norm ** 2
    sj = (pj ** 2).sum(axis=1, keepdims=True) / norm ** 2
    ui, uj = pi * (1 + k * si), pj * (1 + k * sj)        # points idéaux (sans distorsion)

    def half(Rr, src, dst, dsrc_dk, ddst_dk):
        a = np.hstack([src, fcol])
        q = np.einsum("mab,mb->ma", Rr, a)
        z = np.where(np.abs(q[:, 2]) < 1e-9, 1e-9, q[:, 2])
        u = q[:, :2] / z[:, None]
        Jpi = np.zeros((M, 2, 3))                         # dérivée de π(q) = (qx/qz, qy/qz)
        Jpi[:, 0, 0] = Jpi[:, 1, 1] = 1.0 / z
        Jpi[:, 0, 2], Jpi[:, 1, 2] = -u[:, 0] / z, -u[:, 1] / z
        FJ = f * Jpi
        J_dst = FJ @ (-_skew(q))                          # caméra d'arrivée
        J_src = FJ @ (Rr @ _skew(a))                      # caméra de départ
        J_f = u + np.einsum("mab,mb->ma", FJ, Rr[:, :, 2])
        J_k = np.einsum("mab,mb->ma", FJ @ Rr[:, :, :2], dsrc_dk) - ddst_dk
        return f * u - dst, J_f, J_k, J_src, J_dst

    r1, Jf1, Jk1, Ji1, Jj1 = half(Rrel, ui, uj, pi * si, pj * sj)
    r2, Jf2, Jk2, Jj2, Ji2 = half(Rrel.transpose(0, 2, 1), uj, ui, pj * sj, pi * si)
    return (np.hstack([r1, r2]), np.hstack([Jf1, Jf2]), np.hstack([Jk1, Jk2]),
            np.concatenate([Ji1, Ji2], axis=1), np.concatenate([Jj1, Jj2], axis=1))


def bundle_adjust(cameras: dict, matches, ref: int, center, cfg, fit_distortion=None,
                  priors: dict | None = None, prior_sigma_deg: float = 1.5,
                  fit_focal: bool = True) -> tuple[dict, dict]:
    """Optimise conjointement toutes les rotations et la focale partagée.

    Inconnues : la focale f, le coefficient de distorsion radiale k (si fit_distortion, par défaut
    cfg.distortion), et 3 paramètres de rotation par caméra sauf la référence, qui reste fixe (sinon le problème est défini à une rotation
    globale près). Coût : erreurs de reprojection symétriques sur tous les
    inliers, avec une perte de Huber qui limite le poids des faux appariements.

    Méthode : Levenberg-Marquardt. À chaque itération on linéarise autour des
    rotations courantes (jacobien analytique, creux), on résout les équations
    normales amorties, et on applique R <- exp([ω]×) · R. C'est cette étape
    qui répartit sur toutes les images la dérive accumulée et ferme la boucle.

    `priors` (caméra -> rotation mesurée par le téléphone) ajoute un terme qui retient chaque
    rotation près de sa mesure : coût 0,5 · |ω|² / σ², où ω est l'écart à la mesure (radians) et
    σ = prior_sigma_deg. Il fixe les images sans appariement et limite l'effet de la parallaxe.
    `fit_focal=False` garde la focale donnée (celle du téléphone) : avec peu d'appariements et de
    la parallaxe, la laisser libre la fait dériver.
    """
    ids = sorted(cameras)
    pos = {c: k for k, c in enumerate(ids)}
    free = [c for c in ids if c != ref]
    col = np.full(len(ids), -1)
    col[[pos[c] for c in free]] = 2 + 3 * np.arange(len(free))
    ii, jj, pi, pj = _observations(cameras, matches, center, cfg.ba_max_points)
    pii, pjj = np.array([pos[c] for c in ii]), np.array([pos[c] for c in jj])
    M, P = len(ii), 2 + 3 * len(free)
    R = np.stack([cameras[c].R for c in ids])
    f, k, norm = float(cameras[ref].focal), float(cameras[ref].k1), float(cameras[ref].rnorm)
    fit_k = bool(cfg.distortion if fit_distortion is None else fit_distortion)
    delta = float(cfg.ransac_thresh)                      # seuil de Huber, en pixels

    # Indices du jacobien creux (constants) : 4 lignes par observation, 7 colonnes non nulles.
    row = 4 * np.arange(M)[:, None] + np.arange(4)[None, :]                       # (M, 4)
    row3 = np.broadcast_to(row[:, :, None], (M, 4, 3))
    ci, cj = col[pii], col[pjj]
    col_i = ci[:, None, None] + np.arange(3)[None, None, :] + np.zeros((M, 4, 1), int)
    col_j = cj[:, None, None] + np.arange(3)[None, None, :] + np.zeros((M, 4, 1), int)
    keep_i = np.broadcast_to((ci >= 0)[:, None, None], (M, 4, 3))
    keep_j = np.broadcast_to((cj >= 0)[:, None, None], (M, 4, 3))
    rows = np.concatenate([row.ravel(), row.ravel(), row3[keep_i], row3[keep_j]])
    cols = np.concatenate([np.zeros(4 * M, int), np.ones(4 * M, int), col_i[keep_i], col_j[keep_j]])

    w_prior = 1.0 / np.radians(prior_sigma_deg) ** 2
    R_prior = np.stack([priors[c] for c in free]) if priors and free else None

    def prior_terms(R):
        """Écart ω de chaque caméra libre à sa mesure, et coût associé."""
        if R_prior is None:
            return None, 0.0
        om = Rotation.from_matrix(R[col >= 0] @ R_prior.transpose(0, 2, 1)).as_rotvec()
        return om, 0.5 * w_prior * float((om ** 2).sum())

    def evaluate(f, k, R):
        r, Jf, Jk, Ji, Jj = _residuals_and_jacobian(f, k, norm, R[pii], R[pjj], pi, pj)
        if not fit_k:
            Jk = np.zeros_like(Jk)
        if not fit_focal:
            Jf = np.zeros_like(Jf)
        # Huber sur la norme de chaque erreur 2D : poids 1 sous le seuil, delta / |e| au-dessus.
        e = np.sqrt((r.reshape(M, 2, 2) ** 2).sum(axis=2))                        # (M, 2)
        cost = float(np.where(e <= delta, 0.5 * e ** 2, delta * (e - 0.5 * delta)).sum())
        w = np.repeat(np.where(e <= delta, 1.0, delta / np.maximum(e, 1e-12)), 2, axis=1).ravel()
        om, cost_prior = prior_terms(R)
        return r.ravel(), (Jf, Jk, Ji, Jj, om), w, cost + cost_prior

    r, jac, w, cost = evaluate(f, k, R)
    cost0, lam, it = cost, 1e-3, 0
    for it in range(1, 101):
        Jf, Jk, Ji, Jj, om = jac
        J = coo_matrix((np.concatenate([Jf.ravel(), Jk.ravel(), Ji[keep_i], Jj[keep_j]]), (rows, cols)),
                       shape=(4 * M, P)).tocsr()
        JW = J.multiply(w[:, None]).tocsr()
        A = (J.T @ JW).toarray()
        g = J.T @ (w * r)
        if om is not None:                                # a priori : dω/dδ ≈ I pour un petit écart
            A[np.arange(2, P), np.arange(2, P)] += w_prior
            g[2:] += w_prior * om.ravel()
        improved = False
        while lam < 1e12:
            try:
                D = np.diag(A).copy()
                D[D <= 0] = 1.0                           # paramètre gelé (k non estimé)
                frozen = np.zeros(P)                      # paramètres gelés : pivot unitaire, pas nul
                frozen[0], frozen[1] = float(not fit_focal), float(not fit_k)
                dx = np.linalg.solve(A + lam * np.diag(D) + np.diag(frozen), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            f_new, k_new = f + (dx[0] if fit_focal else 0.0), k + (dx[1] if fit_k else 0.0)
            R_new = R.copy()
            R_new[col >= 0] = Rotation.from_rotvec(dx[2:].reshape(-1, 3)).as_matrix() @ R[col >= 0]
            r_new, jac_new, w_new, cost_new = evaluate(f_new, k_new, R_new)
            if f_new > 0 and cost_new < cost:
                improved = True
                break
            lam *= 5.0
        if not improved:
            break
        gain = (cost - cost_new) / max(cost, 1e-12)
        f, k, R, r, jac, w, cost = f_new, k_new, R_new, r_new, jac_new, w_new, cost_new
        lam = max(lam / 5.0, 1e-9)
        if gain < 1e-9:
            break

    width = 2 * center[0] + 1
    if not np.isfinite(R).all() or not (0.2 * width < f < 20 * width):
        raise BundleAdjustmentError(
            f"L'ajustement de faisceaux a divergé (focale {f:.1f} px). Causes probables : "
            "parallaxe forte, faux appariements, ou scène incompatible avec une rotation pure.")
    out = {c: Camera(R[pos[c]], float(f), cameras[c].cx, cameras[c].cy, float(k), norm) for c in ids}
    return out, {"iterations": it, "observations": int(M),
                 "cout_initial": round(cost0, 2), "cout_final": round(cost, 2)}


# ------------------------------------------------------------------ étape 6 : correction d'horizon
def wave_correct(cameras: dict, ref: int, down: np.ndarray | None = None) -> dict:
    """Redresse le panorama pour que l'horizon soit une ligne droite.

    Idée de Brown & Lowe : on tient rarement la caméra en roulis, donc les axes x
    de toutes les caméras restent dans un même plan, le plan horizontal. Sa normale
    (vecteur propre de plus petite valeur propre de Σ xᵢ xᵢᵀ) est la verticale.
    On construit un nouveau repère monde dont l'axe y est cette verticale (vers le
    bas) et dont l'axe z suit la direction de visée de l'image de référence, qui
    se retrouve ainsi au centre du panorama.

    Si `down` est fourni (verticale mesurée par le téléphone), elle est utilisée telle quelle.
    """
    if down is None:
        X = np.stack([c.R[0] for c in cameras.values()]) # axes x des caméras, exprimés dans le monde
        _, vecs = np.linalg.eigh(X.T @ X)
        down = vecs[:, 0]
        if sum(float(down @ c.R[1]) for c in cameras.values()) < 0:   # même sens que les axes y (bas)
            down = -down
    forward = cameras[ref].R[2]
    forward = forward - float(forward @ down) * down     # composante horizontale de la visée
    right = np.cross(down, forward)
    if np.linalg.norm(right) < 1e-6:                     # référence pointée vers le zénith ou le nadir
        return cameras
    right /= np.linalg.norm(right)
    G = np.stack([right, down, np.cross(right, down)])    # lignes = nouveaux axes x, y, z
    return {k: Camera(c.R @ G.T, c.focal, c.cx, c.cy, c.k1, c.rnorm) for k, c in cameras.items()}
