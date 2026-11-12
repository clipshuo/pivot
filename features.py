"""Étapes 2 et 3 : détection des points d'intérêt et appariement entre images."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import geometry
from models import Features, ImageData, PairMatch, PivotError


def create_detector(name: str, max_features: int) -> cv2.Feature2D:
    """SIFT (robuste, par défaut) ou ORB (binaire, plus rapide, moins précis)."""
    if name == "sift":
        return cv2.SIFT_create(nfeatures=max_features)
    if name == "orb":
        return cv2.ORB_create(nfeatures=max_features)
    raise PivotError(f"Détecteur inconnu : {name}")


def detect_features(images: list[ImageData], cfg) -> list[Features]:
    """Détecte les points d'intérêt sur les copies de travail, une image par thread."""
    def one(im: ImageData) -> Features:
        gray = cv2.cvtColor(im.work, cv2.COLOR_BGR2GRAY)
        det = create_detector(cfg.detector, cfg.max_features)   # un détecteur par thread
        kps, desc = det.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kps]).reshape(-1, 2)
        if desc is None:
            desc = np.zeros((0, 128 if cfg.detector == "sift" else 32),
                            np.float32 if cfg.detector == "sift" else np.uint8)
        return Features(pts, desc)

    with ThreadPoolExecutor(cfg.workers) as ex:
        return list(ex.map(one, images))


def candidate_pairs(n: int, ordered: bool, neighbor_range: int) -> list[tuple[int, int]]:
    """Paires à comparer.

    Ordonné : chaque image avec ses neighbor_range suivantes, modulo n, ce qui
    ferme la boucle (dernière <-> première). Non ordonné : toutes les paires.
    """
    pairs = set()
    if ordered:
        for i in range(n):
            for d in range(1, neighbor_range + 1):
                j = (i + d) % n
                if i != j:
                    pairs.add((min(i, j), max(i, j)))
    else:
        pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    return sorted(pairs)


def match_pair(fa: Features, fb: Features, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Apparie deux images : 2 plus proches voisins puis test de ratio de Lowe.

    Renvoie les coordonnées appariées (M, 2) dans chaque image.
    """
    empty = np.zeros((0, 2), np.float32)
    if len(fa.desc) < 2 or len(fb.desc) < 2:
        return empty, empty
    if cfg.detector == "sift":
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    best = {}                                    # un seul appariement par point de l'image b
    for pair in matcher.knnMatch(fa.desc, fb.desc, k=2):
        if len(pair) == 2 and pair[0].distance < cfg.ratio * pair[1].distance:
            m = pair[0]
            if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
                best[m.trainIdx] = m
    if not best:
        return empty, empty
    ms = list(best.values())
    return (fa.pts[[m.queryIdx for m in ms]], fb.pts[[m.trainIdx for m in ms]])


def match_all(features: list[Features], pairs, cfg, center: tuple[float, float]) -> list[PairMatch]:
    """Apparie toutes les paires candidates en parallèle et filtre chacune par RANSAC.

    L'homographie est estimée en coordonnées centrées sur le point principal :
    c'est la forme attendue par l'estimation de focale.
    """
    c = np.float32(center)

    def one(pair) -> PairMatch:
        i, j = pair
        pi, pj = match_pair(features[i], features[j], cfg)
        m = PairMatch(i, j, pi, pj, np.zeros(len(pi), bool))
        if len(pi) >= 8:
            H, inl = geometry.ransac_homography(pi - c, pj - c, cfg.ransac_thresh)
            if H is not None:
                m.H, m.inliers = H, inl
                m.n_ransac = int(inl.sum())
                m.confidence = m.n_ransac / (8.0 + 0.3 * m.n_matches)
                m.valid = m.n_ransac >= cfg.min_inliers and m.confidence >= cfg.min_confidence
        return m

    with ThreadPoolExecutor(cfg.workers) as ex:
        return list(ex.map(one, pairs))
