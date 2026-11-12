"""Visualisations et journal de diagnostic (option --debug), et chronométrage des étapes."""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager

import cv2
import numpy as np

log = logging.getLogger("pivot")


@contextmanager
def timed(report, step: str):
    """Mesure la durée d'une étape et l'enregistre dans le rapport."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    report.timings[step] = round(report.timings.get(step, 0.0) + dt, 3)
    log.info("%-28s %.2f s", step, dt)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _palette(n: int) -> np.ndarray:
    hsv = np.zeros((max(n, 1), 1, 3), np.uint8)
    hsv[:, 0, 0] = (np.arange(max(n, 1)) * 179 * 0.618) % 180   # teintes bien séparées
    hsv[:, 0, 1], hsv[:, 0, 2] = 200, 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]


class DebugWriter:
    """Écrit les images et métriques de diagnostic. Toutes les méthodes sont sans effet si désactivé."""

    def __init__(self, enabled: bool, folder: str):
        self.enabled = enabled
        self.folder = folder
        if enabled:
            for sub in ("keypoints", "matches", "masks"):
                os.makedirs(os.path.join(folder, sub), exist_ok=True)
            handler = logging.FileHandler(os.path.join(folder, "log.txt"), mode="w", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            log.addHandler(handler)

    def _write(self, rel: str, img) -> None:
        cv2.imwrite(os.path.join(self.folder, rel), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def keypoints(self, images, features) -> None:
        """Points d'intérêt détectés sur chaque image."""
        if not self.enabled:
            return
        for im, ft in zip(images, features):
            vis = im.work.copy()
            for x, y in ft.pts.astype(int):
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.putText(vis, f"{len(ft.pts)} points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            self._write(f"keypoints/{os.path.splitext(im.name)[0]}.jpg", vis)

    def matches(self, images, matches, n: int) -> None:
        """Appariements des paires voisines : avant RANSAC (rouge) et après (vert)."""
        if not self.enabled:
            return
        for m in matches:
            if min((m.j - m.i) % n, (m.i - m.j) % n) != 1:
                continue
            a, b = images[m.i].work, images[m.j].work
            off = a.shape[1]
            for tag, sel, color in (("avant", np.ones(m.n_matches, bool), (0, 0, 255)),
                                    ("apres", m.inliers, (0, 255, 0))):
                vis = np.hstack([a, b])
                for (x1, y1), (x2, y2) in zip(m.pts_i[sel].astype(int), m.pts_j[sel].astype(int)):
                    cv2.line(vis, (int(x1), int(y1)), (int(x2) + off, int(y2)), color, 1, cv2.LINE_AA)
                label = f"{m.i}-{m.j} {tag} RANSAC : {int(sel.sum())}" + ("" if m.valid else " (paire rejetee)")
                cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                self._write(f"matches/{m.i:03d}_{m.j:03d}_{tag}.jpg", vis)

    def graph(self, matches, n: int, kept) -> None:
        """Graphe de connexion : un sommet par image, épaisseur des arêtes selon le nombre d'inliers."""
        if not self.enabled:
            return
        size = 1000
        img = np.full((size, size, 3), 255, np.uint8)
        ang = 2 * np.pi * np.arange(n) / n - np.pi / 2
        pts = np.stack([size / 2 + 0.42 * size * np.cos(ang), size / 2 + 0.42 * size * np.sin(ang)], 1).astype(int)
        top = max([m.n_inliers for m in matches if m.valid] or [1])
        for m in matches:
            p, q = tuple(map(int, pts[m.i])), tuple(map(int, pts[m.j]))
            if m.valid:
                cv2.line(img, p, q, (60, 140, 60), 1 + int(5 * m.n_inliers / top), cv2.LINE_AA)
            else:
                cv2.line(img, p, q, (200, 200, 235), 1, cv2.LINE_AA)
        kept = set(kept)
        for i in range(n):
            c = tuple(map(int, pts[i]))
            cv2.circle(img, c, 16, (255, 200, 120) if i in kept else (120, 120, 255), -1, cv2.LINE_AA)
            cv2.circle(img, c, 16, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(i), (c[0] - 10, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "vert : paire valide | rose : paire rejetee | rouge : image ecartee",
                    (20, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.folder, "graph.png"), img)

    def masks(self, warped, seams, coverage) -> None:
        """Masque de couverture, nombre d'images par pixel, et carte des coutures (une couleur par image)."""
        if not self.enabled:
            return
        H, W = coverage.shape
        cv2.imwrite(os.path.join(self.folder, "masks", "coverage.png"), coverage.astype(np.uint8) * 255)
        count = np.zeros((H, W), np.uint8)
        label = np.zeros((H, W, 3), np.uint8)
        pal = _palette(len(warped))
        for k, (w, s) in enumerate(zip(warped, seams)):
            rows, cols = slice(w.y0, w.y0 + w.h), w.cols(W)
            count[rows, cols] += (w.mask > 0).astype(np.uint8)
            cur = label[rows, cols]
            cur[s] = pal[k]
            label[rows, cols] = cur
        vis = cv2.applyColorMap(np.minimum(count * 50, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        vis[count == 0] = 0
        cv2.imwrite(os.path.join(self.folder, "masks", "overlap_count.png"), vis)
        cv2.imwrite(os.path.join(self.folder, "masks", "seams.png"), label)

    def metrics(self, report) -> None:
        """Écrit metrics.json : inliers par paire, focale, erreurs de reprojection, temps par étape."""
        if not self.enabled:
            return
        with open(os.path.join(self.folder, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(report.to_dict()), f, indent=2, ensure_ascii=False)
