"""Ligne de commande de Pivot : assemblage d'un panorama sphérique 360° équirectangulaire.

Exemple :
    pivot --input ./dataset --output pano.jpg --width 4096 \
          --detector sift --blend multiband --fill inpaint --debug

Sans installation : PYTHONPATH=src python -m pivot --input ./dataset
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import config_from_args
from .models import PivotError
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pivot", description="Assemble un dossier d'images en panorama équirectangulaire 360x180°.")
    p.add_argument("--input", required=True, help="dossier d'images (jpg, png), triées par nom")
    p.add_argument("--output", default="pano.jpg", help="image de sortie (JPEG recommandé)")
    p.add_argument("--width", type=int, default=4096, help="largeur de sortie ; hauteur = largeur / 2")
    p.add_argument("--detector", choices=["sift", "orb"], default="sift")
    p.add_argument("--ratio", type=float, default=0.75, help="seuil du test de ratio de Lowe")
    p.add_argument("--unordered", action="store_true", help="ordre des images inconnu : compare toutes les paires")
    p.add_argument("--neighbor-range", type=int, default=2, help="mode ordonné : nombre de voisines comparées")
    p.add_argument("--focal", type=float, default=None, help="focale imposée, en pixels pleine résolution")
    p.add_argument("--work-mp", type=float, default=1.0, help="taille de la copie de travail, en mégapixels")
    p.add_argument("--no-distortion", action="store_true", help="n'estime pas la distorsion radiale")
    p.add_argument("--no-sensors", action="store_true",
                   help="ignore capture.json (orientations du téléphone) même s'il est présent")
    p.add_argument("--no-wave", action="store_true", help="désactive la correction d'horizon")
    p.add_argument("--no-expo", action="store_true", help="désactive la compensation d'exposition")
    p.add_argument("--seam", choices=["voronoi", "graphcut"], default="voronoi")
    p.add_argument("--blend", choices=["multiband", "linear"], default="multiband")
    p.add_argument("--bands", type=int, default=5, help="niveaux de la fusion multi-bandes (1 à 6)")
    p.add_argument("--fill", choices=["none", "inpaint", "color", "gradient"], default="inpaint")
    p.add_argument("--fill-color", type=int, nargs=3, default=[0, 0, 0], metavar=("B", "G", "R"))
    p.add_argument("--quality", type=int, default=95, help="qualité JPEG")
    p.add_argument("--workers", type=int, default=0, help="threads ; 0 = tous les cœurs disponibles")
    p.add_argument("--baseline", action="store_true", help="assemble avec cv2.Stitcher (référence de comparaison)")
    p.add_argument("--debug", action="store_true", help="écrit visualisations et métriques dans --debug-dir")
    p.add_argument("--debug-dir", default="debug")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    try:
        result = run_pipeline(config_from_args(args))
    except PivotError as e:
        logging.getLogger("pivot").error("Échec : %s", e)
        return 2
    n = len(result.report.warnings)
    if n:
        logging.getLogger("pivot").info("%d avertissement(s) : voir ci-dessus%s.", n,
                                        " et debug/metrics.json" if args.debug else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
