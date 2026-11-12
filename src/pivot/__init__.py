"""Pivot : assemblage d'images en panorama sphérique 360° équirectangulaire.

    from pivot import PivotConfig, run_pipeline
    result = run_pipeline(PivotConfig(input="./dataset", output="pano.jpg"))
"""
from .config import PivotConfig
from .models import PivotError, StitchReport, StitchResult
from .pipeline import run_pipeline

__version__ = "0.1.0"
__all__ = ["PivotConfig", "PivotError", "StitchReport", "StitchResult", "run_pipeline", "__version__"]
