"""Bout en bout sur un tour synthétique dont on connaît la vérité, et cas d'échec."""
import json
import os

import cv2
import numpy as np
import pytest

from pivot import geometry
from pivot import cli
from pivot.config import PivotConfig
from pivot.models import NotEnoughImagesError, PivotError
from pivot.pipeline import run_pipeline


@pytest.fixture(scope="module")
def result(synthetic_dataset, tmp_path_factory):
    folder, focal, step = synthetic_dataset
    out = tmp_path_factory.mktemp("out")
    cfg = PivotConfig(input=folder, output=str(out / "pano.jpg"), width=1024,
                      debug=True, debug_dir=str(out / "debug"))
    return run_pipeline(cfg), cfg, focal, step


def test_tour_complet_synthetique(result):
    res, cfg, focal, step = result
    m = res.report.metrics
    assert len(res.cameras) == 12 and m["boucle_fermee"]
    assert m["focale"]["px_pleine_resolution"] == pytest.approx(focal, rel=0.02)
    assert m["ajustement_de_faisceaux"]["erreur_moyenne_apres_px"] < 1.0
    assert m["couverture"]["horizontale_deg"] == 360.0
    assert not any("Boucle non fermée" in w or "Tour incomplet" in w for w in res.report.warnings)
    for k in range(12):                                                # 30° entre images consécutives
        rel = res.cameras[(k + 1) % 12].R @ res.cameras[k].R.T
        assert geometry.rotation_angle_deg(rel) == pytest.approx(step, abs=0.5)


def test_sortie_equirectangulaire_fidele(result, texture):
    res, cfg, _, _ = result
    data = open(cfg.output, "rb").read()
    assert b'GPano:ProjectionType="equirectangular"' in data
    pano = cv2.imread(cfg.output)
    assert pano.shape == (512, 1024, 3)
    # Le panorama doit redonner la scène d'origine, à une rotation en longitude près
    # (l'image de référence est placée au centre). On cherche ce décalage puis on compare.
    ref = cv2.resize(texture, (1024, 512), interpolation=cv2.INTER_AREA)
    band = slice(200, 312)
    a = cv2.cvtColor(pano[band], cv2.COLOR_BGR2GRAY).astype(np.float32).mean(axis=0)
    b = cv2.cvtColor(ref[band], cv2.COLOR_BGR2GRAY).astype(np.float32).mean(axis=0)
    corr = np.fft.irfft(np.fft.rfft(a - a.mean()) * np.conj(np.fft.rfft(b - b.mean())), 1024)
    shift = int(np.argmax(corr))
    diff = np.abs(np.roll(ref, shift, axis=1)[band].astype(float) - pano[band].astype(float))
    assert diff.mean() < 15.0


def test_fichiers_de_debug(result):
    _, cfg, _, _ = result
    d = cfg.debug_dir
    for rel in ("graph.png", "log.txt", "metrics.json", "masks/coverage.png", "masks/seams.png",
                "keypoints/img_000.jpg", "matches/000_001_avant.jpg", "matches/000_001_apres.jpg"):
        assert os.path.isfile(os.path.join(d, rel)), rel
    rep = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))
    assert {"paires", "focale", "ajustement_de_faisceaux", "couverture"} <= set(rep["metrics"])
    assert "5 ajustement de faisceaux" in rep["timings_s"]


def test_tour_incomplet_est_signale(synthetic_dataset, tmp_path):
    """Avec 5 images sur 12 : sortie 2:1 valide, mais boucle non fermée et tour incomplet signalés."""
    folder, _, _ = synthetic_dataset
    part = tmp_path / "part"
    part.mkdir()
    for k in range(5):
        os.symlink(os.path.join(folder, f"img_{k:03d}.png"), part / f"img_{k:03d}.png")
    out = str(tmp_path / "p.jpg")
    res = run_pipeline(PivotConfig(input=str(part), output=out, width=1024, fill="gradient"))
    assert cv2.imread(out).shape == (512, 1024, 3)
    assert not res.coverage.all()
    assert any("Boucle non fermée" in w for w in res.report.warnings)
    assert any("Tour incomplet" in w for w in res.report.warnings)


def test_erreurs_bloquantes(tmp_path):
    with pytest.raises(PivotError):
        run_pipeline(PivotConfig(input=str(tmp_path / "absent"), output=str(tmp_path / "o.jpg")))
    cv2.imwrite(str(tmp_path / "seule.png"), np.zeros((32, 32, 3), np.uint8))
    with pytest.raises(NotEnoughImagesError):
        run_pipeline(PivotConfig(input=str(tmp_path), output=str(tmp_path / "o.jpg")))
    # En ligne de commande, une erreur bloquante donne le code de sortie 2.
    assert cli.main(["--input", str(tmp_path), "--output", str(tmp_path / "o.jpg")]) == 2


def test_images_sans_recouvrement(tmp_path):
    """Deux images sans rien de commun : erreur explicite plutôt qu'un panorama faux."""
    rng = np.random.default_rng(0)
    for k in range(2):
        cv2.imwrite(str(tmp_path / f"{k}.png"), rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    with pytest.raises(PivotError):
        run_pipeline(PivotConfig(input=str(tmp_path), output=str(tmp_path / "o.jpg"), width=512))
