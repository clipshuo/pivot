"""Orientations du téléphone (capture.json) : conversion de repères et pipeline guidé par le gyroscope."""
import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import render_view, yaw_camera
from pivot import sensors
from pivot.config import PivotConfig
from pivot.pipeline import run_pipeline


def test_conversion_de_reperes():
    # Téléphone en portrait, vertical, caméra arrière vers le nord d'Android (axe y) :
    # x appareil = est, y appareil = haut, z appareil = sud. Quaternion appareil -> monde.
    R_dw = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    x, y, z, w = Rotation.from_matrix(R_dw).as_quat()
    R = sensors.rotation_from_quaternion([w, x, y, z])
    assert np.allclose(R[2], [0, 0, 1])           # visée : horizontale (z du monde Pivot)
    assert np.allclose(R[1], [0, 1, 0])           # bas de l'image = bas du monde
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    cam = yaw_camera(40.0, 500.0, 640, 480, pitch_deg=25.0)
    assert np.allclose(sensors.rotation_from_quaternion(sensors.quaternion_from_rotation(cam.R)), cam.R)


def test_paires_qui_se_recouvrent():
    R = [yaw_camera(a, 500.0, 640, 480).R for a in (0.0, 30.0, 180.0)]
    assert sensors.overlapping_pairs(R, 60.0) == [(0, 1)]


@pytest.fixture(scope="module")
def capture(tmp_path_factory, texture):
    """Tour de 12 vues dont 3 consécutives sans texture, avec capture.json bruité de 1°."""
    folder = tmp_path_factory.mktemp("capture")
    w, h, focal = 640, 480, 520.0
    rng = np.random.default_rng(3)
    shots = []
    for k in range(12):
        cam = yaw_camera(k * 30.0, focal, w, h)
        img = render_view(texture, cam, w, h)
        if k in (5, 6, 7):
            img[:] = 180                                  # mur blanc : aucun point d'intérêt
        cv2.imwrite(str(folder / f"{k:03d}.png"), img)
        noisy = Rotation.from_rotvec(np.radians(1.0) * rng.normal(size=3) / np.sqrt(3)).as_matrix() @ cam.R
        shots.append({"file": f"{k:03d}.png", "quaternion_wxyz": sensors.quaternion_from_rotation(noisy)})
    (folder / "capture.json").write_text(json.dumps({"camera": {"focal_px": focal}, "shots": shots}))
    return str(folder), focal


def test_images_sans_texture_placees(capture, tmp_path):
    folder, focal = capture
    res = run_pipeline(PivotConfig(input=folder, output=str(tmp_path / "p.jpg"), width=1024))
    m = res.report.metrics
    assert len(res.cameras) == 12                         # aucune image écartée
    assert set(m["capteurs"]["images_placees_par_le_gyroscope_seul"]) >= {"006.png"}
    assert m["couverture"]["horizontale_deg"] == 360.0
    assert m["focale"]["px_pleine_resolution"] == pytest.approx(focal)     # focale du téléphone gardée
    # Les images texturées sont recalées par la vision : rotation relative proche des 30° réels.
    rel = res.cameras[1].R @ res.cameras[0].R.T
    assert np.degrees(np.arccos((np.trace(rel) - 1) / 2)) == pytest.approx(30.0, abs=0.7)


def test_sans_capteurs_les_images_sont_ecartees(capture, tmp_path):
    folder, _ = capture
    res = run_pipeline(PivotConfig(input=folder, output=str(tmp_path / "p.jpg"), width=1024, sensors=False))
    assert len(res.cameras) < 12 and "capteurs" not in res.report.metrics
