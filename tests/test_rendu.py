"""Fusion, remplissage des zones vides, métadonnées GPano."""
import cv2
import numpy as np
import pytest

from pivot import blending
from pivot import fill
from pivot import io_utils
from pivot import projection
from pivot.config import PivotConfig
from conftest import render_view, yaw_camera


def _two_views(texture, W=1024, H=512):
    w, h, f = 480, 360, 400.0
    out = []
    for k, yaw in enumerate((-15.0, 15.0)):
        cam = yaw_camera(yaw, f, w, h)
        out.append(projection.warp_image(render_view(texture, cam, w, h), cam, W, H, k))
    return out, W, H


def test_gains_compensent_une_difference_d_exposition(texture):
    warped, W, H = _two_views(texture)
    warped[1].image = np.clip(warped[1].image.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    gains = blending.compute_gains(warped, W, H)
    assert gains[1] / gains[0] == pytest.approx(1 / 0.7, rel=0.12)


@pytest.mark.parametrize("seam", ["voronoi", "graphcut"])
def test_coutures_forment_une_partition(texture, seam):
    """Chaque pixel couvert appartient à exactement une image."""
    warped, W, H = _two_views(texture)
    seams = blending.find_seams(warped, W, H, seam)
    count = np.zeros((H, W), np.int32)
    for w, s in zip(warped, seams):
        count[w.y0:w.y0 + w.h, w.cols(W)] += s
    assert (count[blending.coverage_mask(warped, W, H)] == 1).all()
    assert (count[~blending.coverage_mask(warped, W, H)] == 0).all()


@pytest.mark.parametrize("mode", ["multiband", "linear"])
def test_fusion_d_images_coherentes_redonne_la_scene(texture, mode):
    warped, W, H = _two_views(texture)
    seams = blending.find_seams(warped, W, H)
    pano, cov = blending.blend(warped, seams, PivotConfig(width=W, blend=mode, workers=2))
    assert pano.shape == (H, W, 3) and pano.dtype == np.uint8
    assert cov.any() and not cov.all()
    ref = cv2.resize(texture, (W, H), interpolation=cv2.INTER_AREA)
    inner = cv2.erode(cov.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert np.abs(pano.astype(float) - ref.astype(float))[inner].mean() < 12.0
    assert (pano[~cov] == 0).all()


@pytest.mark.parametrize("method", ["none", "color", "gradient", "inpaint"])
def test_remplissage_des_zones_vides(method):
    H, W = 256, 512
    rng = np.random.default_rng(0)
    pano = rng.integers(50, 200, (H, W, 3), dtype=np.uint8)
    cov = np.zeros((H, W), bool)
    cov[90:170] = True                                                 # bande équatoriale seulement
    pano[~cov] = 0
    out = fill.fill_uncovered(pano, cov, method, (10, 20, 30))
    assert out.shape == pano.shape and out.dtype == np.uint8
    assert (out[cov] == pano[cov]).all()                               # les pixels couverts ne changent pas
    if method == "none":
        assert (out[~cov] == 0).all()
    elif method == "color":
        assert (out[~cov] == (10, 20, 30)).all()
    else:
        assert out[~cov].mean() > 60                                   # les trous ont pris les couleurs voisines


def test_metadonnees_gpano(tmp_path):
    path = str(tmp_path / "p.jpg")
    io_utils.save_panorama(path, np.full((64, 128, 3), 127, np.uint8), 90)
    data = open(path, "rb").read()
    assert data[:2] == b"\xff\xd8"
    assert b"http://ns.adobe.com/xap/1.0/\x00" in data[:200]
    for s in (b'GPano:ProjectionType="equirectangular"', b'GPano:FullPanoWidthPixels="128"',
              b'GPano:FullPanoHeightPixels="64"', b'GPano:CroppedAreaImageWidthPixels="128"'):
        assert s in data
    img = cv2.imread(path)                                             # le fichier reste un JPEG valide
    assert img is not None and img.shape == (64, 128, 3)
