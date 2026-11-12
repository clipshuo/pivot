"""Projection équirectangulaire : conventions, aller-retour, raccord à ±180°."""
import numpy as np
from scipy.spatial.transform import Rotation

import projection
from models import Camera
from conftest import render_view, yaw_camera


def test_rayons_conventions():
    W, H = 720, 360
    rays = projection.equirect_rays(W, H)
    assert np.allclose(np.linalg.norm(rays, axis=-1), 1.0)
    assert np.allclose(rays[H // 2, W // 2], [0, 0, 1], atol=0.01)     # centre de l'image : devant
    assert rays[0, 0, 1] < -0.99                                       # première ligne : zénith (-y)
    assert rays[-1, 0, 1] > 0.99                                       # dernière ligne : nadir
    assert rays[H // 2, 3 * W // 4, 0] > 0.99                          # 3/4 de la largeur : à droite (+x)


def test_aller_retour_point_3d():
    """Un point 3D projeté dans la caméra puis reprojeté en rayon redonne la même direction."""
    rng = np.random.default_rng(1)
    R = Rotation.from_euler("yxz", [40, -15, 5], degrees=True).as_matrix()
    for k1 in (0.0, 0.03, -0.03):
        cam = Camera(R, 800.0, 639.5, 359.5, k1, float(np.hypot(1280, 720) / 2))
        uv = rng.uniform([0, 0], [1279, 719], (200, 2))
        rays = projection.pixel_to_ray(uv[:, 0], uv[:, 1], cam)
        mx, my, front = projection.rays_to_pixels(rays, cam)
        assert front.all()
        assert np.allclose(mx, uv[:, 0], atol=1e-3) and np.allclose(my, uv[:, 1], atol=1e-3)


def test_aller_retour_equirectangulaire():
    W, H = 1024, 512
    u, v = np.meshgrid(np.arange(0, W, 37.0), np.arange(1, H - 1, 23.0))
    rays = projection.equirect_rays(W, H)[v.astype(int), u.astype(int)]
    u2, v2 = projection.ray_to_equirect(rays, W, H)
    assert np.allclose(u2, u, atol=1e-6) and np.allclose(v2, v, atol=1e-6)


def test_rayon_derriere_la_camera_est_rejete():
    cam = Camera(np.eye(3), 500.0, 319.5, 239.5)
    _, _, front = projection.rays_to_pixels(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]), cam)
    assert front.tolist() == [False, True]


def test_zone_a_cheval_sur_180_degres(texture):
    """Une caméra tournée de 180° produit une zone qui traverse le bord droit du panorama."""
    W, H, w, h = 1024, 512, 320, 240
    cam = yaw_camera(180.0, 300.0, w, h)
    warped = projection.warp_image(render_view(texture, cam, w, h), cam, W, H)
    assert warped.wraps(W)
    cols = warped.cols(W)[(warped.mask > 0).any(axis=0)]
    assert cols.min() < 100 and cols.max() > W - 100


def test_projection_redonne_la_texture(texture):
    """Rendre une vue puis la reprojeter doit retrouver le panorama d'origine dans la zone couverte."""
    import cv2
    W, H, w, h = 1024, 512, 640, 480
    cam = yaw_camera(25.0, 520.0, w, h, pitch_deg=10.0)
    warped = projection.warp_image(render_view(texture, cam, w, h), cam, W, H)
    ref = cv2.resize(texture, (W, H), interpolation=cv2.INTER_AREA)
    ref = ref[warped.y0:warped.y0 + warped.h][:, warped.cols(W)]
    inner = warped.weight > 0.1
    diff = np.abs(ref.astype(float) - warped.image.astype(float))[inner]
    assert diff.mean() < 12.0                                          # écart dû au seul rééchantillonnage


def test_reduction_avant_projection():
    assert projection.render_scale(4000.0, 4096) < 0.3                 # source bien plus fine que la sortie
    assert projection.render_scale(300.0, 4096) == 1.0
