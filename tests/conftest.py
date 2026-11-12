"""Configuration commune des tests : scène synthétique dont la géométrie est connue."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pivot.models import Camera
from pivot import projection


def make_texture(width: int = 2048, seed: int = 0) -> np.ndarray:
    """Panorama équirectangulaire artificiel, riche en détails (formes aléatoires + bruit)."""
    rng = np.random.default_rng(seed)
    h = width // 2
    img = cv2.resize(rng.integers(60, 200, (h // 32, width // 32, 3), dtype=np.uint8), (width, h),
                     interpolation=cv2.INTER_CUBIC)
    for _ in range(1500):
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, width)), int(rng.integers(0, h))
        s = int(rng.integers(4, 28))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + s, y + int(s * rng.uniform(0.5, 1.5))), c, -1)
        else:
            cv2.circle(img, (x, y), s // 2, c, -1)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def render_view(texture: np.ndarray, cam: Camera, w: int, h: int) -> np.ndarray:
    """Image vue par une caméra sténopé placée au centre de la sphère texturée."""
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    rays = projection.pixel_to_ray(u, v, cam)
    mx, my = projection.ray_to_equirect(rays, texture.shape[1], texture.shape[0])
    return cv2.remap(texture, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def yaw_camera(yaw_deg: float, focal: float, w: int, h: int, pitch_deg: float = 0.0) -> Camera:
    """Caméra tournée de yaw (vers la droite) puis pitch (vers le haut), R monde -> caméra."""
    R_cam_to_world = Rotation.from_euler("yx", [yaw_deg, pitch_deg], degrees=True).as_matrix()
    return Camera(R_cam_to_world.T, focal, (w - 1) / 2.0, (h - 1) / 2.0, 0.0, float(np.hypot(w, h) / 2.0))


@pytest.fixture(scope="session")
def texture():
    return make_texture()


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory, texture):
    """Tour complet synthétique : 12 vues espacées de 30°, focale connue. Renvoie (dossier, focale, pas)."""
    folder = tmp_path_factory.mktemp("tour")
    w, h, focal, step = 640, 480, 520.0, 30.0
    for k in range(12):
        img = render_view(texture, yaw_camera(k * step, focal, w, h), w, h)
        cv2.imwrite(str(folder / f"img_{k:03d}.png"), img)
    return str(folder), focal, step
