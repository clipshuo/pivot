"""Géométrie : focale, rotations relatives, ajustement de faisceaux, correction d'horizon."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import features
import geometry
from config import PivotConfig
from models import Camera, InsufficientOverlapError, PairMatch

W, H, F = 1280, 720, 900.0
CENTER = ((W - 1) / 2.0, (H - 1) / 2.0)
RNORM = float(np.hypot(W, H) / 2.0)


def _true_rotations(n=12, step=30.0):
    """Tour complet avec un léger tangage variable. R monde -> caméra."""
    return [Rotation.from_euler("yx", [k * step, 8 * np.sin(k)], degrees=True).as_matrix().T for k in range(n)]


def _project(R, f, k1, X):
    """Points monde -> pixels centrés observés (avec distorsion), et masque de visibilité."""
    q = X @ R.T
    cam = Camera(R, f, 0.0, 0.0, k1, RNORM)
    z = np.where(q[:, 2] > 1e-6, q[:, 2], 1.0)
    x, y = cam.distort(f * q[:, 0] / z, f * q[:, 1] / z)
    ok = (q[:, 2] > 1e-6) & (np.abs(x) < W / 2) & (np.abs(y) < H / 2)
    return np.column_stack([x, y]), ok


def _synthetic_matches(Rs, f=F, k1=0.0, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6000, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    n, c, out = len(Rs), np.float64(CENTER), []
    for i, j in features.candidate_pairs(n, True, 2):
        pi, oi = _project(Rs[i], f, k1, X)
        pj, oj = _project(Rs[j], f, k1, X)
        both = oi & oj
        if both.sum() < 20:
            continue
        a = pi[both] + rng.normal(0, noise, (both.sum(), 2)) + c
        b = pj[both] + rng.normal(0, noise, (both.sum(), 2)) + c
        out.append(PairMatch(i, j, a, b, np.ones(len(a), bool), valid=True))
    return out


def test_focale_depuis_une_homographie():
    K = np.diag([F, F, 1.0])
    R = Rotation.from_euler("yxz", [20, 4, 2], degrees=True).as_matrix()
    assert geometry.focal_from_homography(K @ R @ np.linalg.inv(K)) == pytest.approx(F, rel=1e-6)


def test_rotation_relative_par_kabsch():
    R = Rotation.from_euler("yxz", [25, -6, 3], degrees=True).as_matrix()
    rng = np.random.default_rng(0)
    p = rng.uniform(-300, 300, (100, 2))
    q = np.column_stack([p, np.full(100, F)]) @ R.T
    pj = F * q[:, :2] / q[:, 2:3]
    est = geometry.rotation_from_matches(p, pj, F)
    assert geometry.rotation_angle_deg(est @ R.T) < 1e-4
    assert geometry.transfer_error(est, F, p, pj).max() < 1e-6


def test_jacobien_analytique():
    """Le jacobien analytique de l'ajustement de faisceaux coïncide avec des différences finies."""
    rng = np.random.default_rng(0)
    M, f, k = 6, 850.0, 0.02
    Ri = Rotation.from_rotvec(rng.normal(0, 0.2, (M, 3))).as_matrix()
    Rj = Rotation.from_rotvec(rng.normal(0, 0.2, (M, 3))).as_matrix()
    pi, pj = rng.uniform(-300, 300, (M, 2)), rng.uniform(-300, 300, (M, 2))
    r0, Jf, Jk, Ji, Jj = geometry._residuals_and_jacobian(f, k, RNORM, Ri, Rj, pi, pj)
    res = lambda f_, k_, Ri_, Rj_: geometry._residuals_and_jacobian(f_, k_, RNORM, Ri_, Rj_, pi, pj)[0]
    e = 1e-6
    assert np.allclose((res(f + e, k, Ri, Rj) - r0) / e, Jf, atol=1e-4)
    assert np.allclose((res(f, k + e, Ri, Rj) - r0) / e, Jk, atol=1e-2)
    for a in range(3):
        d = Rotation.from_rotvec(e * np.eye(3)[a]).as_matrix()
        assert np.allclose((res(f, k, d @ Ri, Rj) - r0) / e, Ji[:, :, a], atol=1e-2)
        assert np.allclose((res(f, k, Ri, d @ Rj) - r0) / e, Jj[:, :, a], atol=1e-2)


@pytest.mark.parametrize("k1", [0.0, 0.03])
def test_ajustement_de_faisceaux_ferme_la_boucle(k1):
    """Départ : focale fausse de 10 %, rotations chaînées avec dérive. Arrivée : vérité retrouvée."""
    Rs = _true_rotations()
    matches = _synthetic_matches(Rs, k1=k1)
    cfg = PivotConfig(distortion=True)
    f0 = 1.1 * F
    geometry.refine_pairs(matches, f0, CENTER, cfg)
    graph = geometry.build_graph(matches, len(Rs))
    nodes = geometry.largest_component(graph)
    assert nodes == list(range(len(Rs)))
    R0 = geometry.initial_rotations(graph, nodes, 0)
    cams = {k: Camera(R0[k], f0, *CENTER, 0.0, RNORM) for k in nodes}
    before = geometry.reprojection_errors(cams, matches, CENTER).mean()
    cams, info = geometry.bundle_adjust(cams, matches, 0, CENTER, cfg)
    after = geometry.reprojection_errors(cams, matches, CENTER).mean()
    assert after < before and after < 1.0
    assert cams[0].focal == pytest.approx(F, rel=0.01)
    assert cams[0].k1 == pytest.approx(k1, abs=0.005)
    for k in nodes:                                                    # rotations relatives à la référence
        est, true = cams[k].R @ cams[0].R.T, Rs[k] @ Rs[0].T
        assert geometry.rotation_angle_deg(est @ true.T) < 0.2
    geometry.update_pair_rotations(matches, cams[0], CENTER)           # paires recalculées avec la focale finale
    assert geometry.edge_consistency_deg(cams, matches).max() < 0.3


def test_composante_connexe_et_recouvrement_insuffisant():
    def edge(i, j):
        return PairMatch(i, j, np.zeros((30, 2)), np.zeros((30, 2)), np.ones(30, bool), valid=True)

    graph = geometry.build_graph([edge(0, 1), edge(1, 2), edge(3, 4)], 6)
    assert geometry.largest_component(graph) == [0, 1, 2]              # 3, 4 et 5 sont écartées
    with pytest.raises(InsufficientOverlapError):
        geometry.largest_component(geometry.build_graph([], 4))


def test_correction_d_horizon():
    """Des caméras sans roulis, vues dans un repère monde penché, sont remises à l'horizontale."""
    tilt = Rotation.from_euler("zx", [12, 7], degrees=True).as_matrix()
    cams = {k: Camera(Rotation.from_euler("y", k * 45, degrees=True).as_matrix().T @ tilt, F, *CENTER)
            for k in range(8)}
    out = geometry.wave_correct(cams, 0)
    for c in out.values():
        assert abs(c.R[0, 1]) < 1e-9                                   # axe x de la caméra : horizontal
    assert np.allclose(out[0].R[2], [0, 0, 1], atol=1e-9)              # la référence regarde le centre du panorama


def test_paires_candidates():
    assert features.candidate_pairs(5, True, 1) == [(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)]   # boucle fermée
    assert len(features.candidate_pairs(6, False, 2)) == 15
    assert (0, 8) in features.candidate_pairs(10, True, 2)
