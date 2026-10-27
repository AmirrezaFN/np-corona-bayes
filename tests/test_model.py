import numpy as np

from npcb import Params, rates, solve, solve_rk4

T = np.array([0.25, 1, 4, 8, 24, 48])


def test_mass_conservation():
    y = solve(100, 0.3, Params(), T)
    np.testing.assert_allclose(y[:, 1:].sum(axis=1), 1.0, atol=1e-6)


def test_corona_saturates_and_peg_slows_it():
    bare = solve(100, 0.0, Params(), T)[:, 0]
    pegged = solve(100, 0.8, Params(), T)[:, 0]
    assert bare[-1] > 0.999
    assert np.all(pegged <= bare)


def test_fixed_particle_ignores_peg():
    # with alpha = 0 the corona has no effect, so PEG cannot change biodistribution
    prm = Params(alpha=0.0)
    np.testing.assert_allclose(solve(100, 0.0, prm, T)[:, 1:], solve(100, 0.9, prm, T)[:, 1:], atol=1e-8)


def test_evolving_particle_clears_faster():
    fixed = solve(100, 0.0, Params(alpha=0.0), T)
    evolving = solve(100, 0.0, Params(), T)
    assert evolving[-1, 2] > fixed[-1, 2]  # more in liver
    assert evolving[-1, 3] < fixed[-1, 3]  # less in tumour


def test_size_window_peaks_at_optimum():
    prm = Params()
    k = [rates(d, 0.0, prm)[1] for d in (20, prm.d_opt, 250)]
    assert k[1] > k[0] and k[1] > k[2]


def test_rk4_matches_reference():
    prm = Params()
    np.testing.assert_allclose(solve_rk4(60, 0.5, prm, T), solve(60, 0.5, prm, T), atol=1e-4)
