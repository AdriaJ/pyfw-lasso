import pyfwl.utils as ut
import numpy as np

N = 10
a = np.arange(N).astype(float)
b = (-1) ** a
tau = .5


def test_apply_pos():
    norm = ut.L1NormPositivityConstraint(shape=(1, N))

    value_a = (N - 1) * N / 2  # n*(n+1)/2 with n = N-1

    assert np.allclose(norm(a), value_a)
    assert np.allclose(norm(np.stack([a, ] * 3)), np.stack([value_a, ] * 3))


def test_apply_neg():
    norm = ut.L1NormPositivityConstraint(shape=(1, N))

    value_b = np.inf

    assert np.allclose(norm(b), value_b)
    assert np.allclose(norm(np.stack([b, ] * 3)), np.stack([value_b, ] * 3))


def test_prox_pos():
    norm = ut.L1NormPositivityConstraint(shape=(1, N))

    prox_a = a - tau
    prox_a[prox_a < 0.] = 0.

    assert np.allclose(norm.prox(a, tau), prox_a)


def test_prox_neg():
    norm = ut.L1NormPositivityConstraint(shape=(1, N))

    prox_b = np.zeros_like(b)
    prox_b[b >= tau] = b[b >= tau] - tau
    assert np.allclose(norm.prox(b, tau), prox_b)
