"""
Minimal from-scratch Gaussian Process regressor, used as the cheap surrogate
model for Bayesian calibration of the viscoelastic bar model (see
core/bayes_visco_calibration.py).

Deliberately dependency-light: numpy + scipy only (both already required by
the app), no scikit-learn/GPy/BoTorch. The problem it needs to solve is
small (a few dozen points in up to 4 dimensions), so a ~150-line
implementation is both sufficient and easy to audit/traceable, in the same
spirit as the rest of core/ preferring literal, checkable code over an
opaque dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

_JITTER = 1e-10


class GaussianProcess:
    """A zero-mean-after-standardization GP with an anisotropic (ARD)
    squared-exponential kernel:

        k(x, x') = sigma_f^2 * exp(-0.5 * sum_d ((x_d - x'_d) / l_d)^2)

    plus an i.i.d. noise term sigma_n^2 on the diagonal. Hyperparameters
    (l_1..l_d, sigma_f, sigma_n) are fit by maximizing the log marginal
    likelihood ("evidence"), with a handful of random restarts to dodge bad
    local optima. This is what makes the surrogate Bayesian rather than a
    plain interpolator: the kernel hyperparameters are inferred from the
    data instead of hand-tuned, and every prediction carries a calibrated
    variance (how much the surrogate actually knows at that point), not
    just a point estimate -- exactly what the acquisition function needs to
    trade off exploring unknown regions against exploiting known-good ones.

    Input `X` is expected already normalized to roughly [0, 1] per
    dimension (see `bayes_visco_calibration._sample_free`/`_assemble_
    physical`); this class does not rescale it.
    """

    def __init__(self, n_restarts: int = 6, random_state=None):
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(random_state)
        self.X = None
        self.y = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self._log_l = None
        self._sigma_f2 = None
        self._sigma_n2 = None
        self._L = None
        self._alpha = None

    @staticmethod
    def _kernel(X1, X2, log_l, sigma_f2):
        length = np.exp(log_l)
        diff = (X1[:, None, :] - X2[None, :, :]) / length
        sq_dist = np.sum(diff ** 2, axis=-1)
        return sigma_f2 * np.exp(-0.5 * sq_dist)

    def _neg_log_marginal_likelihood(self, theta, X, y):
        n, dim = X.shape
        log_l = theta[:dim]
        sigma_f2 = np.exp(2 * theta[dim])
        sigma_n2 = np.exp(2 * theta[dim + 1])
        K = self._kernel(X, X, log_l, sigma_f2) + (sigma_n2 + _JITTER) * np.eye(n)
        try:
            c, low = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e25
        alpha = cho_solve((c, low), y)
        log_det = 2 * np.sum(np.log(np.diag(c)))
        nll = 0.5 * y @ alpha + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
        return float(nll)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, dim = X.shape
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) or 1.0
        yc = (y - self.y_mean) / self.y_std

        # log_l in [-3, 2] -> length scale roughly [0.05, 7.4] on a unit
        # cube; log_sigma_f in [-2, 2] -> signal std roughly [0.14, 7.4] of
        # the (standardized) target; log_sigma_n in [-4, 0] -> noise std up
        # to the full signal std (the forward model is near-deterministic,
        # but sympy's per-frequency nsolve can wobble slightly between
        # nearby parameter sets, so a little noise headroom keeps the GP
        # from over-fitting that wobble).
        bounds = [(-3.0, 2.0)] * dim + [(-2.0, 2.0), (-4.0, 0.0)]
        best_theta, best_nll = None, np.inf
        for _ in range(self.n_restarts):
            x0 = np.array([self.rng.uniform(lo, hi) for lo, hi in bounds])
            res = minimize(
                self._neg_log_marginal_likelihood, x0, args=(X, yc),
                method="L-BFGS-B", bounds=bounds,
            )
            if res.fun < best_nll:
                best_nll, best_theta = float(res.fun), res.x

        dim = X.shape[1]
        log_l = best_theta[:dim]
        sigma_f2 = np.exp(2 * best_theta[dim])
        sigma_n2 = np.exp(2 * best_theta[dim + 1])
        K = self._kernel(X, X, log_l, sigma_f2) + (sigma_n2 + _JITTER) * np.eye(n)
        c, low = cho_factor(K, lower=True)

        self.X, self.y = X, yc
        self._log_l, self._sigma_f2, self._sigma_n2 = log_l, sigma_f2, sigma_n2
        self._L = (c, low)
        self._alpha = cho_solve((c, low), yc)
        return self

    def predict(self, Xs: np.ndarray):
        """Returns (mean, std) in the original (unstandardized) target
        scale, one pair per row of `Xs`. This is the *marginal* (pointwise)
        predictive std at each row independently -- it says nothing about
        how two rows' predictions co-vary. For that, see `sample_y`."""
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Ks = self._kernel(Xs, self.X, self._log_l, self._sigma_f2)
        mean = Ks @ self._alpha
        v = cho_solve(self._L, Ks.T)
        var = self._sigma_f2 - np.sum(Ks * v.T, axis=1)  # k(x*,x*) == sigma_f2 for a stationary kernel
        var = np.maximum(var, 1e-12)
        return mean * self.y_std + self.y_mean, np.sqrt(var) * self.y_std

    def sample_y(self, Xs: np.ndarray, rng: np.random.Generator, n_samples: int = 1) -> np.ndarray:
        """Draws `n_samples` joint posterior sample function(s), evaluated
        at every row of `Xs`, as an `(len(Xs), n_samples)` array in the
        original target scale. Unlike `predict`'s pointwise (marginal)
        mean/std, this accounts for the *correlation* between nearby
        candidate points, which is what makes a Thompson-Sampling
        acquisition (bayes_visco_calibration._propose_next_thompson: "pick
        the argmax of one random posterior draw") behave like one smooth
        random function rather than `len(Xs)` independent per-point coin
        flips.

        Cost is dominated by a Cholesky factorization of the `len(Xs)` x
        `len(Xs)` posterior covariance matrix -- O(len(Xs)^3) -- so this is
        only meant for a moderate number of candidate points (hundreds to
        low thousands), unlike `predict`, which is cheap at any size."""
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        n = Xs.shape[0]
        Ks = self._kernel(Xs, self.X, self._log_l, self._sigma_f2)
        Kss = self._kernel(Xs, Xs, self._log_l, self._sigma_f2)
        mean = Ks @ self._alpha
        v = cho_solve(self._L, Ks.T)
        cov = Kss - Ks @ v
        # Posterior covariance is only PSD up to floating-point error, and
        # unlike predict()'s single marginal variance per point, this is a
        # full n x n matrix that can include *many* close-together
        # candidate rows (Thompson Sampling's candidate pool is drawn
        # densely) -- pairs the GP considers nearly perfectly correlated,
        # making that block nearly singular. `_JITTER` alone (tuned for
        # `fit`'s well-separated training points) is too small to keep
        # such a block's Cholesky numerically stable, so this scales the
        # diagonal jitter with the kernel's own signal variance instead of
        # using a fixed absolute constant.
        cov = cov + max(1e-6 * self._sigma_f2, _JITTER) * np.eye(n)
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal(size=(n, n_samples))
        samples = mean[:, None] + L @ z
        return samples * self.y_std + self.y_mean


def latin_hypercube(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """`n` points in [0, 1]^dim via Latin Hypercube Sampling: each
    dimension's marginal is split into `n` equal bins and each bin is hit
    exactly once (with a random jitter inside it, and the bins independently
    shuffled across dimensions). This is the initial design for the
    Bayesian search below -- it spreads the very first, most valuable
    evaluations evenly across every dimension without needing
    `n_per_dim ** n_dims` points the way a grid does, and without the large
    unsampled gaps plain uniform-random sampling leaves for small `n`.
    """
    samples = np.empty((n, dim))
    for d in range(dim):
        cut = (np.arange(n) + rng.uniform(size=n)) / n
        rng.shuffle(cut)
        samples[:, d] = cut
    return samples
