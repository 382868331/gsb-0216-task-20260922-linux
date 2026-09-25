"""Independent reference used ONLY by the tests.

This module deliberately re-derives every expected quantity from scratch with
its own logsumexp code and does NOT import the package under test. The finite
difference checks compare the core's analytic derivatives against central
differences of this independent objective; round-trip agreement with the core
is never the source of an expected value here.
"""
import math


def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def ref_log_prob(ranking, participants, theta):
    """Staged Plackett-Luce log probability, written independently of the core."""
    remaining = list(participants)
    total = 0.0
    for winner in ranking:
        total += theta[winner] - logsumexp([theta[i] for i in remaining])
        remaining.remove(winner)
    return total


def ref_free_theta(x, n):
    return [x[i] if i < n - 1 else -sum(x) for i in range(n)]


def ref_objective_free(x, records, n, lam):
    """records: iterable of (ranking_tuple, participants_tuple)."""
    theta = ref_free_theta(x, n)
    ll = 0.0
    for ranking, participants in records:
        ll += ref_log_prob(ranking, participants, theta)
    return -ll + 0.5 * lam * sum(t * t for t in theta)


def fd_gradient(x, records, n, lam, h=1e-6):
    g = [0.0] * len(x)
    for i in range(len(x)):
        xp = list(x)
        xm = list(x)
        xp[i] += h
        xm[i] -= h
        g[i] = (ref_objective_free(xp, records, n, lam)
                - ref_objective_free(xm, records, n, lam)) / (2.0 * h)
    return g


def fd_hessian(x, records, n, lam, h=1e-4):
    m = len(x)
    H = [[0.0] * m for _ in range(m)]
    f = lambda y: ref_objective_free(y, records, n, lam)
    for i in range(m):
        for j in range(i, m):
            ypp = list(x)
            ypm = list(x)
            ymp = list(x)
            ymm = list(x)
            ypp[i] += h; ypp[j] += h
            ypm[i] += h; ypm[j] -= h
            ymp[i] -= h; ymp[j] += h
            ymm[i] -= h; ymm[j] -= h
            val = (f(ypp) - f(ypm) - f(ymp) + f(ymm)) / (4.0 * h * h)
            H[i][j] = val
            H[j][i] = val
    return H


def two_object_winner_root(m, lam, lo=0.0, hi=100.0, tol=1e-14):
    """Independent bisection root of lam*a = m/(1+exp(2a)).

    For m observations in which object 0 beats object 1 (top-1 or a full
    ranking of two -- both give the same per-observation logistic term), the
    penalized MLE under sum(theta)=0 is theta = (a, -a) with this root.
    """
    def residual(a):
        return lam * a - m / (1.0 + math.exp(2.0 * a))

    flo = residual(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = residual(mid)
        if abs(fm) < tol or 0.5 * (hi - lo) < 1e-13:
            return mid
        if (flo < 0.0) == (fm < 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Anchored-mode independent references
# --------------------------------------------------------------------------- #
def ref_anchored_theta(free_indices, free_vals, anchors, n):
    """Assemble ambient theta; fixed slots keep the caller constants as given."""
    theta = [0.0] * n
    for idx, val in anchors.items():
        theta[idx] = val
    for idx, v in zip(free_indices, free_vals):
        theta[idx] = v
    return theta


def ref_anchored_objective(x, records, n, lam, anchors):
    """-loglik + lambda/2 sum over FREE coordinates only."""
    free = [i for i in range(n) if i not in anchors]
    theta = ref_anchored_theta(free, x, anchors, n)
    ll = 0.0
    for ranking, participants in records:
        ll += ref_log_prob(ranking, participants, theta)
    pen = 0.5 * lam * sum(theta[i] * theta[i] for i in free)
    return -ll + pen


def fd_anchored_gradient(x, records, n, lam, anchors, h=1e-6):
    g = [0.0] * len(x)
    for i in range(len(x)):
        xp = list(x)
        xm = list(x)
        xp[i] += h
        xm[i] -= h
        g[i] = (ref_anchored_objective(xp, records, n, lam, anchors)
                - ref_anchored_objective(xm, records, n, lam, anchors)) \
               / (2.0 * h)
    return g


def fd_anchored_hessian(x, records, n, lam, anchors, h=1e-4):
    m = len(x)
    H = [[0.0] * m for _ in range(m)]
    f = lambda y: ref_anchored_objective(y, records, n, lam, anchors)
    for i in range(m):
        for j in range(i, m):
            ypp = list(x)
            ypm = list(x)
            ymp = list(x)
            ymm = list(x)
            ypp[i] += h; ypp[j] += h
            ypm[i] += h; ypm[j] -= h
            ymp[i] -= h; ymp[j] += h
            ymm[i] -= h; ymm[j] -= h
            val = (f(ypp) - f(ypm) - f(ymp) + f(ymm)) / (4.0 * h * h)
            H[i][j] = val
            H[j][i] = val
    return H
