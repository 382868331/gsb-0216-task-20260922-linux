"""Independent brute-force reference for tests.

This module deliberately does NOT import the package under test.  Everything
is recomputed from the model definition using direct ``math.exp``/``log``
expressions, kept to moderate theta ranges where no shifting is needed.
"""

from __future__ import annotations

import math


def ref_level_logprob(theta, winner, remaining):
    denom = 0.0
    for i in remaining:
        denom += math.exp(theta[i])
    return theta[winner] - math.log(denom)


def ref_ranking_logprob(theta, ranked, participants):
    """Direct sequential softmax (the model definition, no logsumexp)."""
    remaining = list(participants)
    total = 0.0
    for w in ranked:
        total += ref_level_logprob(theta, w, remaining)
        remaining.remove(w)
    return total


def ref_objective(theta, observations, lam):
    """Penalised negative log-likelihood: F = -sum logp + lam/2 ||theta||^2."""
    f = 0.0
    for ranked, participants in observations:
        f -= ref_ranking_logprob(theta, ranked, participants)
    f += 0.5 * lam * sum(t * t for t in theta)
    return f


def ref_gradient(theta, observations, lam):
    """Gradient of ref_objective (the penalised NEGATIVE log-likelihood)."""
    n = len(theta)
    g = [lam * t for t in theta]
    for ranked, participants in observations:
        remaining = list(participants)
        for w in ranked:
            exps = [math.exp(theta[i]) for i in remaining]
            z = sum(exps)
            for pos, i in enumerate(remaining):
                g[i] += exps[pos] / z
            g[w] -= 1.0
            remaining.remove(w)
    return g


def ref_hessian(theta, observations, lam):
    """Analytic Hessian of ref_objective, written independently."""
    n = len(theta)
    h = [[0.0] * n for _ in range(n)]
    for i in range(n):
        h[i][i] = lam
    for ranked, participants in observations:
        remaining = list(participants)
        for w in ranked:
            exps = [math.exp(theta[i]) for i in remaining]
            z = sum(exps)
            p = [e / z for e in exps]
            for a, ia in enumerate(remaining):
                for b, ib in enumerate(remaining):
                    if a == b:
                        h[ia][ib] += p[a] - p[a] * p[b]
                    else:
                        h[ia][ib] -= p[a] * p[b]
            remaining.remove(w)
    return h


def fd_gradient(theta, observations, lam, h=1e-6):
    g = [0.0] * len(theta)
    for i in range(len(theta)):
        up = theta.copy()
        dn = theta.copy()
        up[i] += h
        dn[i] -= h
        g[i] = (ref_objective(up, observations, lam)
                - ref_objective(dn, observations, lam)) / (2 * h)
    return g


def fd_hessian(theta, observations, lam, h=1e-5):
    n = len(theta)
    hess = [[0.0] * n for _ in range(n)]
    base = ref_gradient(theta, observations, lam)
    for j in range(n):
        bumped = theta.copy()
        bumped[j] += h
        gj = ref_gradient(bumped, observations, lam)
        for i in range(n):
            hess[i][j] = (gj[i] - base[i]) / h
    return hess


def two_object_winner_equation(diff, n0, n1, lam):
    """Stationarity for winner-only data on 2 objects, d = theta0 - theta1.

    F(d) = -n0*log sigmoid(d) - n1*log sigmoid(-d) + lam/4 d^2 (sum-zero),
    F'(d) = -n0*sigmoid(-d) + n1*sigmoid(d) + lam*d/2.
    """
    s_pos = 1.0 / (1.0 + math.exp(-diff))
    s_neg = 1.0 - s_pos
    return -n0 * s_neg + n1 * s_pos + 0.5 * lam * diff


def bisect_root(fn, lo, hi, tol=1e-14):
    flo, fhi = fn(lo), fn(hi)
    assert flo * fhi < 0 or flo == 0.0 or fhi == 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if abs(hi - lo) < tol or fm == 0.0:
            return mid
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)
