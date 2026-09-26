"""Independent reference for the at-most-two-way-tied Plackett-Luce model.

Re-derives everything from scratch and never imports the package under test.
Finite differences of ref_free_objective are the reference for the core's
analytic gradient/Hessian; core-vs-reference agreement is not manufactured.
"""
import math


def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def outcomes(remaining):
    out = [(i,) for i in remaining]
    for a in range(len(remaining)):
        for b in range(a + 1, len(remaining)):
            i, j = remaining[a], remaining[b]
            out.append((i, j) if i < j else (j, i))
    return out


def score(outcome, theta, tau):
    if len(outcome) == 1:
        return theta[outcome[0]]
    return math.log(tau) + 0.5 * (theta[outcome[0]] + theta[outcome[1]])


def ref_log_prob(groups, participants, theta, tau):
    remaining = list(participants)
    total = 0.0
    for group in groups:
        group = tuple(sorted(group))
        cands = outcomes(remaining)
        total += score(group, theta, tau) - logsumexp(
            [score(o, theta, tau) for o in cands])
        for obj in group:
            remaining.remove(obj)
    return total


def ref_free_theta(x, n):
    return [x[i] if i < n - 1 else -sum(x) for i in range(n)]


def ref_free_objective(batch, x, n, tau, lam):
    """batch: list of (groups, participants). Penalized NLL in free coords."""
    theta = ref_free_theta(x, n)
    ll = sum(ref_log_prob(g, a, theta, tau) for g, a in batch)
    return -ll + 0.5 * lam * sum(t * t for t in theta)


def ref_free_gradient(batch, x, n, tau, lam, h=1e-6):
    m = len(x)
    grad = [0.0] * m
    for k in range(m):
        xp = x[:]; xp[k] += h
        xm = x[:]; xm[k] -= h
        grad[k] = (ref_free_objective(batch, xp, n, tau, lam)
                   - ref_free_objective(batch, xm, n, tau, lam)) / (2 * h)
    return grad
