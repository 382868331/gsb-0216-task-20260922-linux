"""Plackett-Luce extension allowing at most two-way ties per rank position.

At every stage with remaining object set R, the joint normalized choice is

    weight of singleton {i}      = exp(theta_i)
    weight of tied pair {i, j}   = tau * exp((theta_i + theta_j) / 2)

over ALL singletons and ALL unordered pairs (a single common denominator).
tau > 0 is fixed and given; a tied pair is ONE joint outcome, never the
average of two ordered finishes. Chosen objects (one for a singleton, two for
a pair) are removed from every later stage.

This module is deliberately scoped: it models only singleton / two-way-tied
choices. It makes no claim to be a unified standard for real tied rankings in
general (three-way ties, partial orders, interval censoring, etc.).

Standard library only; see README.md for the full contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Optional

from . import model as _model
from .model import (
    MAX_OBJECTS,
    PLInputError,
    ConvergenceStatus,
)

__all__ = [
    "TiedObservation",
    "TiedPLResult",
    "fit_tied_plackett_luce",
    "tied_ranking_log_probability",
    "tied_ranking_probability",
    "tied_choice_probabilities",
    "tied_penalized_objective",
    "tied_penalized_gradient_free",
    "tied_penalized_hessian_free",
]


@dataclass(frozen=True)
class TiedObservation:
    """One partial ranking whose rank positions are singletons or tied pairs.

    choices:      ordered groups, best first. Each group is a sequence of
                  either one object index (singleton) or two DISTINCT object
                  indices (a two-way tie, stored canonically in sorted order).
                  The same object may not appear in two groups.
    participants: distinct indices of eligible objects; None means all n.
                  Every chosen object must belong to the participant set.
                  Unchosen participants stay in all later denominators.
    """

    choices: Sequence = ()
    participants: Optional[Sequence[int]] = None


@dataclass(frozen=True)
class TiedPLResult:
    status: ConvergenceStatus
    converged: bool
    iterations: int
    theta: tuple[float, ...]
    tau: float
    log_likelihood: float
    penalized_objective: float
    log_contributions: tuple[float, ...]
    gradient_free: tuple[float, ...]
    gradient_norm: float
    message: str


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _canonical_group(group, where: str, n: int):
    """Validate one rank position: exactly 1 or 2 distinct valid int indices."""
    if isinstance(group, (str, bytes)) or not hasattr(group, "__iter__"):
        raise PLInputError(f"{where}: each choice must be a tuple/list of 1 or 2 indices")
    members = tuple(group)
    if len(members) not in (1, 2):
        raise PLInputError(
            f"{where}: a tied choice must contain exactly 1 or 2 objects, "
            f"got {len(members)} (three-way ties are rejected)")
    for pos, v in enumerate(members):
        _model._reject_bool(v, f"{where}[{pos}]")
        if not isinstance(v, int):
            raise PLInputError(
                f"{where}[{pos}]: expected int index, got {type(v).__name__}")
        if v < 0 or v >= n:
            raise PLInputError(
                f"{where}[{pos}]: index {v} is outside [0, {n - 1}]")
    if len(members) == 2 and members[0] == members[1]:
        raise PLInputError(
            f"{where}: tied pair ({members[0]}, {members[1]}) repeats the same object")
    return tuple(sorted(members))


def _normalize_tied(observations, n: int):
    """Validate the whole batch before any computation.

    Returns a list of (tuple_of_canonical_groups, participants_tuple).
    Rejects: non-TiedObservation elements, illegal group sizes, bad indices,
    repeated objects across groups, duplicated participants, ranked objects
    outside the participant set.
    """
    if isinstance(observations, (str, bytes)) or not hasattr(observations, "__iter__"):
        raise PLInputError("observations: expected an iterable of TiedObservation")

    normalized = []
    for t, ob in enumerate(observations):
        where = f"observations[{t}]"
        if not isinstance(ob, TiedObservation):
            raise PLInputError(
                f"{where}: expected TiedObservation, got {type(ob).__name__}")
        if isinstance(ob.choices, (str, bytes)) or not hasattr(ob.choices, "__iter__"):
            raise PLInputError(f"{where}.choices: expected an ordered sequence of groups")

        groups = []
        seen = []
        for g_pos, group in enumerate(ob.choices):
            canon = _canonical_group(group, f"{where}.choices[{g_pos}]", n)
            for obj in canon:
                if obj in seen:
                    raise PLInputError(
                        f"{where}.choices: object {obj} is selected in two "
                        f"rank positions (position {g_pos}); each object is "
                        "removed after its first selection")
                seen.append(obj)
            groups.append(canon)

        if ob.participants is None:
            part = tuple(range(n))
        else:
            part = _model._checked_index_seq(
                ob.participants, f"{where}.participants", n, allow_empty=True)
            if len(set(part)) != len(part):
                dup = sorted({i for i in part if part.count(i) > 1})
                raise PLInputError(
                    f"{where}.participants: duplicated object index/indices {dup}")

        part_set = set(part)
        for obj in seen:
            if obj not in part_set:
                raise PLInputError(
                    f"{where}: chosen object {obj} is not in the participant set")
        normalized.append((tuple(groups), part))
    return normalized


def _check_tau(tau):
    tau = _model._checked_finite_float(tau, "tau", positive=False)
    if tau <= 0.0:
        raise PLInputError(f"tau: must be > 0, got {tau!r}")
    return tau


# --------------------------------------------------------------------------- #
# One-stage choice probabilities
# --------------------------------------------------------------------------- #
def _stage_outcomes(remaining):
    """All candidate outcomes: singletons (i,) and unordered pairs (i, j)."""
    outcomes = [(i,) for i in remaining]
    for a in range(len(remaining)):
        for b in range(a + 1, len(remaining)):
            i, j = remaining[a], remaining[b]
            outcomes.append((i, j) if i < j else (j, i))
    return outcomes


def _stage_scores(remaining, theta, tau):
    """Return (outcomes, log_weights, max_log_weight) for one stage."""
    log_tau = math.log(tau)
    outcomes = _stage_outcomes(remaining)
    values = []
    for out in outcomes:
        if len(out) == 1:
            values.append(theta[out[0]])
        else:
            values.append(log_tau + 0.5 * (theta[out[0]] + theta[out[1]]))
    return outcomes, values, max(values)


def _outcome_members(outcome):
    if len(outcome) == 1:
        return ((outcome[0], 1.0),)
    return ((outcome[0], 0.5), (outcome[1], 0.5))


def _fill_tied(normalized, theta, tau, lam, want_derivs):
    """NLL/likelihood with analytic ambient gradient and Hessian.

    At each stage every singleton and every unordered pair shares one common
    log-denominator (joint logsumexp). Returns
    (objective, log_likelihood, contributions, ambient_g, ambient_H).
    """
    n = len(theta)
    g = [0.0] * n
    H = [[0.0] * n for _ in range(n)] if want_derivs else None
    total_ll = 0.0
    contributions = []

    for groups, part in normalized:
        remaining = list(part)
        ll = 0.0
        for group in groups:
            outcomes, values, m = _stage_scores(remaining, theta, tau)
            exps = [math.exp(v - m) for v in values]
            s = math.fsum(exps)
            probs = [e / s for e in exps]
            log_z = m + math.log(s)
            chosen_index = outcomes.index(group)
            ll += values[chosen_index] - log_z

            if want_derivs:
                # NLL = log Z - v_chosen; g = E_p[u] - u_chosen,
                # H = Var_p(u), where u is the score gradient of an outcome
                # (1 for a singleton member, 1/2 for each pair member).
                expected = [0.0] * n
                for k, outcome in enumerate(outcomes):
                    p = probs[k]
                    members = _outcome_members(outcome)
                    for a, ca in members:
                        expected[a] += p * ca
                        H[a][a] += p * ca * ca
                    if len(members) == 2:
                        (a, ca), (b, cb) = members
                        cross = p * ca * cb
                        H[a][b] += cross
                        H[b][a] += cross
                for a in range(n):
                    ea = expected[a]
                    if ea:
                        for b in range(n):
                            H[a][b] -= ea * expected[b]
                    g[a] += ea
                for a, ca in _outcome_members(group):
                    g[a] -= ca

            for obj in group:
                remaining.remove(obj)

        total_ll += ll
        contributions.append(ll)

    penalty_half = 0.5 * lam * math.fsum(t * t for t in theta)
    objective = -total_ll + penalty_half
    if want_derivs:
        for i in range(n):
            g[i] += lam * theta[i]
            H[i][i] += lam
    return objective, total_ll, contributions, g, H


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #
def tied_choice_probabilities(remaining, theta, tau):
    """Enumerate every choice probability on set `remaining` (small sets only).

    Returns ((outcome, probability), ...); an outcome is a one-element tuple
    for a singleton and a sorted two-element tuple for a tied pair. The
    probabilities share one common denominator and are returned normalized
    with math.fsum; callers can check that sum directly.
    """
    theta = _model._validate_theta_arg(theta)
    n = len(theta)
    rem = _model._checked_index_seq(remaining, "remaining", n)
    if len(set(rem)) != len(rem):
        raise PLInputError("remaining: duplicated object indices")
    tau = _check_tau(tau)
    outcomes, values, m = _stage_scores(list(rem), theta, tau)
    exps = [math.exp(v - m) for v in values]
    s = math.fsum(exps)
    return tuple((outcome, e / s) for outcome, e in zip(outcomes, exps))


def tied_ranking_log_probability(choices, theta, tau, participants=None):
    """Stable log P of one singleton/pair partial ranking with fixed tau."""
    theta = _model._validate_theta_arg(theta)
    n = len(theta)
    tau = _check_tau(tau)
    normalized = _normalize_tied(
        [TiedObservation(choices, participants)], n)
    groups, part = normalized[0]
    remaining = list(part)
    lp = 0.0
    for group in groups:
        outcomes, values, m = _stage_scores(remaining, theta, tau)
        s = math.fsum(math.exp(v - m) for v in values)
        lp += values[outcomes.index(group)] - (m + math.log(s))
        for obj in group:
            remaining.remove(obj)
    return lp


def tied_ranking_probability(choices, theta, tau, participants=None):
    """Probability of one singleton/pair partial ranking (may underflow)."""
    return math.exp(
        tied_ranking_log_probability(choices, theta, tau, participants))


def tied_penalized_objective(observations, theta, tau, lam=1.0):
    """f(theta) = -log P(data | theta, tau) + lambda/2 * ||theta||^2."""
    theta = _model._validate_theta_arg(theta)
    n = len(theta)
    tau = _check_tau(tau)
    lam = _model._checked_finite_float(lam, "lam", positive=True)
    normalized = _normalize_tied(observations, n)
    obj, _, _, _, _ = _fill_tied(normalized, theta, tau, lam, False)
    return obj


def tied_penalized_gradient_free(observations, theta, tau, lam=1.0):
    """Analytic gradient in the n-1 free (sum-to-zero) coordinates."""
    theta = _model._validate_theta_arg(theta)
    n = len(theta)
    tau = _check_tau(tau)
    lam = _model._checked_finite_float(lam, "lam", positive=True)
    normalized = _normalize_tied(observations, n)
    _, _, _, g, _ = _fill_tied(normalized, theta, tau, lam, True)
    return _model._reduce_vector(g)


def tied_penalized_hessian_free(observations, theta, tau, lam=1.0):
    """Analytic Hessian in the n-1 free (sum-to-zero) coordinates."""
    theta = _model._validate_theta_arg(theta)
    n = len(theta)
    tau = _check_tau(tau)
    lam = _model._checked_finite_float(lam, "lam", positive=True)
    normalized = _normalize_tied(observations, n)
    _, _, _, _, H = _fill_tied(normalized, theta, tau, lam, True)
    return _model._reduce_matrix(H)


def fit_tied_plackett_luce(observations, n_objects, tau, *, lam=1.0,
                           initial=None, max_iter=100, gtol=1e-10):
    """Fit abilities theta (sum theta = 0) for singleton/pair partial rankings.

    tau > 0 is the FIXED, user-given pair weight multiplier; it is not
    estimated. Returns TiedPLResult with the log likelihood, per-observation
    contributions, the analytic free-coordinate gradient at the optimum and
    the same convergence-status contract as fit_plackett_luce.
    """
    n = _model._checked_int(
        n_objects, "n_objects", minimum=1, maximum=MAX_OBJECTS)
    tau = _check_tau(tau)
    lam = _model._checked_finite_float(lam, "lam", positive=True)
    max_iter = _model._checked_int(max_iter, "max_iter", minimum=1)
    gtol = _model._checked_finite_float(gtol, "gtol", positive=True)

    if initial is None:
        theta0 = [0.0] * n
    else:
        theta0 = _model._validate_theta(initial, n, "initial")
        if n == 1:
            theta0 = [0.0]

    normalized = _normalize_tied(observations, n)

    if n == 1:
        obj, ll, contribs, g, _ = _fill_tied(normalized, [0.0], tau, lam, True)
        return TiedPLResult(
            status=ConvergenceStatus.CONVERGED, converged=True, iterations=0,
            theta=(0.0,), tau=tau, log_likelihood=ll,
            penalized_objective=obj, log_contributions=tuple(contribs),
            gradient_free=(), gradient_norm=0.0,
            message="trivial optimum: with one object sum(theta)=0 pins theta=0")

    def fill_all(th, want_derivs):
        return _fill_tied(normalized, th, tau, lam, want_derivs)

    def objective_fn(th):
        obj, _, _, _, _ = _fill_tied(normalized, th, tau, lam, False)
        return obj

    status, steps, theta, message = _model._newton_solve(
        n, lam, theta0, fill_all, objective_fn, gtol, max_iter)

    obj, ll, contribs, g, _ = _fill_tied(normalized, theta, tau, lam, True)
    gr = _model._reduce_vector(g)
    gnorm = max((abs(v) for v in gr), default=0.0)
    return TiedPLResult(
        status=status,
        converged=(status == ConvergenceStatus.CONVERGED),
        iterations=steps,
        theta=tuple(theta),
        tau=tau,
        log_likelihood=ll,
        penalized_objective=obj,
        log_contributions=tuple(contribs),
        gradient_free=tuple(gr),
        gradient_norm=gnorm,
        message=message,
    )
