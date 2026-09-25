"""Plackett-Luce fitting for partial (top-k) rankings.

Public API:
    Observation              -- one top-k ranking with optional participant set
    PLResult                 -- fit result (status, abilities, contributions)
    ConvergenceStatus        -- converged / budget exhausted / numerical failure
    PLInputError             -- locatable validation error (ValueError subclass)
    MAX_OBJECTS              -- hard interface bound (20)
    fit_plackett_luce        -- penalized MLE under sum(theta) = 0
    ranking_log_probability  -- stable log P of one partial ranking
    ranking_probability      -- P of one partial ranking
    penalized_objective           -- f(theta) = -loglik + lambda/2 ||theta||^2
    penalized_gradient_free       -- analytic gradient in n-1 free coordinates
    penalized_hessian_free        -- analytic Hessian in n-1 free coordinates

Model (see README.md for the full derivation):
    For a ranking (r_1, ..., r_k) over participant set A,
        P = prod_{j=1..k} exp(theta_{r_j}) /
                          sum_{i in A \\ {r_1..r_{j-1}}} exp(theta_i)
    Unranked participants stay in every remaining denominator; objects not in A
    never enter any denominator. k = 0 contributes log P = 0.

Anchored fitting (fit_plackett_luce(..., anchors=..., bounds=...)):
    With at least one anchor the sum-to-zero gauge is dropped entirely; the
    anchored abilities are held at their given constants (never re-centered,
    never Newton-updated) and the L2 penalty applies to the free abilities
    only. Free abilities may carry finite box bounds, handled by a projected
    (active-set) Newton step with Armijo backtracking -- never by clipping an
    unconstrained fit. Anchored objects still enter every denominator their
    participant set implies.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Optional

__all__ = [
    "MAX_OBJECTS",
    "ConvergenceStatus",
    "PLInputError",
    "Observation",
    "PLResult",
    "fit_plackett_luce",
    "ranking_log_probability",
    "ranking_probability",
    "penalized_objective",
    "penalized_gradient_free",
    "penalized_hessian_free",
]

MAX_OBJECTS = 20

# Sum-to-zero tolerance for user-supplied initial theta: relative, documented
# in README. Violations are rejected; the constraint is never silently imposed.
_INITIAL_SUM_TOL = 1e-10

_ARMIJO_C1 = 1e-4
_STEP_SHRINK = 0.5
_MAX_LINESEARCH_TRIES = 80

# Rounding-scale slack for the anchored Armijo test: once the achievable
# objective decrease falls below the resolution of f itself (~1e-16 relative
# per evaluation, accumulated over a batch), a strict decrease test can no
# longer be decided. The slack only releases steps whose required decrease is
# already far below machine resolution; genuine steps are unaffected.
_ARMIJO_ROUNDING_SLACK = 1e-13


class ConvergenceStatus(str, Enum):
    """Terminal state of the Newton solver.

    CONVERGED          -- reduced-gradient infinity norm reached gtol.
    MAX_ITERATIONS     -- search budget exhausted; the (unique) minimizer is
                          known to exist, but it was not reached. Distinct from
                          an infeasibility proof: with lambda > 0 the objective
                          is strictly convex and coercive on the sum-zero
                          hyperplane, so a unique minimizer always exists;
                          infeasible *data* is rejected up front with
                          PLInputError instead.
    LINE_SEARCH_FAILED -- backtracking could not find an Armijo-acceptable step.
    NUMERICAL_ERROR    -- non-finite arithmetic or an unusable Newton system.
    """

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    LINE_SEARCH_FAILED = "line_search_failed"
    NUMERICAL_ERROR = "numerical_error"


class PLInputError(ValueError):
    """Invalid input. Messages identify the offending field/index position."""


@dataclass(frozen=True)
class Observation:
    """One top-k, tie-free partial ranking.

    ranking:      distinct object indices, best first; length k (may be 0).
    participants: distinct indices of objects eligible for this observation.
                  None means all n objects. Every ranking index must belong to
                  the participant set. Unranked participants remain in all
                  denominator stages; non-participants never enter them.
    """

    ranking: Sequence[int] = ()
    participants: Optional[Sequence[int]] = None


@dataclass(frozen=True)
class PLResult:
    status: ConvergenceStatus
    converged: bool
    iterations: int
    theta: tuple[float, ...]
    log_likelihood: float
    penalized_objective: float
    log_contributions: tuple[float, ...]
    gradient_norm: float
    message: str
    # Infinity-norm KKT residual of the box constraints; only meaningful when
    # bounds were supplied (anchored fit), otherwise None.
    boundary_kkt_residual: Optional[float] = None


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def _reject_bool(value, where: str):
    if isinstance(value, bool):
        raise PLInputError(f"{where}: bool is not accepted as an integer argument")


def _checked_int(value, where: str, *, minimum: Optional[int] = None,
                 maximum: Optional[int] = None) -> int:
    _reject_bool(value, where)
    if not isinstance(value, int):
        raise PLInputError(f"{where}: expected int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise PLInputError(f"{where}: value {value} is below minimum {minimum}")
    if maximum is not None and value > maximum:
        raise PLInputError(f"{where}: value {value} exceeds maximum {maximum}")
    return value


def _checked_finite_float(value, where: str, *, positive: bool = False) -> float:
    _reject_bool(value, where)
    if not isinstance(value, (int, float)):
        raise PLInputError(f"{where}: expected real number, got {type(value).__name__}")
    fv = float(value)
    if not math.isfinite(fv):
        raise PLInputError(f"{where}: NaN and Infinity are not accepted")
    if positive and fv <= 0.0:
        raise PLInputError(f"{where}: must be > 0, got {fv!r}")
    return fv


def _checked_index_seq(value, where: str, n: int, *, allow_empty: bool = True
                       ) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise PLInputError(f"{where}: expected a sequence of int indices")
    out = []
    for pos, v in enumerate(value):
        _reject_bool(v, f"{where}[{pos}]")
        if not isinstance(v, int):
            raise PLInputError(
                f"{where}[{pos}]: expected int index, got {type(v).__name__}")
        if v < 0 or v >= n:
            raise PLInputError(
                f"{where}[{pos}]: index {v} is outside [0, {n - 1}]")
        out.append(v)
    if not allow_empty and len(out) == 0:
        raise PLInputError(f"{where}: empty sequence is not allowed")
    return tuple(out)


def _normalize(observations, n: int):
    """Validate the whole batch up front and return (ranking, participants) tuples.

    Nothing is computed before the batch is fully validated, so a rejected batch
    leaves no partial state (the routine is also purely functional).
    """
    if isinstance(observations, (str, bytes)) or not hasattr(observations, "__iter__"):
        raise PLInputError("observations: expected an iterable of Observation")

    normalized = []
    count = 0
    for count, ob in enumerate(observations):
        where = f"observations[{count}]"
        if not isinstance(ob, Observation):
            raise PLInputError(
                f"{where}: expected Observation, got {type(ob).__name__}")
        rank = _checked_index_seq(ob.ranking, f"{where}.ranking", n)
        if ob.participants is None:
            part = tuple(range(n))
        else:
            part = _checked_index_seq(
                ob.participants, f"{where}.participants", n, allow_empty=True)

        if len(set(rank)) != len(rank):
            dup = sorted({i for i in rank if rank.count(i) > 1})
            raise PLInputError(
                f"{where}.ranking: duplicated object index/indices {dup}; "
                "tie-free rankings require distinct indices")
        if len(set(part)) != len(part):
            dup = sorted({i for i in part if part.count(i) > 1})
            raise PLInputError(
                f"{where}.participants: duplicated object index/indices {dup}")
        missing = [i for i in rank if i not in set(part)]
        if missing:
            raise PLInputError(
                f"{where}: ranked object {missing[0]} is not in the participant set")
        # len(rank) <= len(part) follows from distinct + subset, kept explicit.
        if len(rank) > len(part):
            raise PLInputError(
                f"{where}: ranking length {len(rank)} exceeds participant set "
                f"size {len(part)}")
        normalized.append((rank, part))
    return normalized


def _validate_theta_arg(theta, where: str = "theta") -> list[float]:
    if isinstance(theta, (str, bytes)) or not hasattr(theta, "__len__") \
            or len(theta) < 1:
        raise PLInputError(
            f"{where}: expected a non-empty finite sequence of abilities")
    return _validate_theta(theta, len(theta), where)


def _validate_theta(theta, n: int, where: str = "theta") -> list[float]:
    if isinstance(theta, (str, bytes)) or not hasattr(theta, "__iter__"):
        raise PLInputError(f"{where}: expected a sequence of {n} real numbers")
    vals = list(theta)
    if len(vals) != n:
        raise PLInputError(
            f"{where}: expected length {n}, got {len(vals)}")
    out = []
    for pos, v in enumerate(vals):
        out.append(_checked_finite_float(v, f"{where}[{pos}]"))
    s = math.fsum(out)
    scale = max([1.0] + [abs(v) for v in out])
    if abs(s) > _INITIAL_SUM_TOL * scale:
        raise PLInputError(
            f"{where}: abilities must satisfy sum(theta) = 0 "
            f"(|sum| = {abs(s):.3g} exceeds {_INITIAL_SUM_TOL:g} * scale)")
    return out


def _validate_theta_finite(theta, n: int, where: str) -> list[float]:
    """Length-n finite abilities without the sum-to-zero requirement."""
    if isinstance(theta, (str, bytes)) or not hasattr(theta, "__iter__"):
        raise PLInputError(f"{where}: expected a sequence of {n} real numbers")
    vals = list(theta)
    if len(vals) != n:
        raise PLInputError(
            f"{where}: expected length {n}, got {len(vals)}")
    return [_checked_finite_float(v, f"{where}[{pos}]")
            for pos, v in enumerate(vals)]


def _validate_anchors(anchors, n: int) -> dict[int, float]:
    """Validate caller-fixed abilities; reject duplicates, unknown objects,
    and non-finite values before anything is computed."""
    if anchors is None:
        return {}
    if isinstance(anchors, Mapping):
        pairs = list(anchors.items())
    elif isinstance(anchors, (str, bytes)) or not hasattr(anchors, "__iter__"):
        raise PLInputError(
            "anchors: expected a mapping {index: value} or an iterable of "
            "(index, value) pairs")
    else:
        pairs = []
        for pos, item in enumerate(anchors):
            where = f"anchors[{pos}]"
            if isinstance(item, (str, bytes)) or not hasattr(item, "__len__") \
                    or len(item) != 2:
                raise PLInputError(f"{where}: expected an (index, value) pair")
            pairs.append((item[0], item[1]))
    out: dict[int, float] = {}
    for pos, (idx, val) in enumerate(pairs):
        where = f"anchors[{pos}]"
        _reject_bool(idx, f"{where}.index")
        if not isinstance(idx, int):
            raise PLInputError(
                f"{where}.index: expected int, got {type(idx).__name__}")
        if idx < 0 or idx >= n:
            raise PLInputError(
                f"{where}.index: object {idx} is outside [0, {n - 1}]")
        if idx in out:
            raise PLInputError(
                f"{where}.index: duplicate anchor for object {idx}")
        out[idx] = _checked_finite_float(val, f"{where}.value")
    return out


def _validate_bounds(bounds, n: int, anchored: set) -> dict[int, tuple[float, float]]:
    """Validate finite box bounds for free (non-anchored) abilities."""
    if bounds is None:
        return {}
    if isinstance(bounds, Mapping):
        pairs = list(bounds.items())
    elif isinstance(bounds, (str, bytes)) or not hasattr(bounds, "__iter__"):
        raise PLInputError(
            "bounds: expected a mapping {index: (lo, hi)} or an iterable of "
            "(index, (lo, hi)) pairs")
    else:
        pairs = []
        for pos, item in enumerate(bounds):
            where = f"bounds[{pos}]"
            if isinstance(item, (str, bytes)) or not hasattr(item, "__len__") \
                    or len(item) != 2:
                raise PLInputError(f"{where}: expected an (index, (lo, hi)) pair")
            pairs.append((item[0], item[1]))
    out: dict[int, tuple[float, float]] = {}
    for pos, (idx, lh) in enumerate(pairs):
        where = f"bounds[{pos}]"
        _reject_bool(idx, f"{where}.index")
        if not isinstance(idx, int):
            raise PLInputError(
                f"{where}.index: expected int, got {type(idx).__name__}")
        if idx < 0 or idx >= n:
            raise PLInputError(
                f"{where}.index: object {idx} is outside [0, {n - 1}]")
        if idx in anchored:
            raise PLInputError(
                f"{where}.index: object {idx} is anchored; anchored abilities "
                "are fixed constants and cannot carry bounds")
        if idx in out:
            raise PLInputError(
                f"{where}.index: duplicate bounds for object {idx}")
        if isinstance(lh, (str, bytes)) or not hasattr(lh, "__len__") \
                or len(lh) != 2:
            raise PLInputError(f"{where}.value: expected a (lo, hi) pair")
        lo = _checked_finite_float(lh[0], f"{where}.value[0]")
        hi = _checked_finite_float(lh[1], f"{where}.value[1]")
        if lo > hi:
            raise PLInputError(
                f"{where}.value: lower bound {lo!r} exceeds upper bound {hi!r}")
        out[idx] = (lo, hi)
    return out


# --------------------------------------------------------------------------- #
# Core likelihood accumulation (stable logsumexp, analytic gradient & Hessian)
# --------------------------------------------------------------------------- #
def _fill(normalized, theta, lam, want_derivs, penalty_indices=None):
    """Return (objective, log_likelihood, per-obs log contributions, g, H).

    Ambient coordinates (length n). g/H include the lambda penalty. Stages use
    logsumexp with per-stage max subtraction for numerical stability.
    penalty_indices=None penalizes every coordinate (sum-to-zero gauge);
    otherwise only the listed (free) coordinates are penalized.
    """
    n = len(theta)
    g = [0.0] * n
    H = [[0.0] * n for _ in range(n)] if want_derivs else None
    total_ll = 0.0
    contributions = []

    for rank, part in normalized:
        remaining = list(part)
        ll = 0.0
        for winner in rank:
            m = max(theta[i] for i in remaining)
            exps = [math.exp(theta[i] - m) for i in remaining]
            s = math.fsum(exps)
            probs = [e / s for e in exps]  # softmax over the remaining set
            # log P_stage = theta_w - (m + log sum exp(theta_i - m))
            ll += theta[winner] - (m + math.log(s))

            if want_derivs:
                # d(-log P_stage)/d theta = softmax_R - 1_{winner}
                for pos, a in enumerate(remaining):
                    pa = probs[pos]
                    g[a] += pa
                    H[a][a] += pa * (1.0 - pa)
                    for pos2 in range(pos + 1, len(remaining)):
                        b = remaining[pos2]
                        cross = pa * probs[pos2]
                        H[a][b] -= cross
                        H[b][a] -= cross
                g[winner] -= 1.0
            remaining.remove(winner)

        total_ll += ll
        contributions.append(ll)

    pen_idx = range(n) if penalty_indices is None else penalty_indices
    penalty_half = 0.5 * lam * math.fsum(theta[i] * theta[i] for i in pen_idx)
    objective = -total_ll + penalty_half
    if want_derivs:
        for i in pen_idx:
            g[i] += lam * theta[i]
            H[i][i] += lam
    return objective, total_ll, contributions, g, H


def _reduce_vector(g):
    last = g[-1]
    return [g[i] - last for i in range(len(g) - 1)]


def _reduce_matrix(H):
    """Z^T H Z for Z_{i,l} = delta_{i,l} (i<n-1), Z_{n-1,l} = -1."""
    m = len(H) - 1
    hnn = H[m][m]
    return [
        [H[i][j] - H[i][m] - H[m][j] + hnn for j in range(m)]
        for i in range(m)
    ]


def _cholesky(A):
    m = len(A)
    L = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1):
            acc = A[i][j] - math.fsum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if not (acc > 0.0) or not math.isfinite(acc):
                    raise PLInputError("non-positive-definite Newton matrix")
                L[i][j] = math.sqrt(acc)
            else:
                L[i][j] = acc / L[j][j]
    return L


def _solve_cholesky(L, b):
    m = len(b)
    y = [0.0] * m
    for i in range(m):
        y[i] = (b[i] - math.fsum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    x = [0.0] * m
    for i in range(m - 1, -1, -1):
        x[i] = (y[i] - math.fsum(L[j][i] * x[j] for j in range(i + 1, m))) / L[i][i]
    return x


def _theta_from_free(x):
    return [*x, -math.fsum(x)]


def _objective_only(normalized, theta, lam):
    obj, _, _, _, _ = _fill(normalized, theta, lam, False)
    return obj


def _objective_only_anchored(normalized, theta, lam, free_idx):
    obj, _, _, _, _ = _fill(normalized, theta, lam, False,
                            penalty_indices=free_idx)
    return obj


def _kkt_residual_free(x, gf, lo_b, hi_b):
    """Infinity-norm KKT residual of the box constraints at a feasible point.

    Interior coordinate: |g|. At the lower bound: max(0, -g) (a descent
    direction would point below the bound). At the upper bound: max(0, g).
    """
    kkt = 0.0
    for j in range(len(x)):
        gj = gf[j]
        lo, hi = lo_b[j], hi_b[j]
        if lo is not None and x[j] == lo:
            r = max(0.0, -gj)
        elif hi is not None and x[j] == hi:
            r = max(0.0, gj)
        else:
            r = abs(gj)
        if r > kkt:
            kkt = r
    return kkt


def _solve_anchored(norm, theta, free_idx, anchor_map, bound_map, lam,
                    max_iter, gtol):
    """Projected (active-set) Newton over the free coordinates only.

    Anchored abilities stay exactly at their constants inside theta for every
    likelihood/gradient/Hessian evaluation -- they shape the denominators but
    are never part of a Newton update and are never re-centered. Bounds are
    enforced by clamping each backtracked trial step onto the box (a projected
    step), never by fitting unconstrained and clipping afterwards.
    """
    m = len(free_idx)
    lo_b = [bound_map[i][0] if i in bound_map else None for i in free_idx]
    hi_b = [bound_map[i][1] if i in bound_map else None for i in free_idx]
    has_bounds = bool(bound_map)

    x = [theta[i] for i in free_idx]
    for j in range(m):
        lo, hi = lo_b[j], hi_b[j]
        if lo is not None and x[j] < lo:
            raise PLInputError(
                f"initial: free ability for object {free_idx[j]} is "
                f"{x[j]!r}, below its lower bound {lo!r}")
        if hi is not None and x[j] > hi:
            raise PLInputError(
                f"initial: free ability for object {free_idx[j]} is "
                f"{x[j]!r}, above its upper bound {hi!r}")

    def assemble(xv):
        t = list(theta)
        for j, i in enumerate(free_idx):
            t[i] = xv[j]
        return t

    # All abilities fixed: evaluate the likelihood and contributions only.
    if m == 0:
        obj, ll, contribs, _, _ = _fill(norm, theta, lam, False,
                                        penalty_indices=())
        return PLResult(
            status=ConvergenceStatus.CONVERGED, converged=True, iterations=0,
            theta=tuple(theta), log_likelihood=ll, penalized_objective=obj,
            log_contributions=tuple(contribs), gradient_norm=0.0,
            message="all abilities anchored: evaluated likelihood and "
                    "per-observation contributions without optimization",
            boundary_kkt_residual=0.0 if has_bounds else None)

    status = ConvergenceStatus.MAX_ITERATIONS
    message = ""
    steps = 0

    for it in range(max_iter + 1):
        theta = assemble(x)
        obj, ll, contribs, g, H = _fill(norm, theta, lam, True,
                                        penalty_indices=free_idx)
        gf = [g[i] for i in free_idx]
        gnorm = max((abs(v) for v in gf), default=0.0)
        kkt = _kkt_residual_free(x, gf, lo_b, hi_b)
        stationarity = kkt if has_bounds else gnorm

        if not math.isfinite(obj) or not all(map(math.isfinite, gf)):
            status = ConvergenceStatus.NUMERICAL_ERROR
            message = "non-finite objective or gradient encountered"
            steps = it
            break
        if stationarity <= gtol:
            status = ConvergenceStatus.CONVERGED
            kind = "boundary KKT residual" if has_bounds else \
                "free gradient infinity norm"
            message = (f"{kind} {stationarity:.3e} <= gtol {gtol:.3e} "
                       f"after {it} Newton step(s)")
            steps = it
            break
        if it == max_iter:
            status = ConvergenceStatus.MAX_ITERATIONS
            message = (f"budget of {max_iter} Newton steps exhausted with "
                       f"stationarity measure {stationarity:.3e} > gtol "
                       f"{gtol:.3e}")
            steps = max_iter
            break

        # Active set: coordinates pinned at a bound whose gradient does not
        # point back into the feasible box stay out of the Newton system.
        inactive = []
        for j in range(m):
            lo, hi = lo_b[j], hi_b[j]
            if lo is not None and x[j] == lo and gf[j] >= 0.0:
                continue
            if hi is not None and x[j] == hi and gf[j] <= 0.0:
                continue
            inactive.append(j)

        d = [0.0] * m
        if inactive:
            Hsub = [[H[free_idx[a]][free_idx[b]] for b in inactive]
                    for a in inactive]
            gsub = [gf[j] for j in inactive]
            diag_scale = max(Hsub[p][p] for p in range(len(inactive)))
            L = None
            for tau in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
                Hj = [row[:] for row in Hsub]
                if tau:
                    bump = tau * diag_scale
                    for p in range(len(inactive)):
                        Hj[p][p] += bump
                try:
                    L = _cholesky(Hj)
                    break
                except PLInputError:
                    continue
            if L is None:
                status = ConvergenceStatus.NUMERICAL_ERROR
                message = ("Newton Hessian (free block) could not be factored "
                           "as positive definite")
                steps = it
                break
            dsub = _solve_cholesky(L, [-v for v in gsub])
            for p, j in enumerate(inactive):
                d[j] = dsub[p]

        directional = math.fsum(gf[j] * d[j] for j in range(m))
        if directional >= 0.0 or not math.isfinite(directional):
            status = ConvergenceStatus.NUMERICAL_ERROR
            message = "Newton direction is not a descent direction"
            steps = it
            break

        alpha = 1.0
        accepted = False
        for _ in range(_MAX_LINESEARCH_TRIES):
            xn = [x[j] + alpha * d[j] for j in range(m)]
            for j in range(m):  # project the trial point onto the box
                if lo_b[j] is not None and xn[j] < lo_b[j]:
                    xn[j] = lo_b[j]
                if hi_b[j] is not None and xn[j] > hi_b[j]:
                    xn[j] = hi_b[j]
            step = [xn[j] - x[j] for j in range(m)]
            if max((abs(s) for s in step), default=0.0) == 0.0:
                break  # projected step vanished: cannot make progress
            fn = _objective_only_anchored(norm, assemble(xn), lam, free_idx)
            actual_dir = math.fsum(gf[j] * step[j] for j in range(m))
            slack = _ARMIJO_ROUNDING_SLACK * max(1.0, abs(obj))
            if math.isfinite(fn) \
                    and fn <= obj + _ARMIJO_C1 * actual_dir + slack:
                x = xn
                accepted = True
                break
            alpha *= _STEP_SHRINK

        if not accepted:
            status = ConvergenceStatus.LINE_SEARCH_FAILED
            message = (f"backtracking failed after {_MAX_LINESEARCH_TRIES} "
                       f"halvings at iteration {it + 1}")
            steps = it
            break

    theta = assemble(x)
    final_obj, final_ll, final_contribs, fg, _ = _fill(
        norm, theta, lam, True, penalty_indices=free_idx)
    fgf = [fg[i] for i in free_idx]
    final_gnorm = max((abs(v) for v in fgf), default=0.0)
    final_kkt = _kkt_residual_free(x, fgf, lo_b, hi_b)
    return PLResult(
        status=status,
        converged=(status == ConvergenceStatus.CONVERGED),
        iterations=steps,
        theta=tuple(theta),
        log_likelihood=final_ll,
        penalized_objective=final_obj,
        log_contributions=tuple(final_contribs),
        gradient_norm=final_gnorm,
        message=message,
        boundary_kkt_residual=final_kkt if has_bounds else None,
    )


# --------------------------------------------------------------------------- #
# Public modeling functions
# --------------------------------------------------------------------------- #
def penalized_objective(observations, theta, lam):
    """f(theta) = -sum_obs log P(obs | theta) + lambda/2 * ||theta||^2.

    theta must be finite and satisfy sum(theta) = 0.
    """
    if not hasattr(theta, "__len__") or len(theta) < 1:
        raise PLInputError("theta: expected a non-empty finite sequence")
    n = len(theta)
    theta = _validate_theta(theta, n)
    lam = _checked_finite_float(lam, "lam", positive=True)
    norm = _normalize(observations, n)
    return _objective_only(norm, theta, lam)


def penalized_gradient_free(observations, theta, lam):
    """Analytic gradient in free coordinates x: theta_i=x_i, theta_{n-1}=-sum x.

    Length n-1; this is the reduced (constrained) gradient the Newton solver
    drives to zero.
    """
    theta = _validate_theta_arg(theta)
    n = len(theta)
    lam = _checked_finite_float(lam, "lam", positive=True)
    norm = _normalize(observations, n)
    _, _, _, g, _ = _fill(norm, theta, lam, True)
    return _reduce_vector(g)


def penalized_hessian_free(observations, theta, lam):
    """Analytic Hessian in free coordinates (list of n-1 rows of length n-1)."""
    theta = _validate_theta_arg(theta)
    n = len(theta)
    lam = _checked_finite_float(lam, "lam", positive=True)
    norm = _normalize(observations, n)
    _, _, _, _, H = _fill(norm, theta, lam, True)
    return _reduce_matrix(H)


def ranking_log_probability(ranking, theta, participants=None):
    """Stable log P of one tie-free top-k ranking under abilities theta.

    participants=None means all objects. Unranked participants stay in every
    remaining denominator; non-participants are excluded. k=0 returns 0.0.
    """
    theta = _validate_theta_arg(theta)
    n = len(theta)
    norm = _normalize([Observation(ranking, participants)], n)
    rank, part = norm[0]
    remaining = list(part)
    lp = 0.0
    for winner in rank:
        m = max(theta[i] for i in remaining)
        s = math.fsum(math.exp(theta[i] - m) for i in remaining)
        lp += theta[winner] - (m + math.log(s))
        remaining.remove(winner)
    return lp


def ranking_probability(ranking, theta, participants=None):
    """Probability of one partial ranking (may underflow for extremely small P)."""
    return math.exp(ranking_log_probability(ranking, theta, participants))


def fit_plackett_luce(observations, n_objects, *, lam=1.0, initial=None,
                      max_iter=100, gtol=1e-10, anchors=None, bounds=None):
    """Fit penalized Plackett-Luce abilities by backtracking Newton.

    Arguments:
        observations: iterable of Observation (validated as one batch before
            any iteration).
        n_objects:    int in [1, 20], object indices span [0, n_objects).
        lam:          penalty weight lambda > 0 (finite).
        initial:      optional length-n finite abilities with sum ~0; rejected
            otherwise. Defaults to zeros.
        max_iter:     Newton-step budget (int >= 1, bool rejected).
        gtol:         convergence tolerance on the reduced gradient infinity
            norm (finite float > 0).
        anchors:      optional mapping {index: ability} (or iterable of
            (index, ability) pairs) fixing some abilities at caller-supplied
            finite constants. With at least one anchor the sum-to-zero
            constraint is dropped entirely, the L2 penalty applies to the
            free abilities only, and anchored values are never re-centered
            or updated. Duplicate, out-of-range, or non-finite anchors are
            rejected. With no anchors the legacy sum(theta)=0 behavior is
            kept unchanged.
        bounds:       optional mapping {index: (lo, hi)} of finite box bounds
            for free abilities (requires at least one anchor). Enforced by a
            projected active-set Newton step with Armijo backtracking; the
            boundary KKT residual is reported on the result. Bounds on
            anchored objects, lo > hi, or non-finite endpoints are rejected.

    Returns PLResult. Budget exhaustion is reported as
    ConvergenceStatus.MAX_ITERATIONS; invalid/infeasible inputs raise
    PLInputError before any optimization occurs.
    """
    n = _checked_int(n_objects, "n_objects", minimum=1, maximum=MAX_OBJECTS)
    lam = _checked_finite_float(lam, "lam", positive=True)
    max_iter = _checked_int(max_iter, "max_iter", minimum=1)
    gtol = _checked_finite_float(gtol, "gtol", positive=True)

    anchor_map = _validate_anchors(anchors, n)
    if anchor_map:
        bound_map = _validate_bounds(bounds, n, set(anchor_map))
        norm = _normalize(observations, n)
        if initial is None:
            theta = [0.0] * n
        else:
            # Anchored mode drops sum(theta)=0, so the initial point only
            # needs to be finite; anchored entries are overridden below.
            theta = _validate_theta_finite(initial, n, "initial")
        for i, v in anchor_map.items():
            theta[i] = v
        free_idx = [i for i in range(n) if i not in anchor_map]
        return _solve_anchored(norm, theta, free_idx, anchor_map, bound_map,
                               lam, max_iter, gtol)
    if bounds is not None:
        raise PLInputError(
            "bounds: box bounds require anchored fitting; pass at least one "
            "anchor so the sum-to-zero gauge is dropped")

    if initial is None:
        theta = [0.0] * n
    else:
        theta = _validate_theta(initial, n, "initial")
        if n == 1:
            theta = [0.0]

    norm = _normalize(observations, n)

    # n = 1: sum-to-zero pins theta_0 = 0; no free coordinates.
    if n == 1:
        obj, ll, contribs, _, _ = _fill(norm, [0.0], lam, False)
        return PLResult(
            status=ConvergenceStatus.CONVERGED, converged=True, iterations=0,
            theta=(0.0,), log_likelihood=ll, penalized_objective=obj,
            log_contributions=tuple(contribs), gradient_norm=0.0,
            message="trivial optimum: with one object sum(theta)=0 pins theta=0")

    x = theta[:n - 1]
    theta = _theta_from_free(x)
    status = ConvergenceStatus.MAX_ITERATIONS
    message = ""
    steps = 0

    for it in range(max_iter + 1):
        obj, ll, contribs, g, H = _fill(norm, theta, lam, True)
        gr = _reduce_vector(g)
        gnorm = max((abs(v) for v in gr), default=0.0)

        if not math.isfinite(obj) or not all(map(math.isfinite, gr)):
            status = ConvergenceStatus.NUMERICAL_ERROR
            message = "non-finite objective or gradient encountered"
            steps = it
            break
        if gnorm <= gtol:
            status = ConvergenceStatus.CONVERGED
            message = (f"reduced gradient infinity norm {gnorm:.3e} <= "
                       f"gtol {gtol:.3e} after {it} Newton step(s)")
            steps = it
            break
        if it == max_iter:
            status = ConvergenceStatus.MAX_ITERATIONS
            message = (f"budget of {max_iter} Newton steps exhausted with "
                       f"gradient norm {gnorm:.3e} > gtol {gtol:.3e}")
            steps = max_iter
            break

        Hr = _reduce_matrix(H)
        # lambda>0 makes the reduced Hessian analytically positive definite;
        # a small, documented diagonal jitter is only a rounding fallback.
        diag_scale = max(Hr[i][i] for i in range(n - 1))
        L = None
        for tau in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
            Hrj = [row[:] for row in Hr]
            if tau:
                bump = tau * diag_scale
                for i in range(n - 1):
                    Hrj[i][i] += bump
            try:
                L = _cholesky(Hrj)
                break
            except PLInputError:
                continue
        if L is None:
            status = ConvergenceStatus.NUMERICAL_ERROR
            message = "Newton Hessian could not be factored as positive definite"
            steps = it
            break

        d = _solve_cholesky(L, [-v for v in gr])
        directional = math.fsum(gr[i] * d[i] for i in range(n - 1))
        if directional >= 0.0 or not math.isfinite(directional):
            status = ConvergenceStatus.NUMERICAL_ERROR
            message = "Newton direction is not a descent direction"
            steps = it
            break

        alpha = 1.0
        accepted = False
        for _ in range(_MAX_LINESEARCH_TRIES):
            xn = [x[i] + alpha * d[i] for i in range(n - 1)]
            thetan = _theta_from_free(xn)
            fn = _objective_only(norm, thetan, lam)
            if math.isfinite(fn) and fn <= obj + _ARMIJO_C1 * alpha * directional:
                x, theta = xn, thetan
                accepted = True
                break
            alpha *= _STEP_SHRINK

        if not accepted:
            status = ConvergenceStatus.LINE_SEARCH_FAILED
            message = (f"backtracking failed after {_MAX_LINESEARCH_TRIES} "
                       f"halvings at iteration {it + 1}")
            steps = it
            break

    final_obj, final_ll, final_contribs, fg, _ = _fill(norm, theta, lam, True)
    final_gnorm = max((abs(v) for v in _reduce_vector(fg)), default=0.0)
    return PLResult(
        status=status,
        converged=(status == ConvergenceStatus.CONVERGED),
        iterations=steps,
        theta=tuple(theta),
        log_likelihood=final_ll,
        penalized_objective=final_obj,
        log_contributions=tuple(final_contribs),
        gradient_norm=final_gnorm,
        message=message,
    )
