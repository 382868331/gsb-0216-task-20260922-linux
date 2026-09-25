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

Anchored mode (at least one fixed ability):
    fit_plackett_luce(..., fixed_theta=..., bounds=...) / fit_anchored(...)
    fixes named objects at caller-supplied ability constants and optimizes
    the remaining coordinates directly. With one anchor or more, the gauge is
    fixed by the anchors: no sum(theta)=0 constraint is imposed and fixed
    abilities are never recentered. The lambda penalty is applied to the FREE
    coordinates only. Free coordinates may carry finite box bounds; the solver
    takes an active-set constrained Newton step with backtracking (never an
    unconstrained fit followed by truncation) and reports the boundary KKT
    residual. With every object fixed it only evaluates the likelihood and the
    per-observation contributions.
"""
from __future__ import annotations

import math
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
    "fit_anchored",
    "Bound",
    "anchored_penalized_objective",
    "anchored_gradient_free",
    "anchored_hessian_free",
]

MAX_OBJECTS = 20

# Sum-to-zero tolerance for user-supplied initial theta: relative, documented
# in README. Violations are rejected; the constraint is never silently imposed.
_INITIAL_SUM_TOL = 1e-10

_ARMIJO_C1 = 1e-4
_STEP_SHRINK = 0.5
_MAX_LINESEARCH_TRIES = 80


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
class Bound:
    """Finite box bound [lower, upper] on one free ability coordinate."""

    lower: float
    upper: float


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
    # Infinity norm of the box-constrained KKT stationarity residual on the
    # FREE coordinates: min/max of the free gradient at active lower/upper
    # bounds and the raw gradient at strictly interior coordinates. Exactly
    # 0.0 for the unconstrained sum-zero path and for all-fixed evaluations.
    boundary_kkt_residual: float = 0.0


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


def _validate_anchored_initial(initial, n, anchors, bounds_by_free):
    """Validate the anchored start in ambient coordinates.

    No sum-zero requirement: anchors fix the gauge. Fixed slots must agree
    with the anchor constants exactly; free slots must be finite and lie
    within any supplied box bounds.
    """
    if isinstance(initial, (str, bytes)) or not hasattr(initial, "__iter__"):
        raise PLInputError(
            f"initial: expected a sequence of {n} real numbers")
    vals = list(initial)
    if len(vals) != n:
        raise PLInputError(
            f"initial: expected length {n}, got {len(vals)}")
    out = []
    for i, v in enumerate(vals):
        fv = _checked_finite_float(v, f"initial[{i}]")
        if i in anchors:
            if fv != anchors[i]:
                raise PLInputError(
                    f"initial[{i}]: object {i} is anchored at "
                    f"{anchors[i]!r}; fixed abilities are not recentered or "
                    f"overwritten (got {fv!r})")
        else:
            b = bounds_by_free.get(i)
            if b is not None and not (b.lower - 1e-12 * max(
                    1.0, abs(b.lower), abs(b.upper)) <= fv <=
                    b.upper + 1e-12 * max(1.0, abs(b.lower), abs(b.upper))):
                raise PLInputError(
                    f"initial[{i}]: value {fv!r} is outside the finite bound "
                    f"[{b.lower!r}, {b.upper!r}]")
        out.append(fv)
    return out


# --------------------------------------------------------------------------- #
# Core likelihood accumulation (stable logsumexp, analytic gradient & Hessian)
# --------------------------------------------------------------------------- #
def _fill(normalized, theta, lam, want_derivs):
    """Return (objective, log_likelihood, per-obs log contributions, g, H).

    Ambient coordinates (length n). g/H include the lambda penalty. Stages use
    logsumexp with per-stage max subtraction for numerical stability.
    """
    return _fill_masked(normalized, theta, lam, want_derivs, None)


def _fill_masked(normalized, theta, lam, want_derivs, penalty_mask):
    """Like _fill, but the lambda penalty only hits indices with mask True.

    penalty_mask=None penalizes every coordinate (the sum-zero path). In
    anchored mode only the free objects are penalized; fixed abilities enter
    every probability but are never regularized.
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

    if penalty_mask is None:
        penalty_half = 0.5 * lam * math.fsum(t * t for t in theta)
    else:
        penalty_half = 0.5 * lam * math.fsum(
            theta[i] * theta[i] for i in range(n) if penalty_mask[i])
    objective = -total_ll + penalty_half
    if want_derivs:
        for i in range(n):
            if penalty_mask is None or penalty_mask[i]:
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


# --------------------------------------------------------------------------- #
# Anchored mode: fixed ability constants, free box bounds, active-set Newton
# --------------------------------------------------------------------------- #
def _validate_fixed(fixed_theta, n):
    """Validate caller anchors; return {object_index: finite ability}.

    A mapping or an iterable of (index, value) pairs is accepted. Repeated
    anchors, unknown/non-int/bool indices and non-finite values are rejected
    up front.
    """
    if isinstance(fixed_theta, dict) or hasattr(fixed_theta, "items"):
        pairs = list(fixed_theta.items())
        as_pairs = False
    elif not isinstance(fixed_theta, (str, bytes)) \
            and hasattr(fixed_theta, "__iter__"):
        pairs = list(fixed_theta)
        as_pairs = True
    else:
        raise PLInputError(
            "fixed_theta: expected a mapping {object_index: ability} or an "
            "iterable of (object_index, ability) pairs")

    anchors = {}
    for pos, item in enumerate(pairs):
        if as_pairs:
            where = f"fixed_theta[{pos}]"
            try:
                idx, val = item
            except (TypeError, ValueError):
                raise PLInputError(
                    f"{where}: expected a (object_index, ability) pair")
        else:
            idx, val = item
            where = f"fixed_theta[{idx!r}]"
        _reject_bool(idx, f"{where} index")
        if not isinstance(idx, int):
            raise PLInputError(
                f"{where}: object index must be int, got {type(idx).__name__}")
        if idx < 0 or idx >= n:
            raise PLInputError(
                f"{where}: anchored object {idx} is outside [0, {n - 1}]")
        if idx in anchors:
            raise PLInputError(
                f"fixed_theta: anchored object {idx} is given more than once")
        anchors[idx] = _checked_finite_float(val, f"fixed_theta[{idx}]")
    return anchors


def _validate_bounds(bounds, n, anchors):
    """Validate finite box bounds on free coordinates; return {index: Bound}.

    Bounds on fixed/unknown indices, non-finite endpoints and empty or
    reversed intervals are rejected.
    """
    if bounds is None:
        return {}
    if isinstance(bounds, dict) or hasattr(bounds, "items"):
        raw = list(bounds.items())
    elif hasattr(bounds, "__len__") and len(bounds) == n:
        raw = [(i, v) for i, v in enumerate(bounds) if v is not None]
    else:
        raise PLInputError(
            f"bounds: expected a mapping or a length-{n} sequence of "
            "(lower, upper) pairs / None")

    out = {}
    for idx, value in raw:
        _reject_bool(idx, f"bounds key {idx!r}")
        if not isinstance(idx, int) or idx < 0 or idx >= n:
            raise PLInputError(
                f"bounds: object index {idx!r} must be an int in [0, {n - 1}]")
        if idx in anchors:
            raise PLInputError(
                f"bounds[{idx}]: object {idx} is fixed by an anchor; bounds "
                "apply to free coordinates only")
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise PLInputError(f"bounds[{idx}]: expected a (lower, upper) pair")
        lo = _checked_finite_float(value[0], f"bounds[{idx}].lower")
        hi = _checked_finite_float(value[1], f"bounds[{idx}].upper")
        if lo > hi:
            raise PLInputError(
                f"bounds[{idx}]: lower {lo!r} exceeds upper {hi!r}")
        if idx in out:
            raise PLInputError(f"bounds: object {idx} is bounded more than once")
        out[idx] = Bound(lo, hi)
    return out


_BOUND_EPS = 1e-12


def _at_lower(x, b):
    scale = max(1.0, abs(b.lower), abs(b.upper))
    return x <= b.lower + _BOUND_EPS * scale


def _at_upper(x, b):
    scale = max(1.0, abs(b.lower), abs(b.upper))
    return x >= b.upper - _BOUND_EPS * scale


def _project_bounds(x, bounds_by_free, free):
    y = list(x)
    for k, idx in enumerate(free):
        b = bounds_by_free.get(idx)
        if b is not None:
            y[k] = min(b.upper, max(b.lower, y[k]))
    return y


def _projected_gradient_residual(gf, x, bounds_by_free, free):
    """Inf norm of the box-KKT stationarity residual on free coordinates."""
    worst = 0.0
    for k, idx in enumerate(free):
        b = bounds_by_free.get(idx)
        v = gf[k]
        if b is not None and _at_lower(x[k], b):
            v = max(0.0, -v)  # lower-bound KKT requires g >= 0
        elif b is not None and _at_upper(x[k], b):
            v = max(0.0, v)   # upper-bound KKT requires g <= 0
        worst = max(worst, abs(v))
    return worst


def _anchored_theta(free, x, anchors, n):
    theta = [0.0] * n
    for idx, val in anchors.items():
        theta[idx] = val
    for k, idx in enumerate(free):
        theta[idx] = x[k]
    return theta


def _anchored_objective(normalized, theta, lam, penalty_mask):
    obj, _, _, _, _ = _fill_masked(normalized, theta, lam, False, penalty_mask)
    return obj


def _max_step_to_box(x, d, bounds_by_free, free):
    alpha_max = 1.0
    hit = False
    for k, idx in enumerate(free):
        b = bounds_by_free.get(idx)
        if b is None or d[k] == 0.0:
            continue
        if d[k] < 0.0:
            if x[k] <= b.lower + _BOUND_EPS * max(
                    1.0, abs(b.lower), abs(b.upper)):
                continue  # at lower face, moving up/inside: unrestricted
            room = (x[k] - b.lower) / -d[k]
        else:
            if x[k] >= b.upper - _BOUND_EPS * max(
                    1.0, abs(b.lower), abs(b.upper)):
                continue  # at upper face, moving down/inside: unrestricted
            room = (b.upper - x[k]) / d[k]
        if room < 0.0:
            # Direction points inside from the face: the bound does not
            # restrict any positive steplength along this coordinate.
            continue
        if room < alpha_max:
            alpha_max, hit = room, True
    return (min(1.0, alpha_max) if hit else 1.0)


def _fit_anchored(normalized, n, lam, anchors, bounds_by_free, initial,
                  max_iter, gtol):
    """Backtracking Newton on free coordinates with an active-set face.

    Active lower/upper faces are identified each iteration; only the inactive
    free block of the Hessian is factored, the step is clipped to the box, and
    Armijo backtracking runs on feasible points only. Fixed coordinates never
    move and are never updated by Newton. This is a genuinely constrained
    procedure, not an unconstrained fit followed by truncation.
    """
    free = [i for i in range(n) if i not in anchors]
    penalty_mask = [i not in anchors for i in range(n)]

    if initial is None:
        x = [0.0] * len(free)
    else:
        x0 = list(initial)
        x = [x0[idx] for idx in free]
    x = _project_bounds(x, bounds_by_free, free)
    theta = _anchored_theta(free, x, anchors, n)

    # All objects fixed: no decision variables; evaluate likelihood only.
    if not free:
        obj, ll, contribs, _, _ = _fill_masked(
            normalized, theta, lam, False, penalty_mask)
        if not math.isfinite(obj):
            return PLResult(
                status=ConvergenceStatus.NUMERICAL_ERROR, converged=False,
                iterations=0, theta=tuple(theta), log_likelihood=ll,
                penalized_objective=obj, log_contributions=tuple(contribs),
                gradient_norm=0.0,
                message="non-finite likelihood at the fixed abilities")
        return PLResult(
            status=ConvergenceStatus.CONVERGED, converged=True, iterations=0,
            theta=tuple(theta), log_likelihood=ll, penalized_objective=obj,
            log_contributions=tuple(contribs), gradient_norm=0.0,
            message=(
                f"all {n} object(s) fixed: likelihood evaluated at the "
                "caller-supplied abilities, no optimization performed"))

    m = len(free)
    status = ConvergenceStatus.MAX_ITERATIONS
    message = ""
    steps = 0

    for it in range(max_iter + 1):
        obj, ll, contribs, g_amb, H_amb = _fill_masked(
            normalized, theta, lam, True, penalty_mask)
        gf = [g_amb[idx] for idx in free]
        Hf = [[H_amb[a][b] for b in free] for a in free]
        gnorm = _projected_gradient_residual(gf, x, bounds_by_free, free)

        if not math.isfinite(obj) or not all(map(math.isfinite, gf)):
            status = ConvergenceStatus.NUMERICAL_ERROR
            message = "non-finite objective or gradient encountered"
            steps = it
            break
        if gnorm <= gtol:
            status = ConvergenceStatus.CONVERGED
            message = (f"projected gradient infinity norm {gnorm:.3e} <= "
                       f"gtol {gtol:.3e} after {it} constrained Newton "
                       f"step(s)")
            steps = it
            break
        if it == max_iter:
            status = ConvergenceStatus.MAX_ITERATIONS
            message = (f"budget of {max_iter} constrained Newton steps "
                       f"exhausted with projected gradient norm {gnorm:.3e} "
                       f"> gtol {gtol:.3e}")
            steps = max_iter
            break

        # Active set for a minimizer: pin at lower when g >= 0 and at upper
        # when g <= 0. Any other face gradient points toward the interior
        # optimum, so that face is released into the Newton block; the box
        # clip keeps every candidate step feasible.
        inactive = []
        for k, idx in enumerate(free):
            b = bounds_by_free.get(idx)
            interior = b is None or (not _at_lower(x[k], b)
                                     and not _at_upper(x[k], b))
            if interior:
                inactive.append(k)
            elif _at_lower(x[k], b) and gf[k] >= 0.0:
                pass  # pinned at lower face (KKT: g >= 0)
            elif _at_upper(x[k], b) and gf[k] <= 0.0:
                pass  # pinned at upper face (KKT: g <= 0)
            else:
                inactive.append(k)  # face released: move inward

        d = [0.0] * m
        directional = 0.0
        if inactive:
            q = len(inactive)
            Hr = [[Hf[inactive[a]][inactive[b]] for b in range(q)]
                  for a in range(q)]
            gr = [gf[k] for k in inactive]
            diag_scale = max(Hr[i][i] for i in range(q))
            L = None
            # The inactive free block is PD for lambda > 0, but eliminating
            # fixed coordinates can introduce cancellation on the scale of
            # machine epsilon; bump by tau * max(diag, 1) with an absolute
            # floor so tiny diagonal scales still get a usable bump.
            for tau in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
                Hrj = [row[:] for row in Hr]
                if tau:
                    bump = max(tau * diag_scale, tau)
                    for i in range(q):
                        Hrj[i][i] += bump
                try:
                    L = _cholesky(Hrj)
                    break
                except PLInputError:
                    continue
            if L is None:
                status = ConvergenceStatus.NUMERICAL_ERROR
                message = "inactive-block Newton matrix was not positive definite"
                steps = it
                break
            di = _solve_cholesky(L, [-v for v in gr])
            for pos, k in enumerate(inactive):
                d[k] = di[pos]
            directional = math.fsum(gr[a] * di[a] for a in range(q))
            if not math.isfinite(directional):
                status = ConvergenceStatus.NUMERICAL_ERROR
                message = "non-finite constrained Newton direction"
                steps = it
                break

        alpha_max = _max_step_to_box(x, d, bounds_by_free, free)

        # If the Newton direction points out of the box from a released face
        # (the unconstrained optimum lies beyond the wall), no positive
        # Newton steplength is feasible. Use a projected-gradient feasible
        # direction instead; this is a constrained step, not truncation of
        # an unconstrained fit.
        if directional < 0.0 and alpha_max == 0.0:
            inactive_set = set(inactive)
            d = [0.0] * m
            for k in inactive_set:
                # A released face's gradient sign guarantees -g points back
                # into the feasible box.
                d[k] = -gf[k]
            directional = math.fsum(gf[k] * d[k] for k in range(m))
            if directional < 0.0:
                alpha_max = _max_step_to_box(x, d, bounds_by_free, free)

        # Machine-precision stop: if the maximum Newton update cannot move
        # any free coordinate by more than ~100 ulps relative to its scale,
        # no backtracking can improve the point. This is reported as
        # convergence (the floating-point minimizer is reached), not as a
        # line-search failure.
        # Scale is absolute for near-zero coordinates (1 ulp(1) ~ 2e-16),
        # relative otherwise.
        # Machine-resolution stop applies only when no coordinate sits on a
        # newly reached face; pinned-at-boundary coordinates must first be
        # re-examined under the fresh active set at the next iteration.
        on_face = any(
            bounds_by_free.get(idx) is not None and (
                _at_lower(x[k], bounds_by_free[idx])
                or _at_upper(x[k], bounds_by_free[idx]))
            for k, idx in enumerate(free))
        if not on_face:
            step_scale = max(
                1.0, max((abs(x[k]) for k in range(m)), default=0.0))
            max_move = max(
                (alpha_max * abs(d[k]) for k in range(m)), default=0.0)
            if max_move <= 1e-14 * step_scale:
                status = ConvergenceStatus.CONVERGED
                message = (f"Newton step {max_move:.3e} at the floating-point "
                           f"resolution (scale {step_scale:.3e}) after {it} "
                           "constrained Newton step(s)")
                steps = it
                break

        if directional >= 0.0:
            # The Newton face gives no descent (e.g. wrong face after a
            # boundary landing): take a feasible negative-gradient step on
            # non-pinned coordinates instead, still strictly within the box.
            inactive_set = set(inactive)
            d = [-gf[k] if k in inactive_set else 0.0 for k in range(m)]
            directional = math.fsum(gf[k] * d[k] for k in range(m))
            if directional >= 0.0 or not math.isfinite(directional):
                # The current face is stationary. Re-evaluate the KKT
                # residual: if the face choice is wrong the residual finds a
                # blocking sign; otherwise we are at a box optimum.
                face_gnorm = _projected_gradient_residual(
                    gf, x, bounds_by_free, free)
                if face_gnorm <= gtol:
                    status = ConvergenceStatus.CONVERGED
                    message = (f"projected gradient infinity norm "
                               f"{face_gnorm:.3e} <= gtol {gtol:.3e} after "
                               f"{it} constrained Newton step(s)")
                    steps = it
                    break
                status = ConvergenceStatus.NUMERICAL_ERROR
                message = ("active face is stationary but boundary KKT "
                           f"residual {face_gnorm:.3e} exceeds gtol")
                steps = it
                break
            alpha_max = _max_step_to_box(x, d, bounds_by_free, free)

        alpha = alpha_max
        accepted = False
        for _ in range(_MAX_LINESEARCH_TRIES):
            xn = [x[k] + alpha * d[k] for k in range(m)]
            xn = _project_bounds(xn, bounds_by_free, free)
            thetan = _anchored_theta(free, xn, anchors, n)
            fn = _anchored_objective(normalized, thetan, lam, penalty_mask)
            finite = math.isfinite(fn)
            armijo = fn <= obj + _ARMIJO_C1 * alpha * directional
            # At machine precision a full Newton (or max-to-face constrained)
            # step can leave the rounded objective unchanged; that is the
            # converged regime and must be accepted rather than backtracked
            # into zero-length steps.
            exact_no_change = alpha == alpha_max and alpha > 0.0 and fn == obj
            if finite and (armijo or exact_no_change):
                if xn == x:
                    accepted = False
                    break
                x, theta = xn, thetan
                accepted = True
                break
            alpha *= _STEP_SHRINK

        if not accepted:
            half_decrement = -0.5 * directional
            if directional < 0.0 and half_decrement <= 1e-8:
                status = ConvergenceStatus.CONVERGED
                message = (f"Armijo step below rounding at iteration {it + 1}; "
                           f"Newton model decrease {2.0 * half_decrement:.3e}")
                steps = it
                break
            status = ConvergenceStatus.LINE_SEARCH_FAILED
            message = (f"backtracking failed after {_MAX_LINESEARCH_TRIES} "
                       f"halvings at iteration {it + 1}")
            steps = it
            break

    final_obj, final_ll, final_contribs, fg_amb, _ = _fill_masked(
        normalized, theta, lam, True, penalty_mask)
    fg = [fg_amb[idx] for idx in free]
    final_gnorm = _projected_gradient_residual(
        fg, x, bounds_by_free, free)
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
        boundary_kkt_residual=final_gnorm,
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


def ranking_log_probability(ranking, theta, participants=None, fixed_theta=None):
    """Stable log P of one tie-free top-k ranking under abilities theta.

    participants=None means all objects. Unranked participants stay in every
    remaining denominator; non-participants are excluded. k=0 returns 0.0.
    """
    if fixed_theta is None:
        theta = _validate_theta_arg(theta)
    else:
        n0 = len(theta)
        anchors0 = _validate_fixed(fixed_theta, n0)
        theta = _validate_anchored_initial(theta, n0, anchors0, {})
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


def ranking_probability(ranking, theta, participants=None, fixed_theta=None):
    """Probability of one partial ranking (may underflow for extremely small P)."""
    return math.exp(ranking_log_probability(
        ranking, theta, participants, fixed_theta))


def fit_plackett_luce(observations, n_objects, *, lam=1.0, initial=None,
                      max_iter=100, gtol=1e-10, fixed_theta=None,
                      bounds=None):
    """Fit penalized Plackett-Luce abilities by backtracking Newton.

    Arguments:
        observations: iterable of Observation (validated as one batch before
            any iteration).
        n_objects:    int in [1, 20], object indices span [0, n_objects).
        lam:          penalty weight lambda > 0 (finite).
        initial:      optional length-n finite abilities with sum ~0; rejected
            otherwise. Defaults to zeros.
        max_iter:     Newton-step budget (int >= 1, bool rejected).
        gtol:         convergence tolerance on the reduced/projected gradient
            infinity norm (finite float > 0).
        fixed_theta:  optional {object_index: ability} mapping (or iterable
            of pairs) anchoring named objects at finite ability constants.
            Empty/None keeps the legacy sum(theta)=0 calibration. With at
            least one anchor the sum-zero constraint is removed entirely,
            anchors are not recentered, and the lambda penalty acts on FREE
            coordinates only; fixed abilities still enter every denominator.
            Duplicate anchors, unknown indices and non-finite values raise
            PLInputError before any iteration.
        bounds:       optional {object_index: (lower, upper)} mapping (or a
            length-n sequence with None entries) of FINITE box bounds on FREE
            coordinates. Allowed only together with fixed_theta; endpoints
            must be finite and lower <= upper, and a start/iterate is moved
            by constrained active-set Newton steps with backtracking (never
            by post-hoc truncation).

    Returns PLResult. Budget exhaustion is reported as
    ConvergenceStatus.MAX_ITERATIONS; invalid/infeasible inputs raise
    PLInputError before any optimization occurs.
    """
    n = _checked_int(n_objects, "n_objects", minimum=1, maximum=MAX_OBJECTS)
    lam = _checked_finite_float(lam, "lam", positive=True)
    max_iter = _checked_int(max_iter, "max_iter", minimum=1)
    gtol = _checked_finite_float(gtol, "gtol", positive=True)

    norm = _normalize(observations, n)

    if fixed_theta is None:
        anchors = {}
    else:
        anchors = _validate_fixed(fixed_theta, n)

    if not anchors:
        if bounds is not None:
            raise PLInputError(
                "bounds: finite box bounds require at least one entry in "
                "fixed_theta (without anchors the sum-zero path is used)")
        if initial is None:
            theta = [0.0] * n
        else:
            theta = _validate_theta(initial, n, "initial")
            if n == 1:
                theta = [0.0]
    else:
        bounds_by_free = _validate_bounds(bounds, n, anchors)
        if initial is not None:
            initial = _validate_anchored_initial(
                initial, n, anchors, bounds_by_free)
        return _fit_anchored(norm, n, lam, anchors, bounds_by_free, initial,
                             max_iter, gtol)

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

        step_scale = max(1.0, max((abs(v) for v in x), default=0.0))
        if max(abs(v) for v in d) <= 1e-13 * step_scale:
            status = ConvergenceStatus.CONVERGED
            message = (f"Newton step at floating-point resolution after {it} "
                       "Newton step(s)")
            steps = it
            break

        alpha = 1.0
        accepted = False
        for _ in range(_MAX_LINESEARCH_TRIES):
            xn = [x[i] + alpha * d[i] for i in range(n - 1)]
            thetan = _theta_from_free(xn)
            fn = _objective_only(norm, thetan, lam)
            if math.isfinite(fn) and (
                    fn <= obj + _ARMIJO_C1 * alpha * directional
                    or (alpha == 1.0 and fn == obj)):
                if xn == x:
                    accepted = False
                    break
                x, theta = xn, thetan
                accepted = True
                break
            alpha *= _STEP_SHRINK

        if not accepted:
            if directional < 0.0 and -0.5 * directional <= 1e-8:
                status = ConvergenceStatus.CONVERGED
                message = (f"Armijo step below rounding at iteration {it + 1}; "
                           f"Newton model decrease {(-directional):.3e}")
                steps = it
                break
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


def fit_anchored(observations, n_objects, fixed_theta, *, lam=1.0,
                 bounds=None, initial=None, max_iter=100, gtol=1e-10):
    """Fit with caller-anchored ability constants.

    Convenience wrapper around fit_plackett_luce(..., fixed_theta=...). It
    REQUIRES at least one anchor: the zero-anchor (sum-zero) path stays on
    fit_plackett_luce. See fit_plackett_luce for the full contract.

    Arguments:
        fixed_theta: mapping {object_index: ability} or iterable of
            (object_index, ability) pairs; abilities finite, indices known
            and distinct.
        bounds:      optional finite box bounds on FREE objects only.
        initial:     optional ambient start; fixed slots must equal anchors.

    Returns PLResult. With every object fixed the result is a pure
    likelihood evaluation (0 iterations, converged).
    """
    n = _checked_int(n_objects, "n_objects", minimum=1, maximum=MAX_OBJECTS)
    anchors = _validate_fixed(fixed_theta, n)
    if not anchors:
        raise PLInputError(
            "fit_anchored: at least one anchored object is required; use "
            "fit_plackett_luce for the sum(theta)=0 calibration")
    return fit_plackett_luce(
        observations, n_objects, lam=lam, initial=initial,
        max_iter=max_iter, gtol=gtol, fixed_theta=anchors, bounds=bounds)


def _anchored_derivatives(observations, theta, lam, fixed_theta, want_H):
    n = len(theta)
    anchors = _validate_fixed(fixed_theta, n)
    lam = _checked_finite_float(lam, "lam", positive=True)
    if isinstance(theta, (str, bytes)) or not hasattr(theta, "__iter__"):
        raise PLInputError(f"theta: expected a sequence of {n} real numbers")
    vals = list(theta)
    if len(vals) != n:
        raise PLInputError(f"theta: expected length {n}, got {len(vals)}")
    out = [_checked_finite_float(v, f"theta[{i}]") for i, v in enumerate(vals)]
    for idx, val in anchors.items():
        if out[idx] != val:
            raise PLInputError(
                f"theta[{idx}]: must equal the anchored value {val!r}")
    norm = _normalize(observations, n)
    free = [i for i in range(n) if i not in anchors]
    mask = [i not in anchors for i in range(n)]
    obj, ll, contribs, g, H = _fill_masked(norm, out, lam, True, mask)
    gf = [g[i] for i in free]
    if want_H:
        Hf = [[H[a][b] for b in free] for a in free]
        return free, gf, Hf
    return free, gf


def anchored_penalized_objective(observations, theta, lam, fixed_theta):
    """Anchored objective: -loglik + lambda/2 * sum over FREE theta_i**2."""
    n = len(theta)
    anchors = _validate_fixed(fixed_theta, n)
    lam = _checked_finite_float(lam, "lam", positive=True)
    if isinstance(theta, (str, bytes)) or not hasattr(theta, "__iter__"):
        raise PLInputError(f"theta: expected a sequence of {n} real numbers")
    vals = list(theta)
    if len(vals) != n:
        raise PLInputError(f"theta: expected length {n}, got {len(vals)}")
    out = [_checked_finite_float(v, f"theta[{i}]") for i, v in enumerate(vals)]
    for idx, val in anchors.items():
        if out[idx] != val:
            raise PLInputError(
                f"theta[{idx}]: must equal the anchored value {val!r}")
    norm = _normalize(observations, n)
    mask = [i not in anchors for i in range(n)]
    return _anchored_objective(norm, out, lam, mask)


def anchored_gradient_free(observations, theta, lam, fixed_theta):
    """Analytic gradient of the anchored objective over free coordinates.

    Returns (free_indices, gradient) where gradient is ordered by ascending
    free object index. Fixed coordinates are excluded; their abilities affect
    the probabilities but never receive a derivative entry.
    """
    return _anchored_derivatives(
        observations, theta, lam, fixed_theta, False)


def anchored_hessian_free(observations, theta, lam, fixed_theta):
    """Analytic free-coordinate Hessian block: (free_indices, matrix)."""
    free, _gf, Hf = _anchored_derivatives(
        observations, theta, lam, fixed_theta, True)
    return free, Hf
