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
    OrderConstraint               -- one "before must finish ahead of after" edge
    ConstrainedOrderResult        -- summed likelihood over legal linear extensions
    constrained_order_probability -- acyclicity check + sum of full-order
                                     Plackett-Luce probabilities over every legal
                                     linear extension (<= 6 candidates)

Model (see README.md for the full derivation):
    For a ranking (r_1, ..., r_k) over participant set A,
        P = prod_{j=1..k} exp(theta_{r_j}) /
                          sum_{i in A \\ {r_1..r_{j-1}}} exp(theta_i)
    Unranked participants stay in every remaining denominator; objects not in A
    never enter any denominator. k = 0 contributes log P = 0.

Precedence constraints:
    Given a fixed candidate set C (|C| <= 6) and pairwise "a before b" edges,
    the constraints must form a DAG (a cycle is rejected with PLInputError
    before any computation). Every legal linear extension is a distinct FULL
    ranking of C (candidates never compared by the constraints are NOT tied:
    each completion is summed separately):
        L = sum_{pi in LE(constraints)} P(pi is the full order over C | theta)
    The number of summed extensions is returned as n_extensions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Optional

__all__ = [
    "MAX_OBJECTS",
    "MAX_CONSTRAINED_CANDIDATES",
    "ConvergenceStatus",
    "PLInputError",
    "Observation",
    "PLResult",
    "OrderConstraint",
    "ConstrainedOrderResult",
    "fit_plackett_luce",
    "ranking_log_probability",
    "ranking_probability",
    "penalized_objective",
    "penalized_gradient_free",
    "penalized_hessian_free",
    "constrained_order_probability",
]

MAX_OBJECTS = 20

# Precedence-constrained enumeration is an exact, exhaustive method: the
# candidate set of a single constrained-order query is capped at 6, so the
# worst case is 6! = 720 full-order probabilities.
MAX_CONSTRAINED_CANDIDATES = 6

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


@dataclass(frozen=True)
class OrderConstraint:
    """One strict precedence edge: ``before`` must rank ahead of ``after``.

    Transitivity is implied (a path a -> b -> c requires a before c); the
    library takes the transitive closure. An edge that closes a cycle makes
    the whole constraint set contradictory and is rejected up front.
    """

    before: int
    after: int


@dataclass(frozen=True)
class ConstrainedOrderResult:
    """Outcome of an exact constrained-order likelihood query.

    candidates:      the validated fixed candidate set.
    n_extensions:    number of legal linear extensions summed (0 only for a
                     contradictory input; a contradictory input raises, so a
                     returned result always has n_extensions >= 1).
    log_likelihood:  log of the summed probability, stably accumulated.
    likelihood:      exp(log_likelihood) of the summed probability.
    extensions:      every legal linear extension as a tuple best-first;
                     included only when return_extensions=True.
    """

    candidates: tuple[int, ...]
    n_extensions: int
    log_likelihood: float
    likelihood: float
    extensions: tuple[tuple[int, ...], ...] = ()


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


# --------------------------------------------------------------------------- #
# Precedence constraints: validation, acyclicity, linear-extension enumeration
# --------------------------------------------------------------------------- #
def _validate_candidates(candidates, n: int) -> tuple[int, ...]:
    where = "candidates"
    if isinstance(candidates, (str, bytes)) or not hasattr(candidates, "__iter__"):
        raise PLInputError(f"{where}: expected a sequence of int indices")
    out = _checked_index_seq(candidates, where, n)
    if len(out) == 0:
        raise PLInputError(f"{where}: candidate set must contain at least one object")
    if len(set(out)) != len(out):
        dup = sorted({i for i in out if out.count(i) > 1})
        raise PLInputError(
            f"{where}: duplicated candidate index/indices {dup}; the fixed "
            "candidate set must contain distinct objects")
    if len(out) > MAX_CONSTRAINED_CANDIDATES:
        raise PLInputError(
            f"{where}: {len(out)} candidates exceed the exact-enumeration "
            f"bound of {MAX_CONSTRAINED_CANDIDATES}")
    return out


def _validate_constraints(constraints, candidates: tuple[int, ...], n: int):
    """Validate precedence edges and return a position-indexed adjacency pair.

    Returns (pred, succ): sets of candidate positions, with the transitive
    closure taken. A self-loop or any directed cycle is rejected with a
    PLInputError that names the offending cycle. All checks finish before any
    probability is computed, so a rejected call leaves no partial state.
    """
    if constraints is None:
        constraints = ()
    if isinstance(constraints, (str, bytes)) or not hasattr(constraints, "__iter__"):
        raise PLInputError(
            "constraints: expected an iterable of OrderConstraint (or None)")

    m = len(candidates)
    pos = {c: i for i, c in enumerate(candidates)}
    succ: list[set[int]] = [set() for _ in range(m)]
    edges = []
    for idx, con in enumerate(constraints):
        where = f"constraints[{idx}]"
        if not isinstance(con, OrderConstraint):
            raise PLInputError(
                f"{where}: expected OrderConstraint, got {type(con).__name__}")
        b = _checked_int(con.before, f"{where}.before",
                         minimum=0, maximum=n - 1)
        a = _checked_int(con.after, f"{where}.after",
                         minimum=0, maximum=n - 1)
        if b not in pos:
            raise PLInputError(
                f"{where}: object {b} is not in the candidate set {list(candidates)}")
        if a not in pos:
            raise PLInputError(
                f"{where}: object {a} is not in the candidate set {list(candidates)}")
        if b == a:
            raise PLInputError(
                f"{where}: object {b} cannot be required to rank before itself")
        edges.append((pos[b], pos[a]))

    for u, v in edges:
        succ[u].add(v)

    cycle = _directed_cycle(succ)
    if cycle is not None:
        named = " -> ".join(str(candidates[p]) for p in cycle)
        raise PLInputError(
            "constraints: precedence edges contain a cycle "
            f"({named}); contradictory 'before' relations have no legal "
            "linear extension")

    return _transitive_closure(succ)


def _directed_cycle(succ: list[set[int]]):
    """Return one directed cycle as a list of positions (last node repeated),
    or None when the graph is acyclic. DFS coloring; |nodes| <= 6."""
    color = [0] * len(succ)  # 0 unseen, 1 on stack, 2 done
    stack: list[int] = []

    def dfs(u):
        color[u] = 1
        stack.append(u)
        for v in succ[u]:
            if color[v] == 1:
                start = stack.index(v)
                return stack[start:] + [v]
            if color[v] == 0:
                found = dfs(v)
                if found is not None:
                    return found
        stack.pop()
        color[u] = 2
        return None

    for u in range(len(succ)):
        if color[u] == 0:
            found = dfs(u)
            if found is not None:
                return found
    return None


def _transitive_closure(succ: list[set[int]]):
    """Return (pred, succ) after taking the transitive closure (Floyd style)."""
    m = len(succ)
    reach = [set(row) for row in succ]
    for mid in range(m):
        for u in range(m):
            if mid in reach[u]:
                reach[u].update(reach[mid])
    pred = [set() for _ in range(m)]
    for u in range(m):
        for v in reach[u]:
            pred[v].add(u)
    return pred, reach


def _enumerate_extensions(pred: list[set[int]], succ: list[set[int]]
                          ) -> list[tuple[int, ...]]:
    """All linear extensions of the poset via Kahn-style backtracking.

    Each extension is a tuple of candidate positions best first. Worst case
    m! = 720 for m = 6. Candidates never ordered against each other are not
    tied: each distinct completion is enumerated separately.
    """
    m = len(succ)
    remaining_pred = [set(p) for p in pred]
    chosen: list[int] = []
    chosen_mark: set[int] = set()
    extensions: list[tuple[int, ...]] = []

    def backtrack():
        if len(chosen) == m:
            extensions.append(tuple(chosen))
            return
        ready = [v for v in range(m)
                 if v not in chosen_mark and not remaining_pred[v]]
        for v in ready:
            chosen.append(v)
            chosen_mark.add(v)
            newly = [w for w in succ[v] if v in remaining_pred[w]]
            for w in newly:
                remaining_pred[w].discard(v)
            backtrack()
            for w in newly:
                remaining_pred[w].add(v)
            chosen.pop()
            chosen_mark.discard(v)

    backtrack()
    return extensions


def _full_order_log_probability(order_positions, candidates, theta):
    """Log Plackett-Luce probability of one FULL order over the candidate set.

    Every candidate that has not been picked yet stays in the denominator.
    """
    remaining = list(candidates)
    lp = 0.0
    for p in order_positions:
        winner = candidates[p]
        m = max(theta[i] for i in remaining)
        s = math.fsum(math.exp(theta[i] - m) for i in remaining)
        lp += theta[winner] - (m + math.log(s))
        remaining.remove(winner)
    return lp


# --------------------------------------------------------------------------- #
# Core likelihood accumulation (stable logsumexp, analytic gradient & Hessian)
# --------------------------------------------------------------------------- #
def _fill(normalized, theta, lam, want_derivs):
    """Return (objective, log_likelihood, per-obs log contributions, g, H).

    Ambient coordinates (length n). g/H include the lambda penalty. Stages use
    logsumexp with per-stage max subtraction for numerical stability.
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

    penalty_half = 0.5 * lam * math.fsum(t * t for t in theta)
    objective = -total_ll + penalty_half
    if want_derivs:
        for i in range(n):
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


def constrained_order_probability(constraints, candidates, theta, *,
                                  return_extensions=False):
    """Likelihood of precedence constraints = sum over legal linear extensions.

    Arguments:
        constraints: iterable of OrderConstraint (None/empty allowed). Each
            edge "a before b" is a strict precedence requirement; transitivity
            is implied. A directed cycle (contradictory requirements) raises
            PLInputError before any probability is computed.
        candidates:  fixed distinct candidate set (1 <= |candidates| <= 6,
            indices in [0, n)). Every legal linear extension is a distinct
            FULL ranking of exactly these candidates; unordered pairs are
            alternatives, never ties.
        theta:       length-n finite abilities with sum(theta) = 0.
        return_extensions: when True, include every enumerated extension
            (best-first tuples) in the result. Must be a real bool.

    Returns ConstrainedOrderResult with the summed likelihood, its log, and
    the number of extensions summed. With no constraints the m! full orders
    partition the sample space (sum = 1); a total chain leaves one extension
    and the likelihood equals that full order's Plackett-Luce probability.
    """
    if not isinstance(return_extensions, bool):
        raise PLInputError(
            "return_extensions: expected bool, got "
            f"{type(return_extensions).__name__}")
    theta = _validate_theta_arg(theta)
    n = len(theta)
    cand = _validate_candidates(candidates, n)
    pred, succ = _validate_constraints(constraints, cand, n)

    ext_positions = _enumerate_extensions(pred, succ)
    # Defensive: acyclicity was already proved, so at least one extension must
    # exist; fail loudly rather than silently returning an empty sum.
    if not ext_positions:
        raise PLInputError(
            "constraints: internal error: acyclic constraints yielded no "
            "linear extension")

    log_probs = [_full_order_log_probability(order, cand, theta)
                 for order in ext_positions]
    log_total = _logsumexp_values(log_probs)

    extensions: tuple[tuple[int, ...], ...] = ()
    if return_extensions:
        extensions = tuple(tuple(cand[p] for p in order)
                           for order in ext_positions)

    return ConstrainedOrderResult(
        candidates=cand,
        n_extensions=len(ext_positions),
        log_likelihood=log_total,
        likelihood=math.exp(log_total),
        extensions=extensions,
    )


def _logsumexp_values(values):
    """log of a sum of probabilities from their logs (stable max subtraction)."""
    m = max(values)
    return m + math.log(math.fsum(math.exp(v - m) for v in values))


def fit_plackett_luce(observations, n_objects, *, lam=1.0, initial=None,
                      max_iter=100, gtol=1e-10):
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

    Returns PLResult. Budget exhaustion is reported as
    ConvergenceStatus.MAX_ITERATIONS; invalid/infeasible inputs raise
    PLInputError before any optimization occurs.
    """
    n = _checked_int(n_objects, "n_objects", minimum=1, maximum=MAX_OBJECTS)
    lam = _checked_finite_float(lam, "lam", positive=True)
    max_iter = _checked_int(max_iter, "max_iter", minimum=1)
    gtol = _checked_finite_float(gtol, "gtol", positive=True)

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
