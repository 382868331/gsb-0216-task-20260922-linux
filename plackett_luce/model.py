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
    OrderingConstraint            -- "before must rank ahead of after" pair
    ConstrainedLikelihood         -- poset likelihood: P, log P, extension count
    MAX_CANDIDATES                -- candidate-set bound for poset likelihood (6)
    constrained_ranking_probability -- sum ranking P over all linear extensions

Model (see README.md for the full derivation):
    For a ranking (r_1, ..., r_k) over participant set A,
        P = prod_{j=1..k} exp(theta_{r_j}) /
                          sum_{i in A \\ {r_1..r_{j-1}}} exp(theta_i)
    Unranked participants stay in every remaining denominator; objects not in A
    never enter any denominator. k = 0 contributes log P = 0.

Precedence constraints:
    Given a fixed candidate set A (|A| <= 6) and strict precedence pairs
    (before, after), the constraints are first checked to form an acyclic
    poset; the likelihood is the exact sum of the staged ranking probability
    above over every linear extension (full permutation of A satisfying the
    poset). Incomparable objects are never treated as tied: each extension is
    a separate permutation. A cyclic (contradictory) constraint set raises
    PLInputError before any enumeration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Optional

__all__ = [
    "MAX_OBJECTS",
    "MAX_CANDIDATES",
    "ConvergenceStatus",
    "PLInputError",
    "Observation",
    "PLResult",
    "OrderingConstraint",
    "ConstrainedLikelihood",
    "fit_plackett_luce",
    "ranking_log_probability",
    "ranking_probability",
    "constrained_ranking_probability",
    "penalized_objective",
    "penalized_gradient_free",
    "penalized_hessian_free",
]

MAX_OBJECTS = 20

# Hard bound on the candidate set for the exact linear-extension enumeration.
# 6! = 720 full permutations at most, so exhaustive exact summation is cheap.
MAX_CANDIDATES = 6

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
class OrderingConstraint:
    """One strict precedence pair: ``before`` must rank ahead of ``after``.

    Both fields are int object indices in [0, n_objects). The pair must not
    name the same object twice (a self-loop is a contradiction). A batch of
    constraints is accepted only if its directed graph is acyclic; a cycle
    (including a self-loop) raises PLInputError before any enumeration.
    """

    before: int
    after: int


@dataclass(frozen=True)
class ConstrainedLikelihood:
    """Exact likelihood of a poset over a fixed candidate set.

    probability:       sum of the full-ranking Plackett-Luce probabilities of
                       every legal linear extension (may underflow to 0.0).
    log_probability:   stable ``log`` of that sum (logsumexp over extensions).
    n_extensions:      number of legal linear extensions (>= 1 when acyclic).
    candidate_set:     the materialized sorted tuple of candidates used.
    constraints:       the materialized distinct (before, after) pairs used.
    """

    probability: float
    log_probability: float
    n_extensions: int
    candidate_set: tuple[int, ...]
    constraints: tuple[tuple[int, int], ...]


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
# Core likelihood accumulation (stable logsumexp, analytic gradient & Hessian)
# --------------------------------------------------------------------------- #
def _staged_log_probability(rank, part, theta):
    """Log probability of one full/partial ranking over participant set part.

    Pure staged logsumexp, shared by the public ranking function and the exact
    linear-extension summation (so the new capability reuses the existing,
    already-tested ranking probability rather than re-deriving it).
    """
    remaining = list(part)
    lp = 0.0
    for winner in rank:
        m = max(theta[i] for i in remaining)
        s = math.fsum(math.exp(theta[i] - m) for i in remaining)
        lp += theta[winner] - (m + math.log(s))
        remaining.remove(winner)
    return lp


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
    return _staged_log_probability(rank, part, theta)


def ranking_probability(ranking, theta, participants=None):
    """Probability of one partial ranking (may underflow for extremely small P)."""
    return math.exp(ranking_log_probability(ranking, theta, participants))


# --------------------------------------------------------------------------- #
# Precedence constraints over a fixed candidate set (exact, |A| <= 6)
# --------------------------------------------------------------------------- #
def _normalize_constraint_set(constraints, candidates, n):
    """Validate candidates + precedence pairs up front.

    Returns (candidate_tuple, distinct edge tuple, adjacency, indegree).
    Nothing is enumerated before the whole input (including acyclicity) has
    been accepted, so a rejected call leaves no partial state.
    """
    if candidates is None:
        cand = tuple(range(n))
    else:
        cand = _checked_index_seq(candidates, "candidate_set", n)
        if len(set(cand)) != len(cand):
            dup = sorted({i for i in cand if cand.count(i) > 1})
            raise PLInputError(
                f"candidate_set: duplicated candidate index/indices {dup}")
    if len(cand) == 0:
        raise PLInputError("candidate_set: at least one candidate is required")
    if len(cand) > MAX_CANDIDATES:
        raise PLInputError(
            f"candidate_set: size {len(cand)} exceeds the exact-enumeration "
            f"bound of {MAX_CANDIDATES} candidates")

    if isinstance(constraints, (str, bytes)) or not hasattr(constraints, "__iter__"):
        raise PLInputError("constraints: expected an iterable of OrderingConstraint")

    in_set = set(cand)
    edges = set()
    for t, con in enumerate(constraints):
        where = f"constraints[{t}]"
        if not isinstance(con, OrderingConstraint):
            raise PLInputError(
                f"{where}: expected OrderingConstraint, got {type(con).__name__}")
        b = _checked_int(con.before, f"{where}.before", minimum=0, maximum=n - 1)
        a = _checked_int(con.after, f"{where}.after", minimum=0, maximum=n - 1)
        if b not in in_set:
            raise PLInputError(
                f"{where}: 'before' object {b} is not in the candidate set")
        if a not in in_set:
            raise PLInputError(
                f"{where}: 'after' object {a} is not in the candidate set")
        edges.add((b, a))

    adjacency = {i: set() for i in cand}
    indegree = {i: 0 for i in cand}
    for b, a in edges:
        if a not in adjacency[b]:
            adjacency[b].add(a)
            indegree[a] += 1
    return cand, tuple(sorted(edges)), adjacency, indegree


def _assert_acyclic(cand, adjacency, indegree):
    """Kahn's algorithm; raise PLInputError naming a cycle if one exists.

    Self-loops ((x, x)) are cycles too and are rejected here.
    """
    remaining_indegree = dict(indegree)
    ready = sorted(i for i in cand if remaining_indegree[i] == 0)
    seen = 0
    while ready:
        node = ready.pop()
        seen += 1
        for nxt in sorted(adjacency[node]):
            remaining_indegree[nxt] -= 1
            if remaining_indegree[nxt] == 0:
                ready.append(nxt)
    if seen != len(cand):
        cycle_nodes = sorted(
            i for i in cand if remaining_indegree[i] > 0)
        raise PLInputError(
            "constraints: precedence graph contains a cycle among objects "
            f"{cycle_nodes}; contradictory (cyclic) orderings have no legal "
            "linear extension")


def _linear_extensions(cand, adjacency, indegree):
    """Yield every topological ordering (linear extension) as a tuple.

    Deterministic, smallest-available-first backtracking. At most 6! = 720
    permutations are produced because the candidate set is bounded by
    MAX_CANDIDATES.
    """
    order = []
    chosen = set()
    current = dict(indegree)

    def extend():
        if len(order) == len(cand):
            yield tuple(order)
            return
        for node in cand:
            if node not in chosen and current[node] == 0:
                for nxt in adjacency[node]:
                    current[nxt] -= 1
                chosen.add(node)
                order.append(node)
                yield from extend()
                order.pop()
                chosen.remove(node)
                for nxt in adjacency[node]:
                    current[nxt] += 1

    yield from extend()


def constrained_ranking_probability(constraints, theta, candidate_set=None):
    """Exact likelihood of precedence constraints over a fixed candidate set.

    The constraints are strict "before ranks ahead of after" pairs. They are
    first checked to form an acyclic partial order; the likelihood is then the
    exact sum of the full Plackett-Luce ranking probability over every legal
    linear extension of the candidates (incomparable objects are enumerated as
    distinct orders, never treated as tied).

    Arguments:
        constraints:   iterable of OrderingConstraint (validated as one batch).
        theta:         finite length-n abilities with sum(theta) = 0.
        candidate_set: distinct int indices, or None for all n objects. The
            set must contain 1..MAX_CANDIDATES (6) elements; every constraint
            endpoint must belong to it. Candidates outside the set are exactly
            like non-participants: they never enter any denominator.

    Returns ConstrainedLikelihood (probability, log_probability, n_extensions,
    candidate_set, constraints). A cyclic/contradictory constraint set, an
    out-of-range or oversized candidate set, or invalid theta raises
    PLInputError before any enumeration.
    """
    theta = _validate_theta_arg(theta)
    n = len(theta)
    cand, edges, adjacency, indegree = _normalize_constraint_set(
        constraints, candidate_set, n)
    _assert_acyclic(cand, adjacency, indegree)

    log_terms = []
    n_extensions = 0
    for perm in _linear_extensions(cand, adjacency, indegree):
        log_terms.append(_staged_log_probability(perm, cand, theta))
        n_extensions += 1

    # Sum probabilities exactly via logsumexp over the (at most 720) terms.
    m = max(log_terms)
    log_probability = m + math.log(math.fsum(math.exp(v - m) for v in log_terms))
    return ConstrainedLikelihood(
        probability=math.exp(log_probability),
        log_probability=log_probability,
        n_extensions=n_extensions,
        candidate_set=cand,
        constraints=edges,
    )


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
