"""Plackett-Luce fitting for partial (top-k) rankings, standard library only.

Public surface (see ``plackett_luce`` package ``__init__``):

* :class:`Ranking`           -- one top-k observation with an optional
                                participant (eligible) set.
* :class:`FitResult`         -- fit status, abilities and per-observation
                                contributions.
* :func:`fit_plackett_luce`  -- penalised maximum-likelihood fit on the
                                ``sum(theta) == 0`` hyperplane.
* :func:`log_ranking_likelihood` / :func:`ranking_probability`
* :func:`winner_probabilities`

Model
-----
For an observation ``(i_1, ..., i_k)`` chosen from participant set *S*
(``k <= |S|``), the Plackett-Luce likelihood is the sequential softmax

    prod_{j=1..k} exp(theta_{i_j}) /
                  sum_{a in S \\ {i_1,...,i_{j-1}}} exp(theta_a).

Participants that are not ranked stay in every remaining denominator;
objects outside *S* never enter any denominator.  ``k = 0`` contributes 1.

The optimised objective is

    F(theta) = -sum_obs log p(obs | theta) + lambda/2 * ||theta||_2^2

restricted to ``sum(theta) = 0``.  With ``lambda > 0`` the restricted
objective is strictly convex, so a finite unique optimum always exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_OBJECTS = 20
"""Hard interface bound on the number of modelled objects."""

CONVERGED = "converged"
"""Newton iteration satisfied the projected-gradient tolerance."""

MAX_ITERATIONS = "max_iterations"
"""Search budget exhausted before the tolerance was met (not a failure to
prove existence: with lambda > 0 a unique optimum always exists)."""

LINE_SEARCH_FAILED = "line_search_failed"
"""Backtracking could not find a step giving sufficient decrease; the
iteration cannot make further numerical progress.  Distinct from
:data:`MAX_ITERATIONS`."""

_DEFAULT_TOL = 1e-9
_DEFAULT_MAX_ITER = 100
_C1 = 1e-4            # Armijo sufficient-decrease constant
_SHRINK = 0.5         # backtracking factor
_MAX_HALVINGS = 80    # ~0.5**80 minimum step before declaring failure

# One "selection level" of an observation: (winner index, tuple of objects
# still in the denominator at that level).
Level = tuple[int, tuple[int, ...]]
PreparedObs = tuple[tuple[int, ...], tuple[int, ...], list[Level]]


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ranking:
    """One top-k ranking observation.

    ``ranked`` holds the ranked object indices in order (best first);
    it must contain distinct indices and may be empty (``k = 0``).
    ``participants`` is the optional eligible set for this observation:
    every ranked index must belong to it, indices must be distinct and
    it must be non-empty.  ``None`` means "all modelled objects".
    """

    ranked: Sequence[int] = ()
    participants: Sequence[int] | None = None


@dataclass(frozen=True)
class ObservationContribution:
    """Per-observation quantities evaluated at the fitted theta."""

    index: int
    ranked: tuple[int, ...]
    participants: tuple[int, ...]
    log_likelihood: float
    gradient: tuple[float, ...]
    """Gradient of this observation's log-likelihood w.r.t. theta
    (unpenalised)."""


@dataclass(frozen=True)
class FitResult:
    converged: bool
    """True exactly when ``status == CONVERGED``."""

    status: str
    """One of :data:`CONVERGED`, :data:`MAX_ITERATIONS`,
    :data:`LINE_SEARCH_FAILED`."""

    iterations: int
    """Number of Newton steps actually taken."""

    n_objects: int
    theta: tuple[float, ...]
    """Fitted abilities, satisfying ``sum(theta) == 0`` up to rounding."""

    log_likelihood: float
    """Total unpenalised log-likelihood at ``theta``."""

    objective: float
    """Penalised negative log-likelihood at ``theta``."""

    penalty: float
    """``lambda/2 * ||theta||**2`` at ``theta``."""

    grad_norm: float
    """Infinity norm of the projected (sum-zero) gradient on return."""

    contributions: tuple[ObservationContribution, ...]

    @property
    def abilities(self) -> tuple[float, ...]:
        """Alias for :attr:`theta`."""
        return self.theta


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _as_int(x, where: str) -> int:
    if isinstance(x, bool):
        raise TypeError(f"{where} must be an int; bool is not accepted")
    if not isinstance(x, int):
        raise TypeError(f"{where} must be an int, got {type(x).__name__}")
    return x


def _as_finite_float(x, where: str) -> float:
    if isinstance(x, bool):
        raise TypeError(f"{where} must be a real number; bool is not accepted")
    if not isinstance(x, (int, float)):
        raise TypeError(
            f"{where} must be a real number, got {type(x).__name__}"
        )
    fx = float(x)
    if not math.isfinite(fx):
        raise ValueError(f"{where} must be finite (NaN/Infinity rejected)")
    return fx


def _checked_index(x, obs_label: str, where: str) -> int:
    if isinstance(x, bool):
        raise TypeError(
            f"{obs_label}: {where} must be int; bool is not accepted"
        )
    if not isinstance(x, int):
        raise TypeError(
            f"{obs_label}: {where} must be int, got {type(x).__name__}"
        )
    return x


def _distinct(values, obs_label: str, where: str) -> tuple[int, ...]:
    out: list[int] = []
    seen: set[int] = set()
    for v in values:
        idx = _checked_index(v, obs_label, where)
        if idx in seen:
            raise ValueError(
                f"{obs_label}: duplicate object {idx} in {where}"
            )
        seen.add(idx)
        out.append(idx)
    return tuple(out)


def _as_sequence(values, obs_label: str, where: str) -> list:
    if isinstance(values, (str, bytes)) or not isinstance(
        values, Sequence
    ):
        raise TypeError(f"{obs_label}: {where} must be a sequence of ints")
    return list(values)


def _normalize_rankings(
    rankings: Iterable[Ranking | Sequence[int]],
    n_objects: int | None,
) -> tuple[int, list[PreparedObs]]:
    if isinstance(rankings, Ranking) or isinstance(rankings, (str, bytes)):
        raise TypeError(
            "rankings must be an iterable of Ranking or index sequences, "
            "not a single Ranking/str"
        )
    try:
        raw_obs = list(rankings)
    except TypeError as exc:
        raise TypeError("rankings must be an iterable of observations") from exc

    parsed: list[tuple[tuple[int, ...], tuple[int, ...] | None]] = []
    inferred = 0
    for j, raw in enumerate(raw_obs):
        label = f"observation {j}"
        if isinstance(raw, Ranking):
            ranked_raw, parts_raw = raw.ranked, raw.participants
        elif isinstance(raw, (tuple, list)):
            ranked_raw, parts_raw = raw, None
        else:
            raise TypeError(
                f"{label}: must be a Ranking or tuple/list of object "
                f"indices, got {type(raw).__name__}"
            )

        ranked = _distinct(_as_sequence(ranked_raw, label, "ranked"), label, "ranked")
        participants: tuple[int, ...] | None = None
        if parts_raw is not None:
            participants = _distinct(
                _as_sequence(parts_raw, label, "participants"),
                label,
                "participants",
            )
            if not participants:
                raise ValueError(f"{label}: participants must be non-empty")

        if participants is not None:
            part_set = set(participants)
            for w in ranked:
                if w not in part_set:
                    raise ValueError(
                        f"{label}: ranked object {w} is not in the "
                        "participant set"
                    )
            if len(ranked) > len(participants):
                raise ValueError(
                    f"{label}: ranked {len(ranked)} objects but only "
                    f"{len(participants)} participants"
                )
        for v in ranked:
            if v < 0:
                raise ValueError(f"{label}: object index {v} is negative")
            inferred = max(inferred, v + 1)
        if participants is not None:
            for v in participants:
                if v < 0:
                    raise ValueError(f"{label}: object index {v} is negative")
                inferred = max(inferred, v + 1)
        parsed.append((ranked, participants))

    if n_objects is None:
        if inferred == 0:
            raise ValueError(
                "n_objects cannot be inferred: no object indices present; "
                "pass n_objects explicitly"
            )
        n = inferred
    else:
        n = _as_int(n_objects, "n_objects")
        if not 1 <= n <= MAX_OBJECTS:
            raise ValueError(
                f"n_objects must be in [1, {MAX_OBJECTS}], got {n}"
            )
        if inferred > n:
            raise ValueError(
                f"an observation references object index {inferred - 1} "
                f"but n_objects={n}"
            )

    prepared: list[PreparedObs] = []
    for j, (ranked, participants) in enumerate(parsed):
        label = f"observation {j}"
        if participants is None:
            participants = tuple(range(n))
        for v in ranked:
            if v >= n:
                raise ValueError(
                    f"{label}: object index {v} out of range for "
                    f"n_objects={n}"
                )
        for v in participants:
            if v >= n:
                raise ValueError(
                    f"{label}: participant index {v} out of range for "
                    f"n_objects={n}"
                )
        remaining = list(participants)
        levels: list[Level] = []
        for w in ranked:
            levels.append((w, tuple(remaining)))
            remaining.remove(w)
        prepared.append((ranked, participants, levels))

    return n, prepared


# ---------------------------------------------------------------------------
# Core numerics
# ---------------------------------------------------------------------------


def _negative_log_likelihood(
    theta: Sequence[float], levels_by_obs: Iterable[Iterable[Level]]
) -> float:
    f = 0.0
    for levels in levels_by_obs:
        for winner, remaining in levels:
            m = max(theta[i] for i in remaining)
            z = 0.0
            for i in remaining:
                z += math.exp(theta[i] - m)
            f += m + math.log(z) - theta[winner]
    return f


def _objective(
    theta: Sequence[float],
    levels_by_obs: Iterable[Iterable[Level]],
    lam: float,
) -> float:
    f = _negative_log_likelihood(theta, levels_by_obs)
    ridge = 0.0
    for t in theta:
        ridge += t * t
    return f + 0.5 * lam * ridge


def _objective_gradient_hessian(
    theta: Sequence[float],
    levels_by_obs: list[list[Level]],
    lam: float,
) -> tuple[float, list[float], list[list[float]]]:
    """F = -sum log p + lam/2 ||theta||^2 with analytic gradient and Hessian.

    One selection level contributes log-sum-exp(theta_R) - theta_w:
    gradient p - e_w, Hessian diag(p) - p p^T (where p is the softmax
    over the remaining set R).
    """
    n = len(theta)
    f = 0.0
    g = [0.0] * n
    h = [[0.0] * n for _ in range(n)]

    for levels in levels_by_obs:
        for winner, remaining in levels:
            m = max(theta[i] for i in remaining)
            exps = [math.exp(theta[i] - m) for i in remaining]
            z = sum(exps)
            f += m + math.log(z) - theta[winner]
            probs = [e / z for e in exps]
            for pos, i in enumerate(remaining):
                p = probs[pos]
                g[i] += p
                h[i][i] += p * (1.0 - p)
            for a in range(len(remaining)):
                ia = remaining[a]
                pa = probs[a]
                for b in range(a):
                    ib = remaining[b]
                    val = h[ia][ib] - pa * probs[b]
                    h[ia][ib] = h[ib][ia] = val
            g[winner] -= 1.0

    ridge = 0.0
    for i, t in enumerate(theta):
        ridge += t * t
        g[i] += lam * t
        h[i][i] += lam
    f += 0.5 * lam * ridge
    return f, g, h


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (n <= 20)."""
    n = len(a)
    m = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = col
        best = abs(m[col][col])
        for r in range(col + 1, n):
            if abs(m[r][col]) > best:
                best = abs(m[r][col])
                pivot = r
        if best <= 1e-300:
            raise ArithmeticError("singular Newton matrix")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        pivot_val = m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / pivot_val
            if factor != 0.0:
                row_r, row_c = m[r], m[col]
                for c in range(col, n + 1):
                    row_r[c] -= factor * row_c[c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = m[i][n]
        row = m[i]
        for j in range(i + 1, n):
            s -= row[j] * x[j]
        x[i] = s / row[i]
    return x


def _project_sum_zero(v: list[float]) -> None:
    """Subtract the mean in place (projection onto the sum-zero hyperplane)."""
    mean = sum(v) / len(v)
    for i in range(len(v)):
        v[i] -= mean


def _projected_grad_norm(g: Sequence[float]) -> float:
    if len(g) <= 1:
        return 0.0
    mean = sum(g) / len(g)
    return max(abs(v - mean) for v in g)


# ---------------------------------------------------------------------------
# Public fitting entry point
# ---------------------------------------------------------------------------


def fit_plackett_luce(
    rankings: Iterable[Ranking | Sequence[int]],
    *,
    n_objects: int | None = None,
    lam: float = 1.0,
    max_iterations: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
) -> FitResult:
    """Fit the penalised Plackett-Luce model on ``sum(theta) = 0``.

    Parameters
    ----------
    rankings:
        Iterable of :class:`Ranking` or plain index sequences (participants
        default to all objects).
    n_objects:
        Number of objects, 1..20.  Inferred from indices when omitted.
    lam:
        Ridge weight ``lambda > 0`` (finite).
    max_iterations:
        Newton-step budget (non-negative int; bool rejected).  Exhausting it
        before convergence gives status :data:`MAX_ITERATIONS` -- distinct
        from :data:`LINE_SEARCH_FAILED`, which means no acceptable descent
        step could be found at the current point.
    tol:
        Convergence tolerance on the infinity norm of the gradient
        projected onto the sum-zero hyperplane (finite float, > 0).

    Raises
    ------
    TypeError
        Wrong argument types; bool rejected wherever int/float expected.
    ValueError
        NaN/Infinity numeric inputs, bad/duplicate indices, ranked object
        absent from participants, empty participant set, lam/tol out of
        range, or n_objects outside [1, 20].
    """
    lam_f = _as_finite_float(lam, "lam")
    if lam_f <= 0.0:
        raise ValueError(f"lam must be > 0, got {lam}")
    tol_f = _as_finite_float(tol, "tol")
    if tol_f <= 0.0:
        raise ValueError(f"tol must be > 0, got {tol}")
    max_it = _as_int(max_iterations, "max_iterations")
    if max_it < 0:
        raise ValueError(f"max_iterations must be >= 0, got {max_it}")

    n, prepared = _normalize_rankings(rankings, n_objects)
    levels_by_obs = [obs[2] for obs in prepared]

    theta = [0.0] * n
    status = CONVERGED
    iterations = 0

    for step in range(max_it + 1):
        f, g, h = _objective_gradient_hessian(theta, levels_by_obs, lam_f)
        grad_norm = _projected_grad_norm(g)
        if grad_norm <= tol_f:
            status = CONVERGED
            break
        if step == max_it:
            status = MAX_ITERATIONS
            break

        # Tangent Newton step: H is positive definite (lambda > 0), and the
        # minimiser over the affine hyperplane is reached by a projected
        # right-hand side.  The step itself is re-projected to kill drift.
        gp = g[:]
        _project_sum_zero(gp)
        try:
            direction = _solve_linear(h, [-v for v in gp])
        except ArithmeticError:
            status = LINE_SEARCH_FAILED
            break
        _project_sum_zero(direction)
        directional = sum(g[i] * direction[i] for i in range(n))
        if not directional < 0.0:
            status = LINE_SEARCH_FAILED
            break

        alpha = 1.0
        full_candidate = [theta[i] + direction[i] for i in range(n)]
        f_full = _objective(full_candidate, levels_by_obs, lam_f)
        accepted = False
        new_theta = theta
        for _ in range(_MAX_HALVINGS + 1):
            if alpha == 1.0:
                candidate, f_candidate = full_candidate, f_full
            else:
                candidate = [
                    theta[i] + alpha * direction[i] for i in range(n)
                ]
                f_candidate = _objective(candidate, levels_by_obs, lam_f)
            if f_candidate <= f + _C1 * alpha * directional:
                accepted = True
                new_theta = candidate
                break
            alpha *= _SHRINK
        if not accepted:
            status = LINE_SEARCH_FAILED
            break
        if f_candidate >= f:
            # No representable objective decrease: near the optimum the
            # expected decrease can sit below one ULP of f while a nonzero
            # gradient remains.  Trust the full Newton step as a root step
            # only while it keeps shrinking the projected gradient; once it
            # cannot, the search has genuinely stalled.
            if full_candidate == theta:
                status = LINE_SEARCH_FAILED
                break
            _, g_full, _ = _objective_gradient_hessian(
                full_candidate, levels_by_obs, lam_f
            )
            if _projected_grad_norm(g_full) >= grad_norm:
                status = LINE_SEARCH_FAILED
                break
            new_theta = full_candidate
        theta = new_theta
        iterations += 1

    f, g, _h = _objective_gradient_hessian(theta, levels_by_obs, lam_f)
    grad_norm = _projected_grad_norm(g)
    contributions = _contributions(theta, prepared)
    log_lik = sum(c.log_likelihood for c in contributions)
    penalty = 0.5 * lam_f * sum(t * t for t in theta)

    return FitResult(
        converged=(status == CONVERGED),
        status=status,
        iterations=iterations,
        n_objects=n,
        theta=tuple(theta),
        log_likelihood=log_lik,
        objective=f,
        penalty=penalty,
        grad_norm=grad_norm,
        contributions=tuple(contributions),
    )


def _contributions(
    theta: Sequence[float], prepared: list[PreparedObs]
) -> list[ObservationContribution]:
    n = len(theta)
    out: list[ObservationContribution] = []
    for j, (ranked, participants, levels) in enumerate(prepared):
        ll = 0.0
        grad = [0.0] * n
        for winner, remaining in levels:
            m = max(theta[i] for i in remaining)
            exps = [math.exp(theta[i] - m) for i in remaining]
            z = sum(exps)
            ll += theta[winner] - m - math.log(z)
            probs = [e / z for e in exps]
            for pos, i in enumerate(remaining):
                grad[i] -= probs[pos]
            grad[winner] += 1.0
        out.append(
            ObservationContribution(
                index=j,
                ranked=ranked,
                participants=participants,
                log_likelihood=ll,
                gradient=tuple(grad),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Standalone probability helpers
# ---------------------------------------------------------------------------


def _checked_theta(theta: Sequence[float]) -> tuple[float, ...]:
    if isinstance(theta, (str, bytes)) or not isinstance(theta, Sequence):
        raise TypeError("theta must be a sequence of finite floats")
    vals = tuple(
        _as_finite_float(v, f"theta[{i}]") for i, v in enumerate(theta)
    )
    if not 1 <= len(vals) <= MAX_OBJECTS:
        raise ValueError(
            f"len(theta) must be in [1, {MAX_OBJECTS}], got {len(vals)}"
        )
    return vals


def _checked_index_set(
    values, n: int, where: str, *, allow_none: bool, allow_empty: bool
) -> tuple[int, ...] | None:
    if values is None:
        if not allow_none:
            raise TypeError(f"{where} is required")
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{where} must be a sequence of ints")
    out: list[int] = []
    seen: set[int] = set()
    for v in values:
        idx = _checked_index(v, where, "object")
        if idx in seen:
            raise ValueError(f"duplicate {where.rstrip('s')} object {idx}")
        if not 0 <= idx < n:
            raise ValueError(f"{where} object {idx} out of range for {n} objects")
        seen.add(idx)
        out.append(idx)
    if not allow_empty and not out:
        raise ValueError(f"{where} must be non-empty")
    return tuple(out)


def _prepare_probability_call(
    ranked: Sequence[int],
    theta: Sequence[float],
    participants: Sequence[int] | None,
) -> tuple[tuple[float, ...], list[Level], tuple[int, ...], tuple[int, ...]]:
    theta_t = _checked_theta(theta)
    n = len(theta_t)
    ranked_t = _checked_index_set(ranked, n, "ranked", allow_none=False, allow_empty=True)
    parts_in = participants if participants is not None else range(n)
    parts = _checked_index_set(
        parts_in, n, "participants", allow_none=False, allow_empty=False
    )
    part_set = set(parts)
    for w in ranked_t:
        if w not in part_set:
            raise ValueError(f"ranked object {w} is not in participants")
    if len(ranked_t) > len(parts):
        raise ValueError(
            f"ranked {len(ranked_t)} objects but only {len(parts)} participants"
        )

    remaining = list(parts)
    levels: list[Level] = []
    for w in ranked_t:
        levels.append((w, tuple(remaining)))
        remaining.remove(w)
    return theta_t, levels, ranked_t, parts


def log_ranking_likelihood(
    ranked: Sequence[int],
    theta: Sequence[float],
    participants: Sequence[int] | None = None,
) -> float:
    """Log-probability of ``ranked`` (a top-k prefix) under Plackett-Luce.

    Unranked participants stay in every denominator; objects outside
    ``participants`` never enter a denominator.  Empty ``ranked`` returns 0.
    Finite ``theta`` required (NaN/Infinity rejected); the result is finite
    for finite inputs.
    """
    theta_t, levels, _r, _p = _prepare_probability_call(ranked, theta, participants)
    total = 0.0
    for winner, remaining in levels:
        m = max(theta_t[i] for i in remaining)
        z = 0.0
        for i in remaining:
            z += math.exp(theta_t[i] - m)
        total += theta_t[winner] - m - math.log(z)
    return total


# Alias kept for readability in callers that think in terms of probability.
log_ranking_probability = log_ranking_likelihood


def ranking_probability(
    ranked: Sequence[int],
    theta: Sequence[float],
    participants: Sequence[int] | None = None,
) -> float:
    """Probability of ``ranked``; may underflow to 0.0 for extreme long shots.

    Use :func:`log_ranking_likelihood` for log-space work.
    """
    return math.exp(log_ranking_likelihood(ranked, theta, participants))


def winner_probabilities(
    theta: Sequence[float],
    participants: Sequence[int] | None = None,
) -> tuple[float, ...]:
    """Softmax probability that each object wins one Plackett-Luce selection.

    Returned as a full-length vector over all modelled objects (0 for
    non-participants); entries sum to 1.
    """
    theta_t = _checked_theta(theta)
    n = len(theta_t)
    parts_in = participants if participants is not None else range(n)
    parts = _checked_index_set(
        parts_in, n, "participants", allow_none=False, allow_empty=False
    )
    m = max(theta_t[i] for i in parts)
    exps = {i: math.exp(theta_t[i] - m) for i in parts}
    z = sum(exps.values())
    out = [0.0] * n
    for i in parts:
        out[i] = exps[i] / z
    return tuple(out)
