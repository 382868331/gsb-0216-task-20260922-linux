# Partial-Ranking Plackett–Luce Fitting

Reusable, dependency-free Python library for fitting **Plackett–Luce abilities
from partial top-k, tie-free rankings** with optional per-observation
participant sets, a sum-to-zero constraint, and an L2 penalty. It also supports
**anchored fitting**: the caller fixes some objects' abilities at constants and
optimizes the rest, optionally under finite box bounds. Standard library only.

## Requirements

- **Python 3.14.7** (developed/tested on CPython 3.14.7, Linux). No third-party
  packages; no network, accounts, keys, database, middleware, or Docker.
- Tests use only `unittest`; the demo uses only the library plus `math`/`time`.

## Commands

```bash
./.venv/bin/python -m unittest discover -s tests -v   # full suite (54 tests)
./.venv/bin/python demo.py                # deterministic live demo (~0.1 s)
```

Both run fully offline from the repository root.

## Model

Objects `0, …, n-1` have log-strength parameters `theta`. For one observation
with ordered ranking `(r_1, …, r_k)` (best first) and participant set `A`
(`A` contains every ranked object):

```
P(ranking | theta) = Π_{j=1..k} exp(theta_{r_j}) /
                       Σ_{i ∈ A \\ {r_1,…,r_{j-1}}} exp(theta_i)
```

- This is a **partial ranking of length k**, not a complete ranking: only `k`
  Plackett stages contribute.
- **Unranked participants stay in every remaining denominator** (the residual
  choice is left unmodeled, exactly as top-k likelihood requires).
- **Non-participants never enter any denominator** (e.g. a race an object did
  not enter).
- `k = 0` (empty ranking) contributes `log P = 0` (`P = 1`).
- Ties are not part of this contract; repeated indices are rejected.

The estimator minimizes, over the constraint hyperplane `Σ_i theta_i = 0`,

```
f(theta) = -Σ_obs log P(obs | theta) + (lambda/2) * Σ_i theta_i²,   lambda > 0
```

The penalty plus the sum-to-zero gauge makes `f` strictly convex and coercive
on the free `n-1` coordinates, so a **unique minimizer always exists**; there
is no infeasibility regime to report for valid data (separation cannot diverge
the estimate because of the penalty).

## Anchored fitting

Call `fit_plackett_luce(obs, n, fixed_theta=..., bounds=...)` or the
explicit entry `fit_anchored(obs, n, fixed_theta, ...)`. `fixed_theta` is a
mapping `{object_index: ability}` (or an iterable of `(index, ability)`
pairs) naming the anchored objects:

- With **at least one anchor**, the gauge is fixed by the anchors: the
  `Σ theta = 0` constraint is **not** imposed, anchored abilities are **never
  recentered**, and the ability sum is generally nonzero.
- The penalty is `(lambda/2) Σ_{i free} theta_i²` — **free coordinates
  only**. Fixed abilities affect every stage probability (including
  top-k denominators and unranked participants) but receive no derivative or
  penalty.
- Gradient and Hessian are the **free-coordinate block only**; fixed
  coordinates are never Newton-updated.
- If **all objects are fixed**, there are no decision variables: the call
  returns `CONVERGED` with 0 iterations and only the likelihood and
  per-observation `log_contributions` are evaluated.
- With **no anchors** (`fixed_theta` omitted/empty/`None`) the legacy
  sum-zero calibration is used unchanged; `bounds` is then rejected.

### Finite box bounds on free objects

`bounds` is a mapping `{object_index: (lower, upper)}` or a length-`n`
sequence whose entries are `(lower, upper)` pairs or `None`. Bounds may be
given for **free objects only**; both endpoints must be finite and
`lower <= upper`.

The optimizer is a genuinely **constrained active-set Newton** method:

1. each iteration classifies free coordinates as pinned at a face only when
   the face is KKT-correct (lower with `g ≥ 0`, upper with `g ≤ 0` for the
   minimizer);
2. only the **inactive free block** of the Hessian is factored (Cholesky,
   with a relative+absolute diagonal jitter fallback);
3. the step is clipped to the feasible box (`alpha_max` is the distance to
   the nearest blocking face), faces with the wrong KKT sign are released
   with a feasible projected-gradient direction, and Armijo backtracking
   (`c1 = 1e-4`, halving) runs on feasible points only;
4. this is **not** an unconstrained fit followed by truncation.

`PLResult.boundary_kkt_residual` reports the infinity norm of the box-KKT
stationarity residual over free coordinates: the raw gradient at interior
points, `max(0, -g)` at a lower bound, and `max(0, g)` at an upper bound. It
is `0.0` on the sum-zero path and for all-fixed evaluations. `gradient_norm`
is the same projected-gradient infinity norm in anchored mode.

Anchor validation rejects up front: duplicate anchors, unknown/non-int/bool
object indices, and `NaN`/infinite abilities (or bound endpoints).

## Public interface (`plackett_luce` package)

| Name | Purpose |
|---|---|
| `MAX_OBJECTS = 20` | Hard interface bound on `n_objects`. |
| `Observation(ranking=(), participants=None)` | One sample. `ranking`: distinct `int` indices, best first, length `k ≤ n`. `participants`: distinct indices, or `None` for all `n` objects. |
| `fit_plackett_luce(observations, n_objects, *, lam=1.0, initial=None, max_iter=100, gtol=1e-10)` | Penalized MLE by backtracking Newton. |
| `fit_plackett_luce(..., fixed_theta=None, bounds=None)` | Same call with optional anchors and free box bounds; empty/missing anchors keep the sum-zero path. |
| `fit_anchored(observations, n_objects, fixed_theta, *, lam=1.0, bounds=None, initial=None, max_iter=100, gtol=1e-10)` | Anchored entry requiring at least one fixed ability; all-fixed is a pure likelihood evaluation. |
| `Bound(lower, upper)` | One finite box bound on a free coordinate (also accepted as plain `(lower, upper)` pairs). |
| `anchored_penalized_objective(observations, theta, lam, fixed_theta)` | Anchored objective with the penalty on free coordinates only. |
| `anchored_gradient_free(...)` | `(free_indices, gradient)` over ascending free object indices. |
| `anchored_hessian_free(...)` | `(free_indices, free_block_hessian)`. |
| `ranking_log_probability(ranking, theta, participants=None)` | Stable log probability of one partial ranking. |
| `ranking_probability(...)` | `exp` of the above (may underflow for tiny probabilities). |
| `penalized_objective(observations, theta, lam)` | Scalar objective at a sum-zero point. |
| `penalized_gradient_free(...)` | Analytic gradient in the `n-1` free coordinates. |
| `penalized_hessian_free(...)` | Analytic Hessian in the free coordinates. |
| `PLInputError(ValueError)` | All input/contract violations; message names the field and index. |
| `ConvergenceStatus` | `CONVERGED`, `MAX_ITERATIONS`, `LINE_SEARCH_FAILED`, `NUMERICAL_ERROR`. |
| `PLResult` | `status, converged, iterations, theta, log_likelihood, penalized_objective, log_contributions, gradient_norm, message, boundary_kkt_residual` (last defaults to `0.0`; older call sites are unaffected). |

`log_contributions[t]` is the per-observation log likelihood `log P(obs_t |
theta*)`; their exact-predicate sum (`math.fsum`) equals `log_likelihood`.

`n_objects` accepts `1 … 20`. With `n = 1` the constraint pins `theta = 0` and
the result is the trivial optimum (0 iterations).

### Parameterization

Free coordinates `x ∈ ℝ^{n-1}`: `theta_i = x_i` for `i < n-1` and
`theta_{n-1} = -Σ_i x_i`, which enforces `Σ theta = 0` exactly (up to one
floating-point sum). `initial` may be supplied in ambient length-`n`
coordinates and must itself satisfy the constraint (see tolerance below); the
optimizer then maps to free coordinates internally.

### Solver

- Each denominator is evaluated as
  `max + log Σ exp(theta_i - max)` (logsumexp), so scores of magnitude up to
  about ±700 cannot overflow.
- Analytic gradient and Hessian: per stage, with softmax probabilities
  `p` over the remaining set, the negative-log-likelihood contributes
  `p − e_winner` to the gradient and `diag(p) − p pᵀ` to the Hessian; the
  penalty contributes `lambda·theta` / `lambda·I`. Ambient derivatives are
  reduced to the free coordinates with the constraint Jacobian, and the
  `(n-1)×(n-1)` system is solved by a hand-written Cholesky factorization.
- Full Newton step (`alpha = 1`) with Armijo backtracking
  (`c1 = 1e-4`, halving, up to 80 trials per iteration).
- Convergence: infinity norm of the **reduced gradient** `≤ gtol`.
- If a Cholesky factor is not positive definite due to rounding, a small
  diagonal jitter (`1e-12 … 1e-6` times the largest diagonal) is retried;
  analytically the matrix is always PD for `lambda > 0`.

### Convergence vs. failure (kept distinct)

- `CONVERGED` — stationarity tolerance reached.
- `MAX_ITERATIONS` — the **step budget was exhausted** while the known unique
  minimizer was not yet reached. This is *not* an infeasibility proof: the
  minimizer provably exists for every valid batch (`lambda > 0`). Raising
  `max_iter` or relaxing `gtol` changes this outcome.
- `LINE_SEARCH_FAILED` — no Armijo-acceptable step in 80 halvings.
- `NUMERICAL_ERROR` — non-finite arithmetic, a non-descent Newton direction, or
  an unfactorable Newton matrix even with jitter.

Invalid or structurally impossible data is rejected **before any iteration**
with `PLInputError` (there is no separate "proved infeasible" solver state
because none exists for this problem).

## Input and error semantics

All validation runs over the **whole batch before any optimization or
mutation** (the API is purely functional anyway — inputs are never modified), so
a rejected call cannot leave a partial change.

- Floating-point inputs (`lam`, `gtol`, `initial`/`theta` entries) reject
  `NaN`, `+Inf`, `-Inf`.
- `bool` is **never accepted as an integer** (`n_objects`, `max_iter`, index
  values, `lam`, …): Python `True`/`False` raise `PLInputError`.
- `lam > 0`, `max_iter ≥ 1`, `gtol > 0`, `1 ≤ n_objects ≤ 20`.
- Indices must be `int` in `[0, n_objects)`; strings and floats are rejected.
- Ranking indices and participant indices must each be **distinct**. Repeats
  name the offending observation, field, position, and index value.
- Every ranked index must belong to that observation's participant set.
- `theta` (public probability/objective functions and `initial`) must be
  finite, length `n`, and satisfy `|Σ theta_i| ≤ 1e-10 · max(1, max|theta_i|)`.
  The constraint is reported, never silently projected away. Fitted `theta`
  returned by the solver always sums to zero (its free-coordinate construction
  is exact apart from one rounded addition).
- `observations` may be any finite iterable (lists, tuples, generators); each
  element must be an `Observation`.
- In anchored mode `initial` needs **no** sum-zero condition; entries at
  fixed slots must equal the anchor exactly (fixed abilities are not
  recentered), and free entries must lie within their bounds. Anchors and
  bound endpoints reject `NaN`/infinities, duplicate or unknown indices.

## Accuracy and error accounting

All reported tolerances are on absolute or scaled quantities; tests do not use
loose tolerances to mask boundary errors.

| Quantity checked | Reference method | Tolerance |
|---|---|---|
| Per-stage log likelihood | Independent logsumexp implementation (`tests/reference.py`) | `1e-12` absolute |
| Analytic free gradient | Independent central finite differences, `h = 1e-6` | `1e-6 × max(1,‖ref‖∞)` |
| Analytic free Hessian | Independent central cross differences, `h = 1e-4` | `1e-4 × max(1,‖ref‖∞)` |
| Two-object extremum | Independent bisection root of `λa = m/(1+e^{2a})` | `1e-10` absolute |
| Fitted vs. true abilities (600 deterministic samples) | Statistical recovery check, gauge-aligned | spread `≤ 0.25` |
| Sum-to-zero of fitted theta | Direct | `≤ 1e-12` absolute |
| `Σ contributions = log_likelihood` | `math.fsum` | `1e-10` absolute |
| Anchored free gradient | Independent central differences, `h = 1e-6` | `1e-6 × max(1,‖ref‖∞)` |
| Anchored free Hessian block | Independent central cross differences, `h = 1e-4` | `1e-4 × max(1,‖ref‖∞)` |
| Anchored two-object optimum | Independent bisection of the scalar stationarity equation | `1e-9` absolute |
| Active bound solution | Analytic KKT sign + reported `boundary_kkt_residual` | `≤ 1e-10` |

Expected values in tests come from closed-form expressions, bisection,
explicit fraction products, or finite differences of the **independent**
reference in `tests/reference.py` — that module never imports the package under
test. Core-vs-core round-trip comparisons are used only as supplementary
consistency checks.

The solver's default stationarity tolerance is `gtol = 1e-10` on the reduced
gradient infinity norm; output abilities are typically accurate to ~10 digits
near the optimum on the tested cases.

## Design trade-offs

- **Pure Python, no NumPy.** Matrix size is at most `19×19` (n ≤ 20), so
  explicit loops and `math.fsum` are fast enough (400 observations × 5 objects
  fit in ~5 Newton steps, milliseconds) while keeping zero dependencies.
- **Penalized estimator by contract.** `λ > 0` removes the non-existence of
  the unpenalized MLE under object separation, which makes
  `MAX_ITERATIONS` genuinely different from "no solution".
- **Logsumexp with per-stage max subtraction**, plus `math.fsum` for totals,
  favors accuracy and overflow safety over micro-optimization.
- **Participant sets per observation** rather than a global eligibility mask,
  since disjoint/irregular event lineups are the target use case.

## Known limitations

- Implements exactly the contracted subset: tie-free top-k Plackett–Luce
  (staged likelihood only — no partial-order/equivalence-class likelihood),
  one global L2 penalty, a sum-to-zero gauge or caller anchors, and finite
  per-free-object box bounds. No ties, no object-specific or grouped
  penalties, no hierarchical/covariate (Bradley–Terry–Luce with features)
  models, no Bayesian posterior or standard-error output, no claim of
  standards compliance beyond this contract.
- In anchored mode the gauge is whatever constants the caller supplies; the
  library does not second-guess anchor scale. Near the floating-point
  minimizer an unconstrained Newton step may be below `~1e-14` relative to
  the parameter scale; the solver then reports `CONVERGED` with the residual
  it reached (the `gradient_norm` message states the quantity).
- `ranking_probability` exponentiates a log probability and can underflow to
  `0.0`; use `ranking_log_probability` for very small probabilities.
- Inputs must be materializable in memory; the observation iterable is
  validated before fitting (generators are consumed once).
- Extremely large fitted `|theta|` (hundreds) are representable but not
  stress-tested; the L2 penalty keeps ordinary datasets far from that regime.

## Repository layout

```
plackett_luce/__init__.py     # public API re-exports
plackett_luce/model.py        # validation, likelihood, Newton solver
tests/reference.py            # INDEPENDENT reference (no core import)
tests/test_validation.py      # contract / error-semantics tests
tests/test_math.py            # analytic extremum, FD derivatives, fit quality
tests/test_anchored.py        # anchors, unobserved free object, bounds/KKT
demo.py                       # deterministic synthetic-data demonstration
```
