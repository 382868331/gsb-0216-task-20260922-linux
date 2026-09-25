# Partial-Ranking Plackett–Luce Fitting

Reusable, dependency-free Python library for fitting **Plackett–Luce abilities
from partial top-k, tie-free rankings** with optional per-observation
participant sets, a sum-to-zero constraint, an L2 penalty, and optional
**anchored (fixed) abilities with box bounds** on the free ones. Standard
library only.

## Requirements

- **Python 3.14.7** (developed/tested on CPython 3.14.7, Linux). No third-party
  packages; no network, accounts, keys, database, middleware, or Docker.
- Tests use only `unittest`; the demo uses only the library plus `math`/`time`.

## Commands

```bash
python -m unittest discover -s tests -v   # full test suite (56 tests)
python demo.py                            # deterministic live demo (~0.1 s)
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

## Anchored fitting (fixed abilities)

`fit_plackett_luce(..., anchors=..., bounds=...)` lets the caller pin some
abilities at known constants and optimize only the rest:

- `anchors`: mapping `{object_index: ability}` (or an iterable of
  `(index, ability)` pairs). Values must be finite; indices must be valid
  `int` object ids. **Duplicate anchors, unknown objects, and non-finite
  values are rejected up front** with `PLInputError`.
- With **at least one anchor**, the `Σ theta = 0` gauge is **dropped
  entirely** (no re-centering of anything, anchored or free) and the L2
  penalty applies to the **free abilities only**:
  `f = -loglik + (lambda/2) * Σ_{i free} theta_i²`. With **no anchors** the
  legacy behavior is kept exactly: sum-to-zero gauge, penalty on all
  coordinates, reduced `(n-1)`-coordinate Newton.
- Anchored objects still enter **every denominator their participant set
  implies** (ranked or unranked); they shape the probabilities but are never
  part of a Newton update. Gradient and Hessian are taken over the free
  block only.
- If **all** objects are anchored, no optimization happens: the call simply
  evaluates the likelihood and the per-observation contributions at the
  fixed abilities (`iterations = 0`, penalty term exactly 0).
- `initial` in anchored mode must be finite length-`n` but does **not** need
  to sum to zero; anchored entries are overridden by the anchor constants.

### Box bounds on free abilities

- `bounds`: mapping `{object_index: (lo, hi)}` (or iterable of pairs) with
  **finite** `lo <= hi`, only for free (non-anchored) objects, and only in
  anchored mode. Bounds on anchored objects, `lo > hi`, non-finite
  endpoints, or bounds without any anchor are rejected. An `initial` point
  outside its bounds is rejected, never silently clipped.
- Bounds are enforced by a **projected active-set Newton method**: at each
  iteration, coordinates pinned at a bound whose gradient does not point
  back into the box are held out of the Newton system; the remaining free
  block takes a Newton step, each backtracked trial point is **clamped onto
  the box**, and an Armijo test on the actual projected step accepts or
  halves. This is *not* "fit unconstrained, then clip" — every iterate is
  feasible and the anchors never move during backtracking.
- The result reports `boundary_kkt_residual`: the infinity-norm KKT residual
  of the box constraints (`|g_i|` interior, `max(0, -g_i)` at the lower
  bound, `max(0, g_i)` at the upper bound). With bounds it is also the
  convergence measure (`<= gtol`). It is `None` for legacy (anchor-free)
  fits and for anchored fits without bounds.
- The anchored Armijo test allows a documented rounding slack of
  `1e-13 * max(1, |f|)`: once the achievable decrease falls below the
  resolution of the accumulated objective itself, a strict decrease test is
  undecidable in floating point. Genuine steps are unaffected.

## Public interface (`plackett_luce` package)

| Name | Purpose |
|---|---|
| `MAX_OBJECTS = 20` | Hard interface bound on `n_objects`. |
| `Observation(ranking=(), participants=None)` | One sample. `ranking`: distinct `int` indices, best first, length `k ≤ n`. `participants`: distinct indices, or `None` for all `n` objects. |
| `fit_plackett_luce(observations, n_objects, *, lam=1.0, initial=None, max_iter=100, gtol=1e-10, anchors=None, bounds=None)` | Penalized MLE by backtracking Newton; optional anchored/bounded fitting (see above). |
| `ranking_log_probability(ranking, theta, participants=None)` | Stable log probability of one partial ranking. |
| `ranking_probability(...)` | `exp` of the above (may underflow for tiny probabilities). |
| `penalized_objective(observations, theta, lam)` | Scalar objective at a sum-zero point. |
| `penalized_gradient_free(...)` | Analytic gradient in the `n-1` free coordinates. |
| `penalized_hessian_free(...)` | Analytic Hessian in the free coordinates. |
| `PLInputError(ValueError)` | All input/contract violations; message names the field and index. |
| `ConvergenceStatus` | `CONVERGED`, `MAX_ITERATIONS`, `LINE_SEARCH_FAILED`, `NUMERICAL_ERROR`. |
| `PLResult` | `status, converged, iterations, theta, log_likelihood, penalized_objective, log_contributions, gradient_norm, message, boundary_kkt_residual` (last field set only for bounded anchored fits). |

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
| Anchored two-object extremum | Independent bisection root of `λt = m·(1 − σ(t − a))` | `1e-10` absolute |
| Anchored free gradient at optimum | Central FD of the independent anchored objective | `< 1e-5` absolute |
| Anchored per-observation contributions | Independent reference logsumexp | `1e-12` absolute |
| Boundary KKT residual at an active bound | Reported by solver | `≤ 1e-10` |

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
- **Anchors drop the gauge rather than re-centering.** Fixing some abilities
  gives the remaining ones an absolute scale, so the sum-to-zero constraint
  would be wrong (it would move the anchors). The penalty shrinks free
  abilities toward 0, keeping the anchored problem strictly convex.

## Known limitations

- Implements exactly the contracted subset: tie-free top-k Plackett–Luce,
  one global L2 penalty, sum-to-zero gauge. No ties/partial orders with
  equivalence classes, no object-specific or grouped penalties, no
  hierarchical/covariate (Bradley–Terry–Luce with features) models, no
  Bayesian posterior or standard-error output, no claim of standards
  compliance beyond this contract.
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
tests/test_anchored.py         # anchors, bounds/KKT, all-fixed evaluation
demo.py                       # deterministic synthetic-data demonstration
```
