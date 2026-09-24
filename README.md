# Partial-Ranking Plackett–Luce Fitting

Reusable, dependency-free Python library for fitting **Plackett–Luce abilities
from partial top-k, tie-free rankings** with optional per-observation
participant sets, a sum-to-zero constraint, and an L2 penalty, **plus exact
likelihoods for precedence-constrained orders** (a fixed candidate set of at
most 6 with pairwise "a before b" requirements; the probability is summed
over every legal linear extension). Standard library only.

## Requirements

- **Python 3.14.7** (developed/tested on CPython 3.14.7, Linux). No third-party
  packages; no network, accounts, keys, database, middleware, or Docker.
- Tests use only `unittest`; the demo uses only the library plus `math`/`time`.

## Commands

```bash
python -m unittest discover -s tests -v   # full test suite (80 tests)
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

## Precedence-constrained orders

Given a **fixed candidate set** `C` with `1 ≤ |C| ≤ 6` and a collection of
strict pairwise precedence edges `(a, b)` — "a must finish ahead of b" — the
library answers the likelihood of observing an ordering consistent with all
edges:

```
L(constraints | theta) = Σ_{π ∈ LE} P(π is the full order over C | theta)
```

- `LE` is the set of **linear extensions** of the precedence poset, i.e.
  permutations of the *whole* candidate set satisfying every edge (transitivity
  is implied). The extensions are enumerated exactly by Kahn-style
  backtracking over the transitive closure.
- Candidates that the constraints leave unordered are **alternatives, not
  ties**: every distinct completion is a separate summand. There is no
  tie/equivalence-class probability model in this iteration.
- Each summand is the **full-order** Plackett–Luce probability over exactly
  `C` (`|C|` stages, every not-yet-picked candidate in the denominator);
  objects outside `C` never enter any denominator.
- The probability sum is accumulated in log space with a max-shifted
  logsumexp, and both `likelihood` and `log_likelihood` are returned along
  with `n_extensions`.
- The candidate bound is 6 because enumeration is exact and exhaustive: the
  worst case (no constraints) is `6! = 720` full orders.

### Worked checks (all reproduced by the tests)

- **No constraints** over `m` candidates: all `m!` full orders partition the
  sample space, so `n_extensions = m!` and `L = 1` for every `theta`.
- **Total chain** (e.g. `0<1<2<3`): one extension; `L` equals the single
  full-order probability from `ranking_probability`.
- **One edge** on three equally able candidates: 3 of 6 equiprobable orders
  qualify, `L = 1/2`. With two equally able candidates,
  `P(a before b) = sigmoid(theta_a − theta_b)`; the two directional events sum
  to 1 (the pairwise Luce identity).
- **Contradictory edges** that close a directed cycle (including a self-loop)
  are rejected with `PLInputError` before any probability is computed; the
  error names the cycle. Such requirements have zero legal extensions.

### Constrained-order input semantics

- `candidates`: distinct `int` indices in `[0, n)`, non-empty, length ≤ 6
  (`n = len(theta)`). Strings, floats, and `bool` indices are rejected;
  duplicates and an empty set are rejected.
- `constraints`: an iterable of `OrderConstraint` (or `None`/empty). Both
  endpoints must be ints in `[0, n)` **and** members of `candidates`;
  `before == after` is rejected; a directed cycle is rejected with the cycle
  spelled out. Generators are accepted (consumed once).
- `theta` follows the same contract as everywhere else: finite length-`n`
  abilities with `sum(theta) = 0`.
- `return_extensions` must be a real `bool`; when true the result also
  carries every legal extension as best-first tuples (default off; ≤ 720).
- The function is purely functional: inputs are never modified, and all
  validation (theta, candidates, edges, acyclicity) finishes before any
  enumeration or arithmetic, so a rejected call leaves no partial state.

## Public interface (`plackett_luce` package)

| Name | Purpose |
|---|---|
| `MAX_OBJECTS = 20` | Hard interface bound on `n_objects`. |
| `Observation(ranking=(), participants=None)` | One sample. `ranking`: distinct `int` indices, best first, length `k ≤ n`. `participants`: distinct indices, or `None` for all `n` objects. |
| `fit_plackett_luce(observations, n_objects, *, lam=1.0, initial=None, max_iter=100, gtol=1e-10)` | Penalized MLE by backtracking Newton. |
| `ranking_log_probability(ranking, theta, participants=None)` | Stable log probability of one partial ranking. |
| `ranking_probability(...)` | `exp` of the above (may underflow for tiny probabilities). |
| `penalized_objective(observations, theta, lam)` | Scalar objective at a sum-zero point. |
| `penalized_gradient_free(...)` | Analytic gradient in the `n-1` free coordinates. |
| `penalized_hessian_free(...)` | Analytic Hessian in the free coordinates. |
| `PLInputError(ValueError)` | All input/contract violations; message names the field and index. |
| `ConvergenceStatus` | `CONVERGED`, `MAX_ITERATIONS`, `LINE_SEARCH_FAILED`, `NUMERICAL_ERROR`. |
| `PLResult` | `status, converged, iterations, theta, log_likelihood, penalized_objective, log_contributions, gradient_norm, message`. |
| `MAX_CONSTRAINED_CANDIDATES = 6` | Hard bound on the candidate set of a constrained-order query. |
| `OrderConstraint(before, after)` | One strict precedence edge: `before` must rank ahead of `after` (transitivity implied). |
| `constrained_order_probability(constraints, candidates, theta, *, return_extensions=False)` | Validate acyclicity, then sum full-order PL probabilities over all legal linear extensions; returns `ConstrainedOrderResult`. |
| `ConstrainedOrderResult` | `candidates, n_extensions, log_likelihood, likelihood, extensions`. |

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

Constrained-order queries (`constrained_order_probability`) follow the same
rules: candidate/edge indices reject `bool`, floats, strings, out-of-range
and non-member values; the candidate set rejects duplicates, emptiness, and
sizes above 6. All validation precedes enumeration.

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
| Constrained-order extension counts | Independent `itertools.permutations` filter (`tests/reference.py`) | exact integer equality |
| Constrained-order summed likelihood | Independent permutation enumeration + independently coded staged PL probability; hand-computed rationals (`1/m!`, logistic pairs) | `1e-11` absolute; `1e-14` on hand-computed cases |

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
- **Exact exhaustive enumeration for constrained orders.** The candidate set
  is capped at 6 (`≤ 720` full orders), so the result is the exact likelihood
  with a definitive legal-extension count; no approximation, sampling, or
  dynamic-programming partial-order counting is needed in this range.

## Known limitations

- Implements exactly the contracted subset: tie-free top-k Plackett–Luce,
  one global L2 penalty, sum-to-zero gauge, and exact constrained-order
  likelihoods for at most 6 candidates. No ties/partial orders with
  equivalence classes (unordered candidates are summed as distinct
  completions, never modeled as ties), no object-specific or grouped
  penalties, no hierarchical/covariate (Bradley–Terry–Luce with features)
  models, no Bayesian posterior or standard-error output, no claim of
  standards compliance beyond this contract.
- Constrained-order queries are an **exact** method: more than 6 candidates
  are rejected rather than approximated. There is no large-scale approximate
  inference in this iteration (none was required).
- `ranking_probability` exponentiates a log probability and can underflow to
  `0.0`; use `ranking_log_probability` for very small probabilities. The
  constrained-order sum itself is accumulated in log space (stable), but the
  returned `likelihood` field still exponentiates the final log value.
- Inputs must be materializable in memory; the observation iterable is
  validated before fitting (generators are consumed once).
- Extremely large fitted `|theta|` (hundreds) are representable but not
  stress-tested; the L2 penalty keeps ordinary datasets far from that regime.

## Repository layout

```
plackett_luce/__init__.py            # public API re-exports
plackett_luce/model.py               # validation, likelihood, Newton solver,
                                     # constrained-order enumeration
tests/reference.py                   # INDEPENDENT reference (no core import)
tests/test_validation.py             # contract / error-semantics tests
tests/test_math.py                   # analytic extremum, FD derivatives, fit quality
tests/test_constraints_validation.py # constrained-order contract / errors
tests/test_constraints_math.py       # constrained-order counts/likelihoods vs
                                     # independent permutation enumeration
demo.py                              # deterministic synthetic-data demonstration
```
