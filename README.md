# Partial-Ranking Plackett–Luce Fitting

Reusable, dependency-free Python library for fitting **Plackett–Luce abilities
from partial top-k, tie-free rankings** with optional per-observation
participant sets, a sum-to-zero constraint, and an L2 penalty, plus **exact
likelihoods of precedence constraints (partial orders) over a fixed candidate
set** by summation over all legal linear extensions. Standard library only.

## Requirements

- **Python 3.14.7** (developed/tested on CPython 3.14.7, Linux). No third-party
  packages; no network, accounts, keys, database, middleware, or Docker.
- Tests use only `unittest`; the demo uses only the library plus `math`/`time`.

## Commands

```bash
python -m unittest discover -s tests -v   # full test suite (62 tests)
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

### Precedence constraints over a fixed candidate set (exact)

Given a fixed candidate set `A` with `1 ≤ |A| ≤ 6` and strict precedence pairs
`(before, after)` ("`before` must rank ahead of `after`"):

1. The pairs are interpreted as a directed graph and **checked to be acyclic
   first** (Kahn's algorithm). A self-loop or any cycle is a contradiction and
   raises `PLInputError` before enumeration.
2. The likelihood is the **exact sum over every legal linear extension** (a
   full permutation of `A` satisfying all pairs) of the staged ranking
   probability defined above:

```
P(constraints | theta, A) = Σ_{π ∈ LE(A, ≺)} Π_{j=1..|A|}
                              exp(theta_{π_j}) / Σ_{i ∈ A \\ {π_1,…,π_{j-1}}} exp(theta_i)
```

- **Incomparable objects are never treated as tied**: each linear extension is
  a distinct permutation and contributes its own probability.
- Objects outside the candidate set behave exactly like non-participants in the
  ranking model — they never enter any denominator.
- Empty constraints admit all `|A|!` extensions, whose probabilities partition
  the event space, so the likelihood is exactly `1`.
- A total order (chain) has exactly one extension and equals the probability of
  that one full ranking.
- The sum is accumulated with logsumexp, and both the stable log likelihood,
  the probability, the **number of legal extensions**, the materialized
  candidate set, and the deduplicated constraint pairs are returned.
- Enumeration is deterministic smallest-available-first topological
  backtracking; the 6-candidate bound caps the work at `6! = 720` full
  rankings, which is why the capability is exact rather than approximate.

## Public interface (`plackett_luce` package)

| Name | Purpose |
|---|---|
| `MAX_OBJECTS = 20` | Hard interface bound on `n_objects`. |
| `MAX_CANDIDATES = 6` | Hard bound on the candidate-set size for exact poset likelihood. |
| `Observation(ranking=(), participants=None)` | One sample. `ranking`: distinct `int` indices, best first, length `k ≤ n`. `participants`: distinct indices, or `None` for all `n` objects. |
| `OrderingConstraint(before, after)` | One strict precedence pair (frozen dataclass). |
| `ConstrainedLikelihood` | `probability, log_probability, n_extensions, candidate_set, constraints`. |
| `fit_plackett_luce(observations, n_objects, *, lam=1.0, initial=None, max_iter=100, gtol=1e-10)` | Penalized MLE by backtracking Newton. |
| `ranking_log_probability(ranking, theta, participants=None)` | Stable log probability of one partial ranking. |
| `ranking_probability(...)` | `exp` of the above (may underflow for tiny probabilities). |
| `constrained_ranking_probability(constraints, theta, candidate_set=None)` | Acyclicity check then exact sum over all legal linear extensions. |
| `penalized_objective(observations, theta, lam)` | Scalar objective at a sum-zero point. |
| `penalized_gradient_free(...)` | Analytic gradient in the `n-1` free coordinates. |
| `penalized_hessian_free(...)` | Analytic Hessian in the free coordinates. |
| `PLInputError(ValueError)` | All input/contract violations; message names the field and index. |
| `ConvergenceStatus` | `CONVERGED`, `MAX_ITERATIONS`, `LINE_SEARCH_FAILED`, `NUMERICAL_ERROR`. |
| `PLResult` | `status, converged, iterations, theta, log_likelihood, penalized_objective, log_contributions, gradient_norm, message`. |

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

Precedence-constraint inputs follow the same rules and are all validated
(candidates, endpoint indices, types, acyclicity) **before any linear
extension is generated**, so a rejected call enumerates nothing and leaves no
partial state (the API is purely functional):

- `constraints` may be any finite iterable of `OrderingConstraint`; a bare
  tuple `(0, 1)` is not accepted in place of one.
- `candidate_set` must be distinct `int` indices, non-empty, and of size at
  most `MAX_CANDIDATES` (6); `None` means all `n` objects (which then requires
  `n ≤ 6` for this function). Both endpoints of every constraint must belong
  to the candidate set. `bool`/float/string endpoints are rejected.
- A self-loop or any directed cycle raises `PLInputError` whose message names
  the objects still on the cycle. Duplicate or transitively redundant pairs
  are accepted harmlessly and materialized once.
- The shared `theta` rules apply unchanged (finite, length `n`, sum zero).

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
| Poset likelihood & extension count (n ≤ 6) | Independent `itertools.permutations` brute force filtered by the pairs, scored by `tests/reference.py`; hand cases: `P(0≻1)=σ(θ₀−θ₁)` (with `θ=(a,−a)` this is `σ(2a)`), counts `n!`, `4!/(2!2!)`, uniform `P = #ext/n!` | `1e-12` absolute; counts exact |

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
- **Exact linear-extension enumeration for precedence likelihood.** With at
  most 6 candidates (`≤ 720` orders) deterministic topological backtracking is
  trivially fast and gives the exact probability and extension count, so no
  sampling/approximation error budget is needed; larger sets are rejected at
  the documented interface bound rather than silently approximated.
- The poset path **reuses the existing staged ranking probability**
  (`_staged_log_probability`), so the new capability sums the same
  already-tested per-ranking likelihood instead of carrying a second formula.

## Known limitations

- Implements exactly the contracted subset: tie-free top-k Plackett–Luce,
  one global L2 penalty, sum-to-zero gauge, and exact poset likelihoods over
  candidate sets of at most 6. No ties/partial orders with equivalence
  classes, no tie probability model, no object-specific or grouped penalties,
  no hierarchical/covariate (Bradley–Terry–Luce with features) models, no
  Bayesian posterior or standard-error output, no claim of standards
  compliance beyond this contract.
- Precedence likelihood is **exact exhaustive enumeration** and is bounded to
  6 candidates (`≤ 720` extensions); there is intentionally no
  large-scale approximation, sampling estimator, or dynamic-programming path
  for bigger candidate sets — oversize sets are rejected, not approximated.
- `ranking_probability` and `ConstrainedLikelihood.probability` exponentiate
  log values and can underflow to `0.0`; use the log forms for very small
  probabilities.
- Inputs must be materializable in memory; the observation/constraint
  iterables are validated before computation (generators are consumed once).
- Extremely large fitted `|theta|` (hundreds) are representable but not
  stress-tested; the L2 penalty keeps ordinary datasets far from that regime.

## Repository layout

```
plackett_luce/__init__.py     # public API re-exports
plackett_luce/model.py        # validation, likelihood, Newton solver,
                              #   acyclicity check + linear-extension sum
tests/reference.py            # INDEPENDENT reference (no core import)
tests/test_validation.py      # contract / error-semantics tests
tests/test_math.py            # analytic extremum, FD derivatives, fit quality
tests/test_constraints.py     # poset likelihood: hand cases, brute force, errors
demo.py                       # deterministic synthetic-data demonstration
```
