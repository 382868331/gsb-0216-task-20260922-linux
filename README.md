# Partial-Ranking Plackett–Luce Fitting

Reusable, dependency-free Python library for fitting **Plackett–Luce abilities
from partial top-k, tie-free rankings** with optional per-observation
participant sets, a sum-to-zero constraint, and an L2 penalty. Standard library
only.

**Version 2** of this library additionally supports **partial rankings whose
rank positions are either one object or one two-way tie**, modeled as a single
joint choice (see “Two-way-tied model” below). The original tie-free
Plackett–Luce entry point is unchanged; the tied extension is a deliberately
scoped model, **not** a claimed unified standard for all real tied rankings.

## Requirements

- **Python 3.14.7** (developed/tested on CPython 3.14.7, Linux). No third-party
  packages; no network, accounts, keys, database, middleware, or Docker.
- Tests use only `unittest`; the demo uses only the library plus `math`/`time`.

## Environment setup (venv)

The code uses the standard library only; the venv just pins the interpreter.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python --version          # must show Python 3.14.7
```

## Commands

```bash
python -m unittest discover -s tests -v   # full suite: original + tied tests (63 tests)
python demo.py                            # tie-free live demo (~0.1 s)
python demo_iteration.py                  # two-way-tied live demo (< 1 s, budget 90 s)
```

Both run fully offline from the repository root.
Both demos compute everything live: no `sleep`, no pre-recorded output.

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

Expected values in tests come from closed-form expressions, bisection,
explicit fraction products, or finite differences of the **independent**
reference in `tests/reference.py` — that module never imports the package under
test. Core-vs-core round-trip comparisons are used only as supplementary
consistency checks.

The solver's default stationarity tolerance is `gtol = 1e-10` on the reduced
gradient infinity norm; output abilities are typically accurate to ~10 digits
near the optimum on the tested cases.

## Two-way-tied model (`plackett_luce.ties`)

At one stage with remaining object set `R`, every singleton and every
unordered pair is offered in **one common, jointly normalized choice**:

```
w({i})     = exp(theta_i)
w({i, j})  = tau * exp((theta_i + theta_j) / 2),   i < j
P(outcome) = w(outcome) / ( Σ_{i ∈ R} exp(theta_i)
                            + Σ_{i < j, i,j ∈ R} tau·exp((theta_i + theta_j)/2) )
```

- `tau > 0` is a **fixed, user-given** positive number; it is never estimated.
- A tie `{i, j}` is **one joint outcome**, never the average (or sum) of the
  two ordered finishes `i` before `j` and `j` before `i`; the pair appears
  exactly once in the denominator.
- After a stage, **all chosen objects are removed**: one object for a
  singleton, both for a pair. Unchosen participants remain in every later
  denominator, exactly as for tie-free partial rankings.
- A partial ranking is an ordered sequence of groups, each of size one or
  two; e.g. `((0, 1), (2,))` means “0 and 1 tie for first, then 2 is next”.
  Writing a pair as `(1, 0)` denotes the same canonical unordered outcome
  `(0, 1)` and gives the same probability.
- Abilities are fitted under the same `Σ theta_i = 0` gauge and L2 penalty;
  the result includes the log likelihood, per-observation log contributions,
  and the **analytic free-coordinate gradient** at the optimum.
- Stable logsumexp is used per stage, so the pair term at
  `tau = 1e-300` underflows exactly to zero and a singleton-only path then
  agrees bit-for-bit with the original tie-free entry. The model is
  differentiable in `theta`; the score variable is `u = 1` on a singleton
  member and `u = 1/2` on each pair member, giving
  `g = E_p[u] − u_chosen`, `H = Var_p(u)` per stage.

### Tied-model interface

| Name | Purpose |
|---|---|
| `TiedObservation(choices=(), participants=None)` | Ordered singleton/pair groups, best first; same participant semantics as `Observation`. |
| `TiedPLResult` | `status, converged, iterations, theta, tau, log_likelihood, penalized_objective, log_contributions, gradient_free, gradient_norm, message`. |
| `fit_tied_plackett_luce(observations, n_objects, tau, *, lam=1.0, initial=None, max_iter=100, gtol=1e-10)` | Penalized sum-to-zero Newton fit with `tau` held fixed. |
| `tied_ranking_log_probability(choices, theta, tau, participants=None)` | Stable staged log probability. |
| `tied_ranking_probability(...)` | `exp` of the above (may underflow). |
| `tied_choice_probabilities(remaining, theta, tau)` | Enumerates all `|R|` singleton and `C(|R|,2)` pair probabilities of one joint choice (use on small sets); the tuple sums to 1. |
| `tied_penalized_objective(...)` / `tied_penalized_gradient_free(...)` / `tied_penalized_hessian_free(...)` | Objective and analytic derivatives in the `n−1` free coordinates. |

The same validation/error semantics apply (`PLInputError`, bool rejected,
finite numbers, indices in range, sum-to-zero enforced on user theta).
Additionally the following are rejected **before any computation**:

- a group of size 0 or size ≥ 3 (**three-way ties are not supported**);
- a pair repeating the same object, e.g. `(1, 1)`;
- the same object appearing in two different groups (each object is removed
  after its first selection);
- a chosen object absent from that observation's participant set;
- non-finite or non-positive `tau`.

### Tied-model accuracy checks

| Quantity checked | Reference method | Tolerance |
|---|---|---|
| Enumerated singleton/pair choice probabilities | Closed-form weights, common denominator; `fsum` normalization | `1e-14` absolute, sum exactly 1 |
| Staged tied log probability | Independent implementation in `tests/reference_ties.py` | `1e-12` absolute |
| Analytic free gradient | Independent central finite differences, `h = 1e-6` | `1e-6 × max(1,‖ref‖∞)` |
| `tau → 0` singleton path | Direct bit-for-bit equality with original `ranking_log_probability` (`tau = 1e-300`) | exact |
| Sum-to-zero, gradient at optimum, `Σ contributions = log-likelihood` | Direct | `1e-10` … `1e-12` |

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

- Implements exactly the contracted subset: tie-free top-k Plackett–Luce,
  one global L2 penalty, sum-to-zero gauge; plus the scoped two-way-tied
  extension in `plackett_luce.ties` (singletons and joint pairs in one common
  denominator, fixed `tau`). The tied extension is **not** presented as a
  unified standard for all real tied rankings: it does not support three-way
  or larger ties, variable tie widths, ties as partial orders / interval
  censoring, or inference over `tau`. Neither model offers object-specific or
  grouped penalties, hierarchical/covariate (Bradley–Terry–Luce with features)
  models, or Bayesian posterior/standard-error output, and no standards
  compliance beyond this contract is claimed.
- The original tie-free entry (`fit_plackett_luce`, `Observation`,
  `ranking_log_probability`) is preserved unchanged; it never interprets its
  inputs through the tied model.
- `tied_choice_probabilities` enumerates `|R| + C(|R|,2)` outcomes and is
  intended for small choice sets and normalization checks (n ≤ 20).
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
plackett_luce/ties.py         # at-most-two-way-tied extension
tests/reference.py            # INDEPENDENT reference (no core import)
tests/reference_ties.py       # independent tied-model reference (no core import)
tests/test_validation.py      # contract / error-semantics tests
tests/test_math.py            # analytic extremum, FD derivatives, fit quality
tests/test_ties_validation.py # tied-model contract / rejection tests
tests/test_ties_math.py       # tied normalization, FD gradient, tau->0, fit
demo.py                       # deterministic synthetic-data demonstration
demo_iteration.py             # deterministic tied-model iteration demonstration
```
