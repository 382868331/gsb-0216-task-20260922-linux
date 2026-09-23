# Partial-Ranking Plackett–Luce Fitting

A small, dependency-free Python library for fitting the **Plackett–Luce**
model to **partial top-k rankings without ties**, with optional per-observation
participant sets, a sum-zero ability constraint and ridge regularisation.

- **Python:** 3.14.7 (uses only the standard library; works on any recent
  Python 3 with `math`/`dataclasses`)
- **Dependencies:** none — no third-party packages, no network, no accounts,
  no native build. Install Python and run.
- **Platform:** Linux-native, offline.

## Quick start

```bash
python -m unittest discover -s tests -v   # required tests
python demo.py                            # live fit + one triggered rejection
```

Both commands run from the repository root and finish in well under a second
(demo ~0.04 s; no sleeps, no pre-recorded results).

## Model

For observation `(i_1, …, i_k)` drawn from participant set *S*
(`k ≤ |S|`, ranked indices distinct):

```
p(obs | θ) = ∏_{j=1..k} exp(θ_{i_j}) /
             Σ_{a ∈ S \\ {i_1,…,i_{j-1}}} exp(θ_a)
```

- Unranked participants **stay in every remaining denominator** (top-k is a
  marginal prefix, not a complete ranking).
- Objects outside the observation's participant set **never enter any
  denominator**.
- `k = 0` contributes log-probability 0.
- Objective minimised on the hyperplane `Σ θ_i = 0`:

  `F(θ) = −Σ_obs log p(obs|θ) + λ/2 · ‖θ‖²`, with `λ > 0`.

  The ridge makes the restricted objective strictly convex, so a finite
  unique optimum always exists (there is no "proved unsolvable" case for
  valid inputs; the search can still exhaust its budget or stall
  numerically — see statuses below).

## Public interface

Import from the package root:

```python
from plackett_luce import (
    Ranking, fit_plackett_luce,
    log_ranking_likelihood, ranking_probability, winner_probabilities,
    CONVERGED, MAX_ITERATIONS, LINE_SEARCH_FAILED,
)
```

### `Ranking(ranked=(), participants=None)`

Frozen dataclass for one observation.

| field | meaning |
|---|---|
| `ranked` | sequence of distinct object indices, best first; may be empty |
| `participants` | optional distinct non-empty index sequence; `None` ⇒ all objects |

Plain sequences may be passed to `fit_plackett_luce` instead of `Ranking`
(in which case participants default to all objects).

### `fit_plackett_luce(rankings, *, n_objects=None, lam=1.0, max_iterations=100, tol=1e-9)`

- `n_objects`: `1 … 20`; inferred from indices when omitted.
- `lam`: finite float, strictly positive.
- `max_iterations`: non-negative `int` (booleans rejected).
- `tol`: finite float `> 0`, convergence threshold on the infinity norm of
  the gradient projected onto the sum-zero hyperplane.

Returns `FitResult` (frozen):

| field | meaning |
|---|---|
| `converged` / `status` | `converged`, `max_iterations` or `line_search_failed` |
| `iterations` | Newton steps actually taken |
| `n_objects`, `theta` (alias `abilities`) | fitted sum-zero abilities |
| `log_likelihood` | total unpenalised log-likelihood at `theta` |
| `penalty` | `λ/2 · ‖θ‖²`; `objective = −log_likelihood + penalty` |
| `grad_norm` | projected-gradient infinity norm at the returned point |
| `contributions` | per-observation `ObservationContribution(index, ranked, participants, log_likelihood, gradient)` |

**Status semantics (budget exhaustion vs. stall are distinct):**

- `max_iterations`: the Newton-step budget ran out before `tol` was reached.
  The result at the last point is returned; the problem is still solvable.
- `line_search_failed`: backtracking found no progress at the current point
  (no representable objective decrease and a full Newton step cannot shrink
  the projected gradient). Also returned if the Newton matrix is singular
  (not expected with `λ > 0`).

### Probability helpers

- `log_ranking_likelihood(ranked, theta, participants=None) -> float`
- `ranking_probability(...) -> float` (may underflow to `0.0` for extreme
  long shots; use the log form).
- `winner_probabilities(theta, participants=None) -> tuple[float, ...]`
  full-length softmax vector (0 for non-participants), entries sum to 1.

## Input limits and error semantics

- At most **20 objects**. Number of observations is unbounded.
- All numeric inputs must be finite: **NaN and ±Infinity raise
  `ValueError`** (`lam`, `tol`, every `theta` entry).
- Where an integer is required, **`bool` is rejected with `TypeError`**
  (`True` is not silently treated as `1`); non-`int`/non-`float` numbers
  raise `TypeError`.
- Invalid observations raise **`ValueError` with a locatable message** that
  includes the observation index and the offending object, covering:
  duplicate index inside `ranked` or `participants`; negative/out-of-range
  index; ranked object absent from the participant set; empty participant
  set; more ranked objects than participants.
- Validation happens before any fitting state is created, and the fit is a
  single pure function call — a rejected batch leaves no partial state.
- `n_objects=0` or `> 20`, `lam ≤ 0`, `tol ≤ 0`, `max_iterations < 0` are
  rejected.

## Numerical design and error budget

- Each selection level is evaluated with a **max-shifted logsumexp**;
  log-probabilities stay finite for arbitrary finite `θ` (verified for
  gaps of 1400, where naive `exp` would overflow).
- Gradient (`p − e_w` per level plus `λθ`) and Hessian
  (`diag(p) − ppᵀ` per level plus `λI`) are analytic.
- Optimisation: damped/trust-region-free **Newton with backtracking
  (Armijo)** on the sum-zero hyperplane — the gradient RHS and the step
  are projected to the tangent space; the dense (≤20×20) Newton system is
  solved by Gaussian elimination with partial pivoting.
- When the true objective decrease is smaller than one ULP of `F`, a full
  Newton step is still accepted as a root step while it keeps shrinking the
  projected gradient; otherwise the fit stops with `line_search_failed`
  instead of looping.

Reported floating-point accuracy (CPython `float`, IEEE-754 binary64):

| quantity | agreement |
|---|---|
| log-likelihood vs. independent direct-exp reference (`|θ| ≲ 5`) | 1e-12 absolute |
| analytic gradient vs. central finite differences (independent objective) | 1e-6 |
| analytic Hessian vs. finite-difference gradient (independent objective) | 1e-5 |
| two-object fit vs. bisection of the closed-form stationarity equation | 1e-10 on the gap |
| `Σ θ_i` at the returned point | ≤ 2e-14 typical |
| default-stationarity residual (`‖∇F projected‖∞`) | ≤ 1e-9 (the `tol` contract) |

These tolerances reflect binary64 rounding and the finite-difference step
sizes; they are not used to loosen boundary checks (indices, duplicates,
finiteness are exact rejections).

## Design trade-offs

- **Ridge required (`λ > 0`).** The unpenalised likelihood can diverge
  (e.g. one object always wins); the ridge guarantees a unique finite
  optimum and a nonsingular Newton matrix, matching the task contract.
- **Sum-zero parameterisation by projection**, rather than eliminating one
  coordinate: all abilities are returned symmetrically and code stays simple
  at n ≤ 20.
- **Dense direct linear algebra** (no sparse/iterative solver): at 20
  objects a 20×20 elimination is negligible and numerically reliable.
- **No ties, no partial-order DAGs, no dynamic/online fits, no tie-aware or
  stage-dependent models** — only the mathematical subset specified above.
  The library does not claim compatibility with any broader industry
  ranking-data standard.

## Repository layout

```
plackett_luce/__init__.py     public API re-exports
plackett_luce/model.py        validation, likelihood, Newton fit, helpers
tests/reference.py            independent brute-force reference (never
                              imports the package under test)
tests/test_plackett_luce.py   required unittest suite
demo.py                       fixed-input live demonstration
```

The independent reference defines the objective/gradient/Hessian directly
from the model using plain `exp`/`log`, plus central finite differences and
a closed-form two-object stationarity equation; expected test values come
from it. Round-trip calls through the package are used only as supplementary
consistency checks.

## Known limitations

- Probabilities (not log-probabilities) can underflow to `0.0` for very
  extreme long shots; log-probabilities of finite inputs remain finite.
- The implementation is single-threaded pure Python; sized for ≤ 20 objects
  and thousands of observations, not large-scale production workloads.
- Object identity is the integer index; labels/names are the caller's
  responsibility (the demo uses a local name table).
- Ties, missing-by-unknown-mechanism data and covariates are out of scope.
