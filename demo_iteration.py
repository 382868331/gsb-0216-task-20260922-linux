#!/usr/bin/env python3
"""Deterministic, offline demonstration of the two-way-tied selection model.

Run:  python demo_iteration.py

Everything is computed live (no recorded outputs, no sleep): a small choice
set is enumerated to show the common normalization, a partial ranking with a
tied pair is scored and fitted, the tau -> 0 limit is checked against the
original Plackett-Luce entry, and one actual contract rejection is shown.
Standard library only.
"""
import math
import time

from plackett_luce import (
    PLInputError,
    TiedObservation,
    fit_tied_plackett_luce,
    ranking_log_probability,
    tied_choice_probabilities,
    tied_ranking_log_probability,
)


def main():
    started = time.perf_counter()
    theta = (0.55, 0.10, -0.25, -0.40)
    tau = 1.3

    print("=== At-most-two-way-tied Plackett-Luce iteration ===")
    print(f"theta = {theta}")
    print(f"tau   = {tau} (fixed pair-weight multiplier, not estimated)")

    # 1. Enumerate every singleton and pair probability for one common choice.
    print("\n[1] One joint choice over objects {0,1,2,3}: "
          "4 singletons + 6 unordered pairs")
    probs = tied_choice_probabilities([0, 1, 2, 3], theta, tau)
    for outcome, p in probs:
        label = (f"{{{outcome[0]}}}" if len(outcome) == 1
                 else f"{{{outcome[0]},{outcome[1]}}} (tie)")
        print(f"    P({label:>14}) = {p:.6f}")
    total = math.fsum(p for _, p in probs)
    print(f"    sum of all {len(probs)} probabilities = {total!r}")

    # 2. A partial ranking with a real tied pair; chosen objects disappear.
    print("\n[2] Partial ranking: tie {0,1} first, then singleton {2} "
          "(object 3 unranked, stays in denominators)")
    choices = ((0, 1), (2,))
    lp = tied_ranking_log_probability(choices, theta, tau)
    print(f"    log P = {lp:.10f}   P = {math.exp(lp):.8f}")
    # Same joint outcome written in reverse order: identical probability.
    lp_rev = tied_ranking_log_probability(((1, 0), (2,)), theta, tau)
    print(f"    same pair written (1,0): log P = {lp_rev:.10f} "
          f"(equal: {lp == lp_rev})")

    # 3. Fit abilities from a small deterministic batch (sum theta = 0).
    print("\n[3] Fitting abilities on a deterministic batch "
          "(Newton, sum-to-zero, lambda = 0.5)")
    observations = [
        TiedObservation(((0, 1), (2,))),
        TiedObservation(((0, 1), (3,))),
        TiedObservation(((0, 2),)),
        TiedObservation(((1, 0), (3,))),
        TiedObservation(((0,), (1,))),
    ]
    result = fit_tied_plackett_luce(
        observations, 4, tau, lam=0.5)
    print(f"    status         = {result.status.value}")
    print(f"    iterations     = {result.iterations}")
    print(f"    theta_hat      = {tuple(round(v, 6) for v in result.theta)}")
    print(f"    sum(theta_hat) = {math.fsum(result.theta):.3e}")
    print(f"    log likelihood = {result.log_likelihood:.10f}")
    print(f"    |gradient|_inf = {result.gradient_norm:.3e}")
    print(f"    per-observation log contributions = "
          f"{tuple(round(c, 6) for c in result.log_contributions)}")

    # 4. tau -> 0: pair weights vanish and the singleton path coincides with
    #    the original tie-free Plackett-Luce entry bit-for-bit.
    print("\n[4] tau -> 0 limit on a singleton-only path:")
    singleton_path = ((0,), (1,), (2,))
    tied_lp = tied_ranking_log_probability(singleton_path, theta, 1e-300)
    plain_lp = ranking_log_probability([0, 1, 2], theta)
    print(f"    tied model (tau=1e-300): {tied_lp:.15f}")
    print(f"    original PL entry:       {plain_lp:.15f}")
    print(f"    identical: {tied_lp == plain_lp}")
    pair_lp = tied_ranking_log_probability(((0, 1),), theta, 1e-300)
    print(f"    (a genuinely tied pair becomes correspondingly rare: "
          f"log P = {pair_lp:.3g})")

    # 5. Boundary: an illegal observation is rejected before any fitting.
    print("\n[5] Boundary case (rejected up front, actually raised):")
    bad = [TiedObservation(((0, 1, 2),))]  # three-way tie
    try:
        fit_tied_plackett_luce(bad, 4, tau)
    except PLInputError as exc:
        print(f"    PLInputError: {exc}")
    else:
        raise RuntimeError("expected PLInputError for a three-way tie")

    elapsed = time.perf_counter() - started
    print(f"\nFinished live in {elapsed:.3f} s.")


if __name__ == "__main__":
    main()
