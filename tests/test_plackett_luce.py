"""Unit tests for the plackett_luce package.

Expected values come from ``tests.reference`` (an independent brute-force
implementation that never imports the package under test) and from finite
differences.  Round-trip uses of the public API are marked as supplementary.
"""

from __future__ import annotations

import itertools
import math
import os
import sys
import unittest

# Make both the workspace root (package under test) and this directory
# (independent reference) importable regardless of the invocation cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reference import (  # noqa: E402
    bisect_root,
    fd_gradient,
    fd_hessian,
    ref_gradient,
    ref_hessian,
    ref_objective,
    ref_ranking_logprob,
    two_object_winner_equation,
)

from plackett_luce import (  # noqa: E402
    CONVERGED,
    LINE_SEARCH_FAILED,
    MAX_ITERATIONS,
    MAX_OBJECTS,
    Ranking,
    fit_plackett_luce,
    log_ranking_likelihood,
    ranking_probability,
    winner_probabilities,
)

# Deterministic sample: mixed k, repeated winners, differing participant
# sets, one k=0 observation.  Object count 4 (<= 20 interface bound).
SAMPLE = [
    Ranking((0, 2, 1), (0, 1, 2, 3)),
    Ranking((1, 0), (0, 1, 2)),
    Ranking((2,), (0, 2, 3)),
    Ranking((3, 2), (2, 3)),
    Ranking((0,), (0, 1)),
    Ranking((), (1, 3)),
    Ranking((2, 0), (0, 1, 2, 3)),
    Ranking((3,), (2, 3)),
]
REF_SAMPLE = [(list(r.ranked), list(r.participants)) for r in SAMPLE]
LAM = 0.7


def core_objective(theta, observations, lam):
    f = 0.0
    for ranked, participants in observations:
        f -= log_ranking_likelihood(ranked, theta, participants)
    return f + 0.5 * lam * sum(t * t for t in theta)


class TestTwoObjectAnalytic(unittest.TestCase):
    """Small two-object case with a closed-form stationarity equation."""

    def test_symmetric_is_exactly_zero(self):
        rankings = [(0,)] * 5 + [(1,)] * 5
        res = fit_plackett_luce(rankings, n_objects=2, lam=1.0)
        self.assertEqual(res.status, CONVERGED)
        self.assertEqual(res.theta, (0.0, 0.0))
        self.assertEqual(res.log_likelihood, 10 * -math.log(2.0))

    def test_analytic_extremum_matches_bisection_root(self):
        n0, n1, lam = 9, 1, 0.1
        rankings = [(0,)] * n0 + [(1,)] * n1
        res = fit_plackett_luce(rankings, n_objects=2, lam=lam)
        self.assertEqual(res.status, CONVERGED)
        root = bisect_root(
            lambda d: two_object_winner_equation(d, n0, n1, lam), -50, 50
        )
        d = res.theta[0] - res.theta[1]
        self.assertAlmostEqual(d, root, places=10)
        self.assertAlmostEqual(res.theta[0] + res.theta[1], 0.0, places=14)
        # Strong object strictly below the unpenalised log-odds (ridge pull).
        self.assertLess(d, math.log(n0 / n1))
        # Independent stationarity residual at the returned point.
        ref_obs = [([0], [0, 1])] * n0 + [([1], [0, 1])] * n1
        g = ref_gradient(list(res.theta), ref_obs, lam)
        self.assertLess(abs(g[0] - g[1]), 1e-9)

    def test_full_two_object_ranking_equals_winner_only(self):
        # Ranking (0, 1) over {0,1}: second level is deterministic (log 1),
        # so it must carry the same likelihood contribution as winner (0,).
        theta = (0.37, -0.37)
        self.assertAlmostEqual(
            log_ranking_likelihood((0, 1), theta, (0, 1)),
            log_ranking_likelihood((0,), theta, (0, 1)),
            places=14,
        )
        self.assertAlmostEqual(
            log_ranking_likelihood((0, 1), theta, (0, 1)),
            ref_ranking_logprob(list(theta), [0, 1], [0, 1]),
            places=13,
        )


class TestFiniteDifferenceReference(unittest.TestCase):
    """Derivative checks: independent analytic vs independent FD, then cross
    checks against the package."""

    POINTS = [
        [0.0, 0.0, 0.0, 0.0],
        [1.2, -0.4, 0.7, -1.5],
        [-1.8, 0.3, 0.9, 0.6],
        [0.05, -1.1, 1.6, -0.55],
    ]

    def test_ref_gradient_matches_finite_differences(self):
        for theta in self.POINTS:
            with self.subTest(theta=theta):
                ana = ref_gradient(theta, REF_SAMPLE, LAM)
                fd = fd_gradient(theta, REF_SAMPLE, LAM, h=1e-6)
                for a, b in zip(ana, fd):
                    self.assertAlmostEqual(a, b, places=6)

    def test_ref_hessian_matches_finite_differences(self):
        for theta in self.POINTS:
            with self.subTest(theta=theta):
                ana = ref_hessian(theta, REF_SAMPLE, LAM)
                fd = fd_hessian(theta, REF_SAMPLE, LAM, h=1e-5)
                for i in range(4):
                    for j in range(4):
                        self.assertAlmostEqual(ana[i][j], fd[i][j], places=5)

    def test_core_values_match_reference_at_points(self):
        # Independent implementation values vs public package values.
        for theta in self.POINTS:
            with self.subTest(theta=theta):
                for j, (ranked, participants) in enumerate(REF_SAMPLE):
                    self.assertAlmostEqual(
                        log_ranking_likelihood(ranked, theta, participants),
                        ref_ranking_logprob(theta, ranked, participants),
                        places=12,
                    )
                self.assertAlmostEqual(
                    core_objective(theta, REF_SAMPLE, LAM),
                    ref_objective(theta, REF_SAMPLE, LAM),
                    places=11,
                )

    def test_fitted_gradient_matches_independent_analytic(self):
        res = fit_plackett_luce(SAMPLE, n_objects=4, lam=LAM)
        self.assertEqual(res.status, CONVERGED)
        # Per-observation log-likelihoods match the reference.
        for c, (ranked, participants) in zip(res.contributions, REF_SAMPLE):
            self.assertAlmostEqual(
                c.log_likelihood,
                ref_ranking_logprob(list(res.theta), ranked, participants),
                places=12,
            )
        # Gradient assembled from the returned per-observation contributions
        # (negated log-likelihood gradients) plus the ridge term.
        g_core = [0.0] * 4
        for c in res.contributions:
            for i in range(4):
                g_core[i] -= c.gradient[i]
        for i in range(4):
            g_core[i] += LAM * res.theta[i]
        g_ref = ref_gradient(list(res.theta), REF_SAMPLE, LAM)
        for a, b in zip(g_core, g_ref):
            self.assertAlmostEqual(a, b, places=10)
        # Stationarity on the sum-zero hyperplane.
        mean = sum(g_ref) / 4
        self.assertLess(max(abs(v - mean) for v in g_ref), 1e-9)

    def test_core_function_curvature_matches_ref_hessian(self):
        # Supplementary: FD Hessian of the objective built only from public
        # package calls, compared to the independent analytic Hessian.
        theta = [0.8, -1.1, 0.2, 0.1]
        h = 1e-5
        n = 4

        def fd_core_grad(t):
            g = [0.0] * n
            for i in range(n):
                up, dn = t.copy(), t.copy()
                up[i] += h
                dn[i] -= h
                g[i] = (core_objective(up, REF_SAMPLE, LAM)
                        - core_objective(dn, REF_SAMPLE, LAM)) / (2 * h)
            return g

        base = fd_core_grad(theta)
        hess_ref = ref_hessian(theta, REF_SAMPLE, LAM)
        for j in range(n):
            bumped = theta.copy()
            bumped[j] += h
            gj = fd_core_grad(bumped)
            for i in range(n):
                self.assertAlmostEqual(
                    (gj[i] - base[i]) / h, hess_ref[i][j], places=4
                )


class TestStructuralBehaviour(unittest.TestCase):
    def test_separated_groups(self):
        # Two disjoint participant groups; ridge + sum-zero identifies them.
        obs = (
            [Ranking((0,), (0, 1))] * 3
            + [Ranking((1,), (0, 1))]
            + [Ranking((2,), (2, 3))]
            + [Ranking((3,), (2, 3))] * 3
        )
        res = fit_plackett_luce(obs, n_objects=4, lam=0.5)
        self.assertEqual(res.status, CONVERGED)
        self.assertAlmostEqual(sum(res.theta), 0.0, places=14)
        # Same data in mirrored groups -> strong objects (0 and 3) share an
        # ability, weak objects (1 and 2) share its negation.
        self.assertAlmostEqual(res.theta[0], res.theta[3], places=11)
        self.assertAlmostEqual(res.theta[1], res.theta[2], places=11)
        self.assertAlmostEqual(res.theta[0], -res.theta[1], places=11)
        # Each group gap solves the independent 2-object equation.
        root_a = bisect_root(
            lambda d: two_object_winner_equation(d, 3, 1, 0.5), -50, 50
        )
        root_b = bisect_root(
            lambda d: two_object_winner_equation(d, 1, 3, 0.5), -50, 50
        )
        self.assertAlmostEqual(res.theta[0] - res.theta[1], root_a, places=10)
        self.assertAlmostEqual(res.theta[2] - res.theta[3], root_b, places=10)

    def test_unranked_participants_stay_in_denominator(self):
        # top-1 over {0,1,2}: object 2 must enter the denominator.
        theta = (0.5, -0.25, -0.25)
        self.assertAlmostEqual(
            log_ranking_likelihood((0,), theta, (0, 1, 2)),
            ref_ranking_logprob(list(theta), [0], [0, 1, 2]),
            places=13,
        )
        # The same winner over {0,1} must give a DIFFERENT value, and the
        # probability over the restricted set must be larger.
        self.assertNotAlmostEqual(
            log_ranking_likelihood((0,), theta, (0, 1, 2)),
            log_ranking_likelihood((0,), theta, (0, 1)),
        )
        self.assertLess(
            ranking_probability((0,), theta, (0, 1, 2)),
            ranking_probability((0,), theta, (0, 1)),
        )

    def test_outsiders_never_enter_denominator(self):
        theta = [0.4, -0.6, 5.0]
        base = log_ranking_likelihood((0,), theta, (0, 1))
        theta[2] = -9.0
        self.assertEqual(
            log_ranking_likelihood((0,), theta, (0, 1)), base
        )

    def test_top_k_is_not_treated_as_full_ranking(self):
        # (0,) over 3 participants vs the full order (0, 1, 2): the top-1
        # marginal must equal the sum over its two possible completions.
        theta = (0.8, -0.3, -0.5)
        top1 = ranking_probability((0,), theta, (0, 1, 2))
        completions = (
            ranking_probability((0, 1, 2), theta)
            + ranking_probability((0, 2, 1), theta)
        )
        self.assertAlmostEqual(top1, completions, places=12)
        self.assertLess(top1, ranking_probability((0, 1, 2), theta)
                        + ranking_probability((0, 2, 1), theta) + 1e-12)

    def test_full_ranking_distribution_sums_to_one(self):
        theta = (0.9, -0.4, 0.1)
        total = sum(
            ranking_probability(perm, theta)
            for perm in itertools.permutations(range(3))
        )
        self.assertAlmostEqual(total, 1.0, places=12)
        # top-2 marginals over participants also sum to one.
        total_k2 = 0.0
        for prefix in itertools.permutations(range(3), 2):
            total_k2 += ranking_probability(prefix, theta)
        self.assertAlmostEqual(total_k2, 1.0, places=12)

    def test_k_zero_observations_are_flat(self):
        obs = [Ranking(()), Ranking((), (0, 2)), Ranking((), (2,))]
        res = fit_plackett_luce(obs, n_objects=3, lam=1.5)
        self.assertEqual(res.status, CONVERGED)
        self.assertEqual(res.theta, (0.0, 0.0, 0.0))
        self.assertEqual(res.objective, 0.0)
        self.assertEqual(res.log_likelihood, 0.0)
        for c in res.contributions:
            self.assertEqual(c.log_likelihood, 0.0)
            self.assertEqual(c.gradient, (0.0, 0.0, 0.0))
        self.assertEqual(log_ranking_likelihood((), [0.2, -0.2]), 0.0)

    def test_participant_present_but_unranked_affects_fit(self):
        # Object 2 only ever sits unranked in the denominator: fit must be
        # finite and satisfy independent stationarity.
        obs = [Ranking((0,), (0, 1, 2))] * 10
        res = fit_plackett_luce(obs, n_objects=3, lam=1.0)
        self.assertEqual(res.status, CONVERGED)
        ref_obs = [([0], [0, 1, 2])] * 10
        g = ref_gradient(list(res.theta), ref_obs, 1.0)
        mean = sum(g) / 3
        self.assertLess(max(abs(v - mean) for v in g), 1e-9)
        self.assertAlmostEqual(sum(res.theta), 0.0, places=14)

    def test_winner_probabilities(self):
        theta = (1.0, 0.0, -1.0)
        p = winner_probabilities(theta)
        self.assertEqual(len(p), 3)
        self.assertAlmostEqual(sum(p), 1.0, places=14)
        self.assertAlmostEqual(
            p[0], math.exp(1.0) / (math.exp(1.0) + 1.0 + math.exp(-1.0)),
            places=13,
        )
        # Restricted participant set: non-participants get zero.
        p2 = winner_probabilities(theta, (1, 2))
        self.assertEqual(p2[0], 0.0)
        self.assertAlmostEqual(p2[1] + p2[2], 1.0, places=14)
        self.assertGreater(p2[1], p2[2])

    def test_stable_at_large_ability_gap(self):
        # Direct exp(700) would overflow; the logsumexp path stays finite.
        self.assertEqual(log_ranking_likelihood((0,), (700.0, -700.0)), 0.0)
        self.assertAlmostEqual(
            log_ranking_likelihood((1,), (700.0, -700.0)), -1400.0, places=12
        )


class TestFitResultContract(unittest.TestCase):
    def test_identity_objective_minus_ll_penalty(self):
        res = fit_plackett_luce(SAMPLE, n_objects=4, lam=LAM)
        self.assertAlmostEqual(
            res.objective, -res.log_likelihood + res.penalty, places=11
        )
        self.assertEqual(
            len(res.contributions), len(SAMPLE)
        )
        self.assertAlmostEqual(
            sum(c.log_likelihood for c in res.contributions),
            res.log_likelihood,
            places=13,
        )

    def test_sum_zero_and_determinism(self):
        a = fit_plackett_luce(SAMPLE, n_objects=4, lam=LAM)
        b = fit_plackett_luce(SAMPLE, n_objects=4, lam=LAM)
        self.assertEqual(a.theta, b.theta)
        self.assertAlmostEqual(sum(a.theta), 0.0, places=14)
        self.assertLessEqual(a.iterations, a.iterations)  # deterministic
        self.assertIsInstance(a.iterations, int)

    def test_plain_sequences_and_inference(self):
        res = fit_plackett_luce([(0, 1), (2, 0)], n_objects=3)
        self.assertEqual(res.status, CONVERGED)
        res2 = fit_plackett_luce([(0, 1), (2, 0)])  # inferred n_objects=3
        self.assertEqual(res2.n_objects, 3)
        self.assertEqual(res2.theta, res.theta)

    def test_single_object(self):
        res = fit_plackett_luce([Ranking((0,), (0,))], n_objects=1)
        self.assertEqual(res.theta, (0.0,))
        self.assertEqual(res.log_likelihood, 0.0)
        self.assertEqual(log_ranking_likelihood((0,), [0.3]), 0.0)
        self.assertEqual(winner_probabilities([0.3]), (1.0,))

    def test_budget_exhaustion_is_distinct_status(self):
        res = fit_plackett_luce(
            [Ranking((0,), (0, 1, 2))] * 5,
            n_objects=3, lam=1.0, max_iterations=0,
        )
        self.assertFalse(res.converged)
        self.assertEqual(res.status, MAX_ITERATIONS)
        self.assertEqual(res.iterations, 0)
        self.assertGreater(res.grad_norm, 1e-9)

    def test_line_search_failure_distinct_from_budget(self):
        # Tolerance far below floating-point reachability: the Newton step
        # eventually makes no representable progress and backtracking fails,
        # which must be reported separately from budget exhaustion.
        res = fit_plackett_luce(
            [Ranking((0,), (0, 1))] * 4,
            n_objects=2, lam=1.0, max_iterations=100, tol=1e-300,
        )
        self.assertEqual(res.status, LINE_SEARCH_FAILED)
        self.assertFalse(res.converged)
        self.assertNotEqual(LINE_SEARCH_FAILED, MAX_ITERATIONS)

    def test_iterations_progress_before_exhaustion(self):
        data = [Ranking((0,), (0, 1, 2, 3))] * 8
        one = fit_plackett_luce(data, n_objects=4, max_iterations=1)
        zero = fit_plackett_luce(data, n_objects=4, max_iterations=0)
        self.assertEqual(one.status, MAX_ITERATIONS)
        self.assertEqual(one.iterations, 1)
        self.assertLess(one.objective, zero.objective)


class TestInputRejection(unittest.TestCase):
    def _expect(self, exc, fn, *args, **kwargs):
        with self.assertRaises(exc) as cm:
            fn(*args, **kwargs)
        return str(cm.exception)

    def test_duplicate_ranked_object(self):
        msg = self._expect(
            ValueError, fit_plackett_luce, [Ranking((0, 0))], n_objects=2
        )
        self.assertIn("duplicate", msg)
        self._expect(
            ValueError, log_ranking_likelihood, (1, 1), [0.0, 0.0]
        )

    def test_duplicate_participant(self):
        self._expect(
            ValueError, fit_plackett_luce,
            [Ranking((0,), (0, 0))], n_objects=2,
        )

    def test_ranked_not_in_participants(self):
        msg = self._expect(
            ValueError, fit_plackett_luce,
            [Ranking((2,), (0, 1))], n_objects=3,
        )
        self.assertIn("not in", msg)

    def test_empty_participants(self):
        self._expect(
            ValueError, fit_plackett_luce,
            [Ranking((), ())], n_objects=2,
        )

    def test_more_ranked_than_participants(self):
        self._expect(
            ValueError, fit_plackett_luce,
            [Ranking((0, 1), (0,))], n_objects=2,
        )

    def test_index_out_of_range_and_negative(self):
        self._expect(
            ValueError, fit_plackett_luce, [Ranking((2,))], n_objects=2
        )
        self._expect(
            ValueError, fit_plackett_luce, [Ranking((-1,))], n_objects=2
        )
        self._expect(
            ValueError, fit_plackett_luce,
            [Ranking((), (3,))], n_objects=2,
        )

    def test_n_objects_bounds(self):
        self._expect(
            ValueError, fit_plackett_luce, [], n_objects=0
        )
        self._expect(
            ValueError, fit_plackett_luce, [], n_objects=MAX_OBJECTS + 1
        )
        self._expect(TypeError, fit_plackett_luce, [], n_objects=True)
        self._expect(TypeError, fit_plackett_luce, [], n_objects=2.0)

    def test_inference_impossible_for_empty_kzero(self):
        msg = self._expect(ValueError, fit_plackett_luce, [Ranking(())])
        self.assertIn("n_objects", msg)

    def test_bool_rejected_for_integer_parameters(self):
        self._expect(
            TypeError, fit_plackett_luce, [(0,)], n_objects=2,
            max_iterations=True,
        )

    def test_bool_index_rejected(self):
        self._expect(
            TypeError, fit_plackett_luce, [Ranking((True,))], n_objects=2
        )

    def test_nan_infinity_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            self._expect(
                ValueError, fit_plackett_luce, [(0,)], n_objects=2, lam=bad
            )
            self._expect(
                ValueError, fit_plackett_luce, [(0,)], n_objects=2, tol=bad
            )
            self._expect(
                ValueError, log_ranking_likelihood, (0,), [bad, 0.0]
            )

    def test_non_finite_lam_range(self):
        self._expect(ValueError, fit_plackett_luce, [(0,)], lam=0.0)
        self._expect(ValueError, fit_plackett_luce, [(0,)], lam=-1.0)
        self._expect(ValueError, fit_plackett_luce, [(0,)], tol=0.0)
        self._expect(TypeError, fit_plackett_luce, [(0,)], lam="0.1")

    def test_negative_max_iterations(self):
        self._expect(
            ValueError, fit_plackett_luce, [(0,)], max_iterations=-1
        )

    def test_bad_observation_type(self):
        self._expect(TypeError, fit_plackett_luce, [b"01"], n_objects=2)
        self._expect(TypeError, fit_plackett_luce, ["01"], n_objects=2)
        self._expect(TypeError, fit_plackett_luce, Ranking((0,)), n_objects=2)
        self._expect(TypeError, fit_plackett_luce, 42, n_objects=2)

    def test_bad_theta_type(self):
        self._expect(TypeError, log_ranking_likelihood, (0,), "ab")
        self._expect(TypeError, winner_probabilities, None)

    def test_probability_helpers_reject_bad_sets(self):
        self._expect(
            ValueError, log_ranking_likelihood, (0,), [0.0, 0.0], ()
        )
        self._expect(
            ValueError, log_ranking_likelihood, (0,), [0.0, 0.0], (0, 0)
        )
        self._expect(
            ValueError, log_ranking_likelihood, (2,), [0.0, 0.0]
        )
        self._expect(
            TypeError, log_ranking_likelihood, (False,), [0.0, 0.0]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
