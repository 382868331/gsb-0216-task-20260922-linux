"""Mathematical behavior of the at-most-two-way-tied Plackett-Luce model.

All expected values come from closed forms or the independent reference in
tests/reference_ties.py (which never imports the package under test).
"""
import math
import unittest

from plackett_luce import (
    TiedObservation,
    fit_tied_plackett_luce,
    tied_choice_probabilities,
    tied_penalized_gradient_free,
    tied_penalized_hessian_free,
    tied_penalized_objective,
    tied_ranking_log_probability,
    ranking_log_probability,
)
from tests import reference_ties as ref


TAU = 1.3


def manual_stage_probs(remaining, theta, tau):
    singles = {}
    pairs = {}
    for i in remaining:
        singles[(i,)] = math.exp(theta[i])
    for a in range(len(remaining)):
        for b in range(a + 1, len(remaining)):
            i, j = sorted((remaining[a], remaining[b]))
            pairs[(i, j)] = tau * math.exp(0.5 * (theta[i] + theta[j]))
    weights = {**singles, **pairs}
    z = math.fsum(weights.values())
    return {k: v / z for k, v in weights.items()}


class ChoiceNormalizationTests(unittest.TestCase):
    def test_all_outcomes_enumerated_and_normalized(self):
        theta = [0.6, -0.1, -0.2, -0.3]
        probs = tied_choice_probabilities([0, 1, 2, 3], theta, TAU)
        # 4 singletons + C(4,2) = 6 pairs, no orderings
        self.assertEqual(len(probs), 4 + 6)
        self.assertAlmostEqual(math.fsum(p for _, p in probs), 1.0, places=14)

    def test_probabilities_match_closed_form_weights(self):
        theta = [0.4, -0.25, -0.15]
        probs = dict(tied_choice_probabilities([0, 1, 2], theta, TAU))
        expected = manual_stage_probs([0, 1, 2], theta, TAU)
        self.assertEqual(set(probs), set(expected))
        for outcome, p in probs.items():
            self.assertAlmostEqual(p, expected[outcome], places=14)

    def test_pair_weight_is_one_joint_outcome_not_two_orderings(self):
        # With theta = 0 the 3 pair weights are tau each; if a pair were the
        # average of two orderings the pair mass would be 2*tau/(3+2*3*tau).
        theta = [0.0, 0.0, 0.0]
        probs = dict(tied_choice_probabilities([0, 1, 2], theta, TAU))
        expected_pair = TAU / (3.0 + 3.0 * TAU)
        expected_single = 1.0 / (3.0 + 3.0 * TAU)
        for out in ((0,), (1,), (2,)):
            self.assertAlmostEqual(probs[out], expected_single, places=14)
        for out in ((0, 1), (0, 2), (1, 2)):
            self.assertAlmostEqual(probs[out], expected_pair, places=14)


class LikelihoodTests(unittest.TestCase):
    def test_multistage_log_prob_against_independent_reference(self):
        theta = [0.5, 0.1, -0.2, -0.4]
        groups = ((0, 1), (3,))
        got = tied_ranking_log_probability(groups, theta, TAU)
        want = ref.ref_log_prob(groups, (0, 1, 2, 3), theta, TAU)
        self.assertAlmostEqual(got, want, places=14)

    def test_pair_probability_scales_linearly_with_tau(self):
        theta = [0.0, 0.0, 0.0]
        lp1 = tied_ranking_log_probability(((0, 1),), theta, 1.0)
        lp2 = tied_ranking_log_probability(((0, 1),), theta, 2.0)
        # log P_pair(tau) = log tau - log(3 + 3 tau)
        self.assertAlmostEqual(
            lp2 - lp1, math.log(2.0 / 9.0) - math.log(1.0 / 6.0), places=14)
        self.assertAlmostEqual(lp1, -math.log(6.0), places=14)

    def test_chosen_pair_objects_leave_later_denominators(self):
        theta = [0.3, 0.0, -0.3]
        # After tie (0,1) only object 2 remains: P = 1.
        lp = tied_ranking_log_probability(((0, 1), (2,)), theta, TAU)
        first = manual_stage_probs([0, 1, 2], theta, TAU)[(0, 1)]
        self.assertAlmostEqual(lp, math.log(first), places=14)


class TauSmallLimitTests(unittest.TestCase):
    def test_tau_to_zero_reproduces_singleton_path_exactly(self):
        theta = [0.55, 0.05, -0.6]
        ranking = ((0,), (1,), (2,))
        for tau in (1e-60, 1e-150, 1e-300):
            tied_lp = tied_ranking_log_probability(ranking, theta, tau)
            plain_lp = ranking_log_probability([0, 1, 2], theta)
            self.assertEqual(tied_lp, plain_lp)

    def test_singleton_path_probability_monotone_as_tau_shrinks(self):
        theta = [0.4, -0.1, -0.3]
        taus = (1e6, 1.0, 1e-2, 1e-4, 1e-6)
        lps = [tied_ranking_log_probability(((0,),), theta, t) for t in taus]
        for a, b in zip(lps, lps[1:]):
            self.assertGreater(b, a)
        # far below floating-point resolution the singleton choice has
        # converged to its ordinary-PL value (denominator pair terms vanish)
        limit = tied_ranking_log_probability(((0,),), theta, 1e-300)
        plain = ranking_log_probability([0], theta)
        self.assertEqual(limit, plain)


class DerivativeTests(unittest.TestCase):
    def _batch(self):
        return [
            (((0, 1), (2,)), (0, 1, 2, 3)),
            (((2,),), (0, 1, 2, 3)),
            (((3, 0), (1,)), (0, 1, 2, 3)),
            (((1,), (0,)), (0, 1, 2)),
        ]

    def test_gradient_vs_independent_central_differences(self):
        n, lam = 4, 0.7
        obs = [TiedObservation(g, a) for g, a in self._batch()]
        x0 = [0.35, -0.1, 0.2]
        theta = ref.ref_free_theta(x0, n)
        got = tied_penalized_gradient_free(obs, theta, TAU, lam)
        want = ref.ref_free_gradient(self._batch(), x0, n, TAU, lam)
        scale = max(1.0, max(abs(v) for v in want))
        for k in range(n - 1):
            self.assertLessEqual(abs(got[k] - want[k]), 1e-6 * scale)

    def test_objective_vs_independent_reference(self):
        n, lam = 4, 0.7
        obs = [TiedObservation(g, a) for g, a in self._batch()]
        x0 = [0.2, -0.4, 0.1]
        theta = ref.ref_free_theta(x0, n)
        got = tied_penalized_objective(obs, theta, TAU, lam)
        want = ref.ref_free_objective(self._batch(), x0, n, TAU, lam)
        self.assertAlmostEqual(got, want, places=12)

    def test_gradient_zero_at_optimum_and_sum_zero(self):
        obs = [
            TiedObservation(((0, 1), (2,))),
            TiedObservation(((2,), (0,))),
            TiedObservation(((0,),), participants=(0, 1, 2)),
        ]
        result = fit_tied_plackett_luce(obs, 3, TAU, lam=0.5)
        self.assertTrue(result.converged, result.message)
        self.assertLessEqual(result.gradient_norm, 1e-10)
        self.assertLessEqual(abs(math.fsum(result.theta)), 1e-12)
        # contributions sum to log likelihood
        self.assertAlmostEqual(
            math.fsum(result.log_contributions), result.log_likelihood, places=12)

    def test_hessian_vs_independent_cross_differences(self):
        n, lam = 4, 0.7
        obs = [TiedObservation(g, a) for g, a in self._batch()]
        x0 = [0.35, -0.1, 0.2]
        theta = ref.ref_free_theta(x0, n)
        H = tied_penalized_hessian_free(obs, theta, TAU, lam)
        h = 1e-5

        def d_obj(x, k):
            xp = x[:]; xp[k] += h
            xm = x[:]; xm[k] -= h
            return (ref.ref_free_objective(self._batch(), xp, n, TAU, lam)
                    - ref.ref_free_objective(self._batch(), xm, n, TAU, lam)) / (2 * h)

        for k in range(n - 1):
            for l in range(n - 1):
                xp = x0[:]; xp[l] += h
                xm = x0[:]; xm[l] -= h
                fd = (d_obj(xp, k) - d_obj(xm, k)) / (2 * h)
                self.assertAlmostEqual(H[k][l], fd, delta=1e-4 * max(1.0, abs(fd)))
            self.assertGreater(H[k][k], 0.0)


class FitTests(unittest.TestCase):
    def test_identical_objects_tied_data_gives_zero_abilities(self):
        # Symmetric observations over 3 objects: optimum must stay at 0.
        obs = [
            TiedObservation(((0, 1),)),
            TiedObservation(((0, 2),)),
            TiedObservation(((1, 2),)),
        ]
        result = fit_tied_plackett_luce(obs, 3, TAU)
        self.assertTrue(result.converged)
        for value in result.theta:
            self.assertLessEqual(abs(value), 1e-10)

    def test_pairs_let_stronger_objects_rise(self):
        obs = [TiedObservation(((0, 1),)) for _ in range(20)]
        result = fit_tied_plackett_luce(obs, 3, TAU, lam=0.1)
        self.assertTrue(result.converged)
        # object 2 is never selected: it should be weaker than 0 and 1
        self.assertLess(result.theta[2], result.theta[0])
        self.assertLess(result.theta[2], result.theta[1])

    def test_fixed_tau_is_returned_not_estimated(self):
        result = fit_tied_plackett_luce([TiedObservation(((0, 1),))], 3, 0.42)
        self.assertEqual(result.tau, 0.42)


if __name__ == "main":
    unittest.main()
