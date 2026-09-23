"""Mathematical correctness tests against independently derived references.

No expected value here is produced by calling the package's own core; the
references live in tests/reference.py (bisection, independently coded
logsumexp, central finite differences). Round-trip checks appear only in
test_algorithm as a supplement.
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plackett_luce import (  # noqa: E402
    ConvergenceStatus,
    Observation,
    fit_plackett_luce,
    penalized_gradient_free,
    penalized_hessian_free,
    penalized_objective,
    ranking_log_probability,
    ranking_probability,
)
import reference as ref  # noqa: E402


def records_from(obs, n):
    out = []
    for ob in obs:
        part = tuple(range(n)) if ob.participants is None else tuple(ob.participants)
        out.append((tuple(ob.ranking), part))
    return out


class TwoObjectAnalyticTests(unittest.TestCase):
    """Two objects: closed-form logistic extremum, matched by independent solve."""

    def test_top1_extremum(self):
        m, lam = 7, 0.5
        obs = [Observation((0,)) for _ in range(m)]
        r = fit_plackett_luce(obs, 2, lam=lam)
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED, r.message)
        a = ref.two_object_winner_root(m, lam)
        self.assertAlmostEqual(r.theta[0], a, places=10)
        self.assertAlmostEqual(r.theta[1], -a, places=10)

    def test_full_pair_extremum_equivalent_to_top1(self):
        # A full ranking (0>1) has one Plackett stage: same logistic term.
        m, lam = 3, 1.0
        r1 = fit_plackett_luce([Observation((0,))] * m, 2, lam=lam)
        r2 = fit_plackett_luce([Observation((0, 1))] * m, 2, lam=lam)
        for a, b in zip(r1.theta, r2.theta):
            self.assertAlmostEqual(a, b, places=12)
        a = ref.two_object_winner_root(m, lam)
        self.assertAlmostEqual(r2.theta[0], a, places=10)

    def test_analytic_stationarity_equation(self):
        # lam*a = m*sigmoid(-2a) at the optimum; check directly.
        m, lam = 5, 0.25
        r = fit_plackett_luce([Observation((0,))] * m, 2, lam=lam)
        a = r.theta[0]
        self.assertAlmostEqual(lam * a, m / (1.0 + math.exp(2.0 * a)), places=10)
        self.assertLessEqual(r.gradient_norm, 1e-10)

    def test_penalty_shrinks_toward_zero(self):
        obs = [Observation((0,))] * 20
        weak = fit_plackett_luce(obs, 2, lam=0.01)
        strong = fit_plackett_luce(obs, 2, lam=10.0)
        self.assertGreater(weak.theta[0], strong.theta[0])
        self.assertGreater(strong.theta[0], 0.0)

    def test_even_split_is_zero(self):
        obs = [Observation((0,)), Observation((1,))] * 10
        r = fit_plackett_luce(obs, 2)
        self.assertAlmostEqual(r.theta[0], 0.0, places=12)
        self.assertAlmostEqual(r.theta[1], 0.0, places=12)


class LikelihoodReferenceTests(unittest.TestCase):
    def test_logprob_matches_independent_reference(self):
        cases = [
            ((0, 1, 2), (0, 1, 2, 3), (0.4, -0.1, 0.25, -0.55)),
            ((2, 0), (0, 1, 2), (1.2, -0.7, -0.5)),
            ((1,), (0, 1, 2, 3, 4), (0.3, 0.9, -0.6, -0.2, -0.4)),
            ((), (0, 1), (0.5, -0.5)),
            ((0, 2), (0, 1, 2, 3), (-3.0, 2.5, 1.0, -0.5)),
        ]
        for ranking, part, theta in cases:
            with self.subTest(case=(ranking, part)):
                got = ranking_log_probability(ranking, theta, part)
                want = ref.ref_log_prob(ranking, part, list(theta))
                self.assertAlmostEqual(got, want, places=12)
                self.assertLessEqual(ranking_probability(ranking, theta, part), 1.0)

    def test_unnranked_participants_stay_in_denominator(self):
        # top-1 of (0) among {0,1,2} differs from a full ranking 0>1>2.
        theta = (0.6, -0.2, -0.4)
        top1 = ranking_log_probability((0,), theta, (0, 1, 2))
        full = ranking_log_probability((0, 1, 2), theta, (0, 1, 2))
        self.assertNotAlmostEqual(top1, full)
        # Independent explicit fractions.
        p_full = (math.exp(0.6) / sum(math.exp(t) for t in theta)) * \
                 (math.exp(-0.2) / (math.exp(-0.2) + math.exp(-0.4)))
        self.assertAlmostEqual(math.exp(full), p_full, places=12)
        p_top1 = math.exp(0.6) / sum(math.exp(t) for t in theta)
        self.assertAlmostEqual(math.exp(top1), p_top1, places=12)

    def test_nonparticipants_excluded(self):
        theta = (-2.4, -2.6, 5.0)  # object 2 very strong but absent
        present = ranking_log_probability((0,), theta, (0, 1))
        absent = ranking_log_probability((0,), theta, (0, 1, 2))
        self.assertGreater(present, absent)
        # Among {0,1} only: logistic on the gap (shift-invariant check).
        self.assertAlmostEqual(
            present, math.log(math.exp(0.2) / (1.0 + math.exp(0.2))), places=12)

    def test_k0_is_probability_one(self):
        self.assertEqual(ranking_log_probability((), (0.3, -0.3)), 0.0)
        self.assertEqual(ranking_probability((), (0.3, -0.3)), 1.0)

    def test_large_abilities_no_overflow(self):
        theta = (700.0, -700.0)
        lp = ranking_log_probability((0,), theta)
        self.assertTrue(math.isfinite(lp))
        self.assertAlmostEqual(lp, 0.0, places=6)


class DerivativeTests(unittest.TestCase):
    """Analytic gradient/Hessian vs independent central finite differences."""

    CASES = [
        # (n, lam, observations)
        (3, 0.5, [Observation((0, 1)), Observation((1,), (0, 1, 2)),
                  Observation(())]),
        (4, 1.0, [Observation((2, 0)), Observation((0, 1, 2, 3)),
                  Observation((3,), (1, 3))]),
        (2, 0.1, [Observation((0,)), Observation((1,))]),
        (5, 2.0, [Observation((4, 3, 2)), Observation((0,), (0, 2, 4)),
                  Observation((1, 0), (0, 1, 2))]),
    ]

    def test_gradient_finite_difference(self):
        for n, lam, obs in self.CASES:
            with self.subTest(n=n, lam=lam):
                x = [0.1 * (i + 1) * (1 if i % 2 else -1) for i in range(n - 1)]
                theta = ref.ref_free_theta(x, n)
                got = penalized_gradient_free(obs, theta, lam)
                want = ref.fd_gradient(x, records_from(obs, n), n, lam)
                scale = max(1.0, max(abs(v) for v in want))
                for g, w in zip(got, want):
                    self.assertAlmostEqual(g, w, delta=1e-6 * scale)

    def test_hessian_finite_difference(self):
        for n, lam, obs in self.CASES:
            with self.subTest(n=n, lam=lam):
                x = [0.1 * (i + 1) * (1 if i % 2 else -1) for i in range(n - 1)]
                theta = ref.ref_free_theta(x, n)
                got = penalized_hessian_free(obs, theta, lam)
                want = ref.fd_hessian(x, records_from(obs, n), n, lam)
                scale = max(1.0, max(abs(v) for row in want for v in row))
                for i in range(n - 1):
                    for j in range(n - 1):
                        self.assertAlmostEqual(
                            got[i][j], want[i][j], delta=1e-4 * scale)

    def test_objective_matches_reference(self):
        obs = [Observation((0, 1)), Observation((2,), (1, 2, 3))]
        n, lam = 4, 0.7
        x = [0.3, -0.6, 0.2]
        theta = ref.ref_free_theta(x, n)
        got = penalized_objective(obs, theta, lam)
        want = ref.ref_objective_free(x, records_from(obs, n), n, lam)
        self.assertAlmostEqual(got, want, places=12)


class FitQualityTests(unittest.TestCase):
    def test_recovers_known_abilities(self):
        # Deterministic pseudo-data generated from fixed theta via explicit
        # Plackett-Luce stage probabilities (no core call used to make data).
        rng_state = 123456789

        def rng():
            nonlocal rng_state
            rng_state = (1103515245 * rng_state + 12345) & 0x7fffffff
            return rng_state / 0x7fffffff

        true_theta = [0.8, 0.2, -0.3, -0.7]
        n = 4
        obs = []
        n_rounds = 600
        for _ in range(n_rounds):
            remaining = list(range(n))
            ranking = []
            while remaining:
                weights = [math.exp(true_theta[i]) for i in remaining]
                total = sum(weights)
                u = rng() * total
                acc = 0.0
                chosen = remaining[-1]
                for idx, w in zip(remaining, weights):
                    acc += w
                    if u <= acc:
                        chosen = idx
                        break
                ranking.append(chosen)
                remaining.remove(chosen)
            obs.append(Observation(tuple(ranking[:2])))  # top-2 partial

        r = fit_plackett_luce(obs, n, lam=0.02)
        self.assertTrue(r.converged, r.message)
        # Shift-invariance: compare differences from object 0.
        shifts = [r.theta[i] - true_theta[i] for i in range(n)]
        spread = max(shifts) - min(shifts)
        self.assertLess(spread, 0.25,
                        f"recovered abilities too far from truth: {r.theta}")
        self.assertEqual(len(r.log_contributions), n_rounds)
        self.assertAlmostEqual(
            math.fsum(r.log_contributions), r.log_likelihood, places=10)

    def test_separated_rankings_chain(self):
        # Observation 1 only involves objects 0,1; observation 2 only 2,3;
        # still jointly estimated under the shared penalty/sum-zero constraint.
        obs = [Observation((0, 1)), Observation((2, 3))]
        r = fit_plackett_luce(obs, 4, lam=1.0)
        self.assertTrue(r.converged)
        # Symmetry: theta0 = -theta1 and theta2 = -theta3, and pairs equal.
        self.assertAlmostEqual(r.theta[0], -r.theta[1], places=10)
        self.assertAlmostEqual(r.theta[2], -r.theta[3], places=10)
        self.assertAlmostEqual(r.theta[0], r.theta[2], places=10)

    def test_different_participant_sets(self):
        # Object 0 wins a race among {0,1}; object 2 wins among {2,3}.
        obs = [Observation((0,), participants=(0, 1)),
               Observation((2,), participants=(2, 3))] * 5
        r = fit_plackett_luce(obs, 4, lam=0.5)
        self.assertTrue(r.converged)
        self.assertGreater(r.theta[0], r.theta[1])
        self.assertGreater(r.theta[2], r.theta[3])

    def test_contributions_are_per_observation(self):
        obs = [Observation((0, 1)), Observation((1,), (0, 1))]
        r = fit_plackett_luce(obs, 2, lam=1.0)
        c0 = ranking_log_probability((0, 1), r.theta)
        c1 = ranking_log_probability((1,), r.theta, (0, 1))
        self.assertAlmostEqual(r.log_contributions[0], c0, places=12)
        self.assertAlmostEqual(r.log_contributions[1], c1, places=12)
        self.assertAlmostEqual(math.fsum(r.log_contributions),
                               r.log_likelihood, places=12)

    def test_sum_zero_and_minimum_property(self):
        obs = [Observation((0, 1, 2)), Observation((1,), (1, 2))]
        r = fit_plackett_luce(obs, 3, lam=0.3)
        self.assertAlmostEqual(math.fsum(r.theta), 0.0, delta=1e-12)
        f0 = penalized_objective(obs, list(r.theta), 0.3)
        for direction in ([1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [1.0, 0.0, -1.0]):
            eps = 1e-5
            tp = [r.theta[i] + eps * direction[i] for i in range(3)]
            tm = [r.theta[i] - eps * direction[i] for i in range(3)]
            self.assertGreaterEqual(penalized_objective(obs, tp, 0.3), f0 - 1e-9)
            self.assertGreaterEqual(penalized_objective(obs, tm, 0.3), f0 - 1e-9)


class BudgetTests(unittest.TestCase):
    def test_budget_exhaustion_distinct_status(self):
        obs = [Observation((0,))] * 50
        r = fit_plackett_luce(obs, 2, lam=0.01, max_iter=1)
        self.assertEqual(r.status, ConvergenceStatus.MAX_ITERATIONS)
        self.assertFalse(r.converged)
        self.assertEqual(r.iterations, 1)
        self.assertIn("budget", r.message)

    def test_converges_normally_with_default_budget(self):
        obs = [Observation((0,))] * 50
        r = fit_plackett_luce(obs, 2, lam=0.01)
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED)
        self.assertLessEqual(r.gradient_norm, 1e-10)


if __name__ == "__main__":
    unittest.main()
