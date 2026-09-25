"""Tests for the anchored-fit entry: fixed abilities + free box bounds.

Expected values come from closed-form two-object logistic expressions and
central finite differences of the INDEPENDENT reference in reference.py.
No expected value is obtained by calling the package's own solver.
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
    PLInputError,
    anchored_gradient_free,
    anchored_hessian_free,
    anchored_penalized_objective,
    fit_anchored,
    fit_plackett_luce,
    ranking_log_probability,
)
import plackett_luce.model as model  # noqa: E402
import reference as ref  # noqa: E402


def records_from(obs, n):
    out = []
    for ob in obs:
        part = tuple(range(n)) if ob.participants is None else tuple(ob.participants)
        out.append((tuple(ob.ranking), part))
    return out


class StrongAnchorTwoObjectTests(unittest.TestCase):
    def test_two_object_anchor_stationarity(self):
        # m observations where object 0 wins over object 1. With theta_1
        # pinned at c, theta_0 solves lam*theta_0 = m/(1+exp(c-theta_0)).
        m, lam, c = 6, 0.4, 0.8
        obs = [Observation((0,))] * m
        r = fit_anchored(obs, 2, {1: c}, lam=lam)
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED, r.message)
        # Anchor is returned bit-for-bit and never recentered.
        self.assertEqual(r.theta[1], c)
        self.assertNotAlmostEqual(math.fsum(r.theta), 0.0)
        x = r.theta[0]
        # df/dx = m*sigmoid(c-x) - m + lam*x = 0
        residual = m / (1.0 + math.exp(c - x)) - m + lam * x
        self.assertAlmostEqual(residual, 0.0, delta=1e-7)
        p01 = math.exp(x) / (math.exp(x) + math.exp(c))
        self.assertAlmostEqual(math.exp(r.log_contributions[0]), p01, places=12)
        self.assertAlmostEqual(r.log_likelihood, m * math.log(p01), places=12)

    def test_anchor_stationarity_independent_bisection(self):
        # Independently bisect the scalar stationarity equation and compare.
        m, lam, c = 5, 0.3, -0.6
        obs = [Observation((0,))] * m
        ra = fit_anchored(obs, 2, {1: c}, lam=lam)

        def equation(x):
            return m / (1.0 + math.exp(c - x)) - m + lam * x

        # Root lies in [0, m/lam]; bracket generically. Sign flips once.
        lo, hi = 0.0, float(m) / lam
        flo = equation(lo)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fm = equation(mid)
            if abs(fm) < 1e-13:
                break
            if (flo < 0.0) == (fm < 0.0):
                lo, flo = mid, fm
            else:
                hi = mid
        self.assertAlmostEqual(ra.theta[0], 0.5 * (lo + hi), places=9)

    def test_strong_anchor_dominates(self):
        obs = [Observation((0,))] * 3
        # Object 1 pinned extremely strong; with a huge free-coordinate
        # penalty the free optimum is pulled essentially to zero (the data
        # pull can no longer support a large positive theta_0).
        r = fit_anchored(obs, 2, {1: 5.0}, lam=1e6)
        self.assertTrue(r.converged)
        self.assertEqual(r.theta[1], 5.0)
        self.assertLess(abs(r.theta[0]), 1e-4)
        # A moderate penalty leaves a clearly positive (data-driven) value.
        r2 = fit_anchored(obs, 2, {1: 5.0}, lam=0.01)
        self.assertGreater(r2.theta[0], r.theta[0])


class UnobservedFreeObjectTests(unittest.TestCase):
    def test_free_object_with_no_samples_penalized_to_zero(self):
        n = 3
        obs = [Observation((0,), participants=(0, 1)),
               Observation((1,), participants=(0, 1))] * 4
        r = fit_anchored(obs, n, {0: 0.4}, lam=0.7)
        self.assertTrue(r.converged, r.message)
        # Object 2 appears in no participant set; its only force is the
        # penalty, so its free optimum is exactly zero.
        self.assertEqual(r.theta[0], 0.4)
        self.assertAlmostEqual(r.theta[2], 0.0, places=11)
        # Symmetric data: theta_1 optimum solves 4*sigmoid(t-0.4)-4+0.7t=0;
        # the penalty anchors it near zero, not at the constant 0.4.
        # Independently bisect rather than assert the anchor value.
        # Four (0>) and four (>1) pairings vs anchored object 0 at 0.4:
        # df/dt = 4*sigmoid(t-0.4) + 4*(sigmoid(t-0.4)-1) + 0.7 t
        #       = 4*(2*sigmoid(t-0.4)-1) + 0.7 t = 0.
        def eq(t):
            s = 1.0 / (1.0 + math.exp(0.4 - t))
            return 4.0 * (2.0 * s - 1.0) + 0.7 * t
        lo, hi = -10.0, 10.0
        flo = eq(lo)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fm = eq(mid)
            if abs(fm) < 1e-13:
                break
            if (flo < 0.0) == (fm < 0.0):
                lo, flo = mid, fm
            else:
                hi = mid
        self.assertAlmostEqual(r.theta[1], 0.5 * (lo + hi), places=8)
        # Its contributions never include index 2, even with wild theta_2.
        lp = ranking_log_probability((0,), (0.4, r.theta[1], 99.0),
                                     participants=(0, 1),
                                     fixed_theta={2: 99.0})
        self.assertAlmostEqual(
            lp, ref.ref_log_prob((0,), (0, 1), [0.4, r.theta[1], 99.0]),
            places=12)


class PartialParticipantSetsTests(unittest.TestCase):
    def test_anchor_and_unranked_participants_in_denominators(self):
        n = 4
        anchors = {3: 1.2}
        obs = [
            Observation((0, 1), participants=(0, 1, 3)),
            Observation((2,), participants=(0, 2, 3)),
            Observation((1,), participants=(1, 2)),
            Observation(()),
        ]
        r = fit_anchored(obs, n, anchors, lam=0.5)
        self.assertTrue(r.converged, r.message)
        self.assertEqual(r.theta[3], 1.2)
        self.assertEqual(r.log_contributions[3], 0.0)
        # Top-1 denominator for obs 1 includes fixed object 3 and unranked 0.
        th = list(r.theta)
        want = math.exp(th[2]) / (math.exp(th[0]) + math.exp(th[2])
                                  + math.exp(th[3]))
        self.assertAlmostEqual(math.exp(r.log_contributions[1]), want, places=12)
        self.assertAlmostEqual(math.fsum(r.log_contributions),
                               r.log_likelihood, places=10)


class AllFixedTests(unittest.TestCase):
    def test_all_fixed_is_pure_evaluation(self):
        anchors = {0: 0.3, 1: -0.2, 2: 1.1}
        obs = [Observation((0, 1), participants=(0, 1, 2)), Observation(())]
        r = fit_anchored(obs, 3, anchors, lam=0.9)
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED)
        self.assertEqual(r.iterations, 0)
        self.assertEqual(r.gradient_norm, 0.0)
        self.assertEqual(r.boundary_kkt_residual, 0.0)
        self.assertEqual(tuple(r.theta), (0.3, -0.2, 1.1))
        want0 = ref.ref_log_prob((0, 1), (0, 1, 2), [0.3, -0.2, 1.1])
        self.assertAlmostEqual(r.log_contributions[0], want0, places=12)
        self.assertEqual(r.log_contributions[1], 0.0)
        # No penalty is charged on fixed slots: objective == -loglik.
        self.assertAlmostEqual(r.penalized_objective, -r.log_likelihood, places=12)


class AnchoredDerivativeTests(unittest.TestCase):
    CASES = [
        (4, {3: 0.9}, 0.4,
         [Observation((0, 1), participants=(0, 1, 3)),
          Observation((2,), participants=(0, 2, 3))]),
        (3, {0: -0.5, 2: 0.5}, 0.8,
         [Observation((1,), participants=(0, 1, 2))]),
        (5, {4: -1.3}, 1.2,
         [Observation((3, 2), participants=(1, 2, 3, 4)),
          Observation((0,), participants=(0, 2, 4))]),
    ]

    def _theta(self, n, anchors, free_vals):
        free = [i for i in range(n) if i not in anchors]
        return ref.ref_anchored_theta(free, free_vals, anchors, n)

    def test_gradient_central_differences(self):
        for n, anchors, lam, obs in self.CASES:
            with self.subTest(anchors=anchors):
                free = [i for i in range(n) if i not in anchors]
                x = [0.15 * (k + 1) * (1 if k % 2 else -1)
                     for k in range(len(free))]
                theta = self._theta(n, anchors, x)
                idx, got = anchored_gradient_free(obs, theta, lam, anchors)
                self.assertEqual(list(idx), free)
                want = ref.fd_anchored_gradient(
                    x, records_from(obs, n), n, lam, anchors)
                scale = max(1.0, max(abs(v) for v in want))
                for a, b in zip(got, want):
                    self.assertAlmostEqual(a, b, delta=1e-6 * scale)

    def test_hessian_free_block_central_differences(self):
        for n, anchors, lam, obs in self.CASES:
            with self.subTest(anchors=anchors):
                free = [i for i in range(n) if i not in anchors]
                x = [0.15 * (k + 1) * (1 if k % 2 else -1)
                     for k in range(len(free))]
                theta = self._theta(n, anchors, x)
                idx, got = anchored_hessian_free(obs, theta, lam, anchors)
                want = ref.fd_anchored_hessian(
                    x, records_from(obs, n), n, lam, anchors)
                scale = max(1.0, max(abs(v) for row in want for v in row))
                self.assertEqual(list(idx), free)
                for i in range(len(free)):
                    for j in range(len(free)):
                        self.assertAlmostEqual(
                            got[i][j], want[i][j], delta=1e-4 * scale)

    def test_objective_penalizes_free_coordinates_only(self):
        n, anchors, lam, obs = self.CASES[0]
        free = [i for i in range(n) if i not in anchors]
        x = [0.2, -0.4, 0.7]
        theta = ref.ref_anchored_theta(free, x, anchors, n)
        got = anchored_penalized_objective(obs, theta, lam, anchors)
        want = ref.ref_anchored_objective(
            x, records_from(obs, n), n, lam, anchors)
        self.assertAlmostEqual(got, want, places=12)


class ActiveBoundTests(unittest.TestCase):
    def test_solver_actually_hits_upper_bound(self):
        obs = [Observation((0,))] * 10
        r = fit_anchored(obs, 2, {1: 0.0}, lam=0.05,
                         bounds={0: (-2.0, 0.25)})
        self.assertTrue(r.converged, r.message)
        self.assertAlmostEqual(r.theta[0], 0.25, places=10)
        free = fit_anchored(obs, 2, {1: 0.0}, lam=0.05)
        self.assertGreater(free.theta[0], 0.25)
        _, g = anchored_gradient_free(obs, list(r.theta), 0.05, {1: 0.0})
        self.assertLess(g[0], 0.0)  # upper-bound KKT: gradient points in
        self.assertLessEqual(r.boundary_kkt_residual, 1e-10)

    def test_solver_actually_hits_lower_bound(self):
        obs = [Observation((1,))] * 10
        r = fit_anchored(obs, 2, {1: 0.0}, lam=0.05,
                         bounds={0: (-0.25, 2.0)})
        self.assertTrue(r.converged, r.message)
        self.assertAlmostEqual(r.theta[0], -0.25, places=10)
        _, g = anchored_gradient_free(obs, list(r.theta), 0.05, {1: 0.0})
        self.assertGreater(g[0], 0.0)  # lower-bound KKT: gradient points out
        self.assertLessEqual(r.boundary_kkt_residual, 1e-10)

    def test_backtracking_never_moves_anchors(self):
        obs = [Observation((0,), participants=(0, 1, 2)),
               Observation((2,), participants=(0, 1, 2))] * 5
        anchors = {1: 0.66}
        seen = []
        original = model._anchored_objective

        def spy(normalized, theta, lam, mask):
            value = original(normalized, theta, lam, mask)
            seen.append((tuple(theta), value))
            return value

        model._anchored_objective = spy
        try:
            r = fit_anchored(
                obs, 3, anchors, lam=0.3,
                bounds={0: (-3.0, 0.1), 2: (-1.0, 1.0)})
        finally:
            model._anchored_objective = original
        self.assertTrue(r.converged, r.message)
        self.assertGreater(len(seen), 1)
        for theta_eval, value in seen:
            self.assertEqual(theta_eval[1], 0.66)
            self.assertTrue(-3.0 - 1e-12 <= theta_eval[0] <= 0.1 + 1e-12)
            self.assertTrue(-1.0 - 1e-12 <= theta_eval[2] <= 1.0 + 1e-12)
            self.assertTrue(math.isfinite(value))


class AnchorValidationTests(unittest.TestCase):
    def test_bad_anchors_rejected_before_optimization(self):
        obs = [Observation((0,))]
        cases = [
            {0: float("nan")},
            {0: float("inf")},
            {3: 1.0},
            {True: 1.0},
            [(0, 1.0), (0, 2.0)],
            42,
        ]
        for bad in cases:
            with self.subTest(bad=bad):
                with self.assertRaises(PLInputError):
                    fit_anchored(obs, 2, bad)

    def test_empty_anchor_rejected_from_fit_anchored(self):
        with self.assertRaises(PLInputError) as cm:
            fit_anchored([Observation((0,))], 2, {})
        self.assertIn("at least one anchored", str(cm.exception))

    def test_legacy_path_without_anchors_unchanged(self):
        obs = [Observation((0,))] * 4
        r = fit_plackett_luce(obs, 2, fixed_theta={}, lam=0.5)
        self.assertTrue(r.converged)
        self.assertAlmostEqual(math.fsum(r.theta), 0.0, delta=1e-12)
        self.assertEqual(r.boundary_kkt_residual, 0.0)

    def test_bounds_require_anchors(self):
        with self.assertRaises(PLInputError) as cm:
            fit_plackett_luce([Observation((0,))], 2, bounds={0: (-1.0, 1.0)})
        self.assertIn("fixed_theta", str(cm.exception))

    def test_bounds_validation(self):
        obs = [Observation((0,))]
        bad_bounds = [
            {1: (-1.0, 1.0)},           # bound on a fixed object
            {0: (1.0, -1.0)},           # reversed
            {0: (float("-inf"), 1.0)},  # non-finite
        ]
        for bad in bad_bounds[:1]:
            with self.assertRaises(PLInputError):
                fit_anchored(obs, 2, {1: 0.0}, bounds=bad)
        for bad in bad_bounds[1:]:
            with self.assertRaises(PLInputError):
                fit_anchored(obs, 2, {1: 0.0}, bounds=bad)

    def test_initial_must_match_anchor_and_box(self):
        obs = [Observation((0,))]
        with self.assertRaises(PLInputError) as cm:
            fit_anchored(obs, 2, {1: 0.3}, initial=(0.0, 0.4))
        self.assertIn("not recentered", str(cm.exception))
        with self.assertRaises(PLInputError):
            fit_anchored(obs, 2, {1: 0.3}, initial=(2.0, 0.3),
                         bounds={0: (-1.0, 1.0)})

    def test_no_partial_order_likelihood(self):
        # The anchored path is staged Plackett-Luce only: swapping the order
        # of the ranked prefix changes the log probability (no equivalence
        # classes / partial orders).
        th = [0.2, -0.3, 0.1]
        a = ranking_log_probability((0, 1), th, participants=(0, 1, 2))
        b = ranking_log_probability((1, 0), th, participants=(0, 1, 2))
        self.assertNotAlmostEqual(a, b)


if __name__ == "__main__":
    unittest.main()
