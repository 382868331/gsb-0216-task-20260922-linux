"""Anchored (fixed-theta) fitting tests.

Expected values come from the independent reference in tests/reference.py
(bisection roots, independently coded logsumexp, central finite differences
of the free coordinates) -- never from the package's own core.
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
    fit_plackett_luce,
    ranking_probability,
)
import reference as ref  # noqa: E402


def records_from(obs, n):
    out = []
    for ob in obs:
        part = tuple(range(n)) if ob.participants is None else tuple(ob.participants)
        out.append((tuple(ob.ranking), part))
    return out


class StrongAnchorTests(unittest.TestCase):
    def test_anchor_held_exactly_and_not_recentered(self):
        # Object 0 pinned far above zero; the data agree it is strongest.
        obs = [Observation((0, 1, 2)) for _ in range(20)]
        r = fit_plackett_luce(obs, 3, lam=0.5, anchors={0: 4.0})
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED, r.message)
        self.assertEqual(r.theta[0], 4.0)  # exact, never Newton-updated
        # The sum-to-zero gauge is gone: no re-centering of anything.
        self.assertNotAlmostEqual(math.fsum(r.theta), 0.0, places=6)
        # Free objects order themselves below the anchor.
        self.assertLess(r.theta[1], 4.0)
        self.assertGreater(r.theta[1], r.theta[2])

    def test_anchor_forms_iterable_of_pairs(self):
        obs = [Observation((0, 1)) for _ in range(5)]
        r = fit_plackett_luce(obs, 2, lam=1.0, anchors=[(0, 1.5)])
        self.assertEqual(r.theta[0], 1.5)

    def test_initial_need_not_sum_to_zero_in_anchored_mode(self):
        obs = [Observation((1, 0)) for _ in range(4)]
        r = fit_plackett_luce(obs, 2, lam=1.0, anchors={0: 0.25},
                              initial=[9.0, -3.0])
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED, r.message)
        self.assertEqual(r.theta[0], 0.25)


class TwoObjectAnchoredAnalyticTests(unittest.TestCase):
    def test_free_ability_matches_bisection_root(self):
        m, lam, a = 7, 0.5, 1.25
        obs = [Observation((1, 0)) for _ in range(m)]  # free object 1 wins
        r = fit_plackett_luce(obs, 2, lam=lam, anchors={0: a})
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED, r.message)
        t = ref.anchored_two_object_root(m, lam, a)
        self.assertAlmostEqual(r.theta[1], t, places=10)
        self.assertEqual(r.theta[0], a)

    def test_two_object_probability_closed_form(self):
        # P(1 beats 0) = exp(t1) / (exp(t1) + exp(a)), analytically.
        # (public probability helpers keep the legacy sum-to-zero contract)
        a, t1 = 0.75, -0.75
        p = ranking_probability((1, 0), [a, t1])
        expected = math.exp(t1) / (math.exp(t1) + math.exp(a))
        self.assertAlmostEqual(p, expected, places=15)

    def test_anchored_gradient_matches_central_differences(self):
        # FD check of the free-coordinate gradient at the fitted point: the
        # solver must sit at a stationary point of the independent objective.
        lam, a = 0.8, 2.0
        obs = ([Observation((1, 2, 0)) for _ in range(6)]
               + [Observation((2, 1), participants=(1, 2)) for _ in range(3)])
        n = 3
        anchors = {0: a}
        free_idx = [1, 2]
        r = fit_plackett_luce(obs, n, lam=lam, anchors=anchors)
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED, r.message)
        x = [r.theta[i] for i in free_idx]
        fdg = ref.fd_gradient_anchored(x, records_from(obs, n), n, lam,
                                       free_idx, anchors)
        self.assertLess(max(abs(v) for v in fdg), 1e-5)
        self.assertLessEqual(r.gradient_norm, 1e-10)

    def test_backtracking_never_touches_anchor(self):
        # Wildly wrong free start: many backtracked steps, anchor bit-exact.
        obs = [Observation((1, 0)) for _ in range(30)]
        r = fit_plackett_luce(obs, 2, lam=0.05, anchors={0: -3.0},
                              initial=[-3.0, 40.0])
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED, r.message)
        self.assertEqual(r.theta[0], -3.0)


class FreeObjectStructureTests(unittest.TestCase):
    def test_object_in_no_sample_sits_at_zero(self):
        # Object 2 never appears in any sample (explicit participant sets, so
        # it is not an unranked denominator participant either); its objective
        # is lambda/2 * t^2 alone, minimized at exactly 0.
        obs = [Observation((0, 1), participants=(0, 1)) for _ in range(9)]
        r = fit_plackett_luce(obs, 3, lam=1.0, anchors={0: 1.0})
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED, r.message)
        self.assertAlmostEqual(r.theta[2], 0.0, places=12)

    def test_partial_participant_sets_keep_anchor_in_denominators(self):
        # Anchored object 0 participates (unranked) in obs A, so it must sit
        # in every denominator there; it is absent from obs B entirely.
        obs = ([Observation((1,), participants=(0, 1, 2)) for _ in range(4)]
               + [Observation((2, 1), participants=(1, 2)) for _ in range(4)])
        n, lam = 3, 0.7
        anchors = {0: 1.1}
        r = fit_plackett_luce(obs, n, lam=lam, anchors=anchors)
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED, r.message)
        records = records_from(obs, n)
        theta = list(r.theta)
        # Per-observation contributions match the independent reference,
        # including the anchored object inside the participant denominators.
        for (ranking, part), contrib in zip(records, r.log_contributions):
            self.assertAlmostEqual(
                contrib, ref.ref_log_prob(ranking, part, theta), places=12)
        ref_ll = math.fsum(ref.ref_log_prob(rank, part, theta)
                           for rank, part in records)
        self.assertAlmostEqual(r.log_likelihood, ref_ll, places=10)
        # Sanity: removing the anchor from obs A's participants would change
        # the likelihood -- proof the anchor really shapes the denominators.
        obs_no_anchor = ([Observation((1,), participants=(1, 2))
                          for _ in range(4)]
                         + [Observation((2, 1), participants=(1, 2))
                            for _ in range(4)])
        ll_without = math.fsum(
            ref.ref_log_prob(rank, part, theta)
            for rank, part in records_from(obs_no_anchor, n))
        self.assertNotAlmostEqual(r.log_likelihood, ll_without, places=6)


class AllAnchoredTests(unittest.TestCase):
    def test_all_fixed_only_evaluates(self):
        obs = [Observation((0, 2)), Observation((1,), participants=(1, 2)),
               Observation(())]
        n = 3
        anchors = {0: 0.5, 1: -1.25, 2: 2.0}
        r = fit_plackett_luce(obs, n, lam=1.0, anchors=anchors)
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED)
        self.assertEqual(r.iterations, 0)
        self.assertEqual(tuple(r.theta), (0.5, -1.25, 2.0))
        theta = [anchors[i] for i in range(n)]
        records = records_from(obs, n)
        ref_ll = math.fsum(ref.ref_log_prob(rank, part, theta)
                           for rank, part in records)
        self.assertAlmostEqual(r.log_likelihood, ref_ll, places=12)
        for (ranking, part), contrib in zip(records, r.log_contributions):
            self.assertAlmostEqual(
                contrib, ref.ref_log_prob(ranking, part, theta), places=12)
        # No free coordinates: the penalty term is exactly zero.
        self.assertAlmostEqual(r.penalized_objective, -ref_ll, places=12)


class BoundsTests(unittest.TestCase):
    def test_active_upper_bound_and_kkt_residual(self):
        # Unbounded optimum is far above hi=0.5; the bound must bind.
        m, lam, a = 7, 0.5, 0.0
        obs = [Observation((1, 0)) for _ in range(m)]
        unbounded = fit_plackett_luce(obs, 2, lam=lam, anchors={0: a})
        root = ref.anchored_two_object_root(m, lam, a)
        self.assertGreater(root, 0.5)
        self.assertAlmostEqual(unbounded.theta[1], root, places=10)

        r = fit_plackett_luce(obs, 2, lam=lam, anchors={0: a},
                              bounds={1: (-5.0, 0.5)})
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED, r.message)
        self.assertEqual(r.theta[1], 0.5)  # exactly on the bound, not clipped
        self.assertEqual(r.theta[0], a)
        self.assertIsNotNone(r.boundary_kkt_residual)
        self.assertLessEqual(r.boundary_kkt_residual, 1e-10)
        # Feasible-direction check: at the upper bound the free gradient of
        # the independent objective must be <= 0 (minimum against the wall).
        grad = ref.fd_gradient_anchored([r.theta[1]], records_from(obs, 2),
                                        2, lam, [1], {0: a})
        self.assertLess(grad[0], 0.0)

    def test_inactive_bound_leaves_unconstrained_solution(self):
        m, lam, a = 7, 0.5, 0.0
        obs = [Observation((1, 0)) for _ in range(m)]
        root = ref.anchored_two_object_root(m, lam, a)
        r = fit_plackett_luce(obs, 2, lam=lam, anchors={0: a},
                              bounds={1: (-10.0, 10.0)})
        self.assertAlmostEqual(r.theta[1], root, places=10)
        self.assertLessEqual(r.boundary_kkt_residual, 1e-10)

    def test_initial_outside_bounds_rejected(self):
        obs = [Observation((1, 0))]
        with self.assertRaises(PLInputError):
            fit_plackett_luce(obs, 2, lam=1.0, anchors={0: 0.0},
                              bounds={1: (-1.0, 1.0)}, initial=[0.0, 2.0])


class AnchoredValidationTests(unittest.TestCase):
    def setUp(self):
        self.obs = [Observation((0, 1))]

    def test_duplicate_anchor_rejected(self):
        with self.assertRaises(PLInputError):
            fit_plackett_luce(self.obs, 3, anchors=[(0, 1.0), (0, 2.0)])

    def test_unknown_anchor_object_rejected(self):
        with self.assertRaises(PLInputError):
            fit_plackett_luce(self.obs, 2, anchors={5: 1.0})

    def test_non_finite_anchor_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(PLInputError):
                fit_plackett_luce(self.obs, 2, anchors={0: bad})

    def test_anchor_index_type_rejected(self):
        with self.assertRaises(PLInputError):
            fit_plackett_luce(self.obs, 2, anchors={0.5: 1.0})
        with self.assertRaises(PLInputError):
            fit_plackett_luce(self.obs, 2, anchors={True: 1.0})

    def test_bounds_require_anchor(self):
        with self.assertRaises(PLInputError):
            fit_plackett_luce(self.obs, 2, bounds={1: (-1.0, 1.0)})

    def test_bounds_on_anchored_object_rejected(self):
        with self.assertRaises(PLInputError):
            fit_plackett_luce(self.obs, 2, anchors={0: 0.0},
                              bounds={0: (-1.0, 1.0)})

    def test_inverted_or_nonfinite_bounds_rejected(self):
        with self.assertRaises(PLInputError):
            fit_plackett_luce(self.obs, 2, anchors={0: 0.0},
                              bounds={1: (2.0, -2.0)})
        with self.assertRaises(PLInputError):
            fit_plackett_luce(self.obs, 2, anchors={0: 0.0},
                              bounds={1: (0.0, float("inf"))})

    def test_no_anchor_legacy_behavior_unchanged(self):
        # Without anchors the sum-to-zero gauge and full penalty still apply.
        m, lam = 7, 0.5
        obs = [Observation((0,)) for _ in range(m)]
        r = fit_plackett_luce(obs, 2, lam=lam)
        a = ref.two_object_winner_root(m, lam)
        self.assertAlmostEqual(r.theta[0], a, places=10)
        self.assertAlmostEqual(r.theta[1], -a, places=10)
        self.assertIsNone(r.boundary_kkt_residual)


if __name__ == "__main__":
    unittest.main()
