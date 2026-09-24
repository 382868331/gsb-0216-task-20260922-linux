"""Mathematical correctness tests for constrained-order likelihood.

Expected values come from:
  * tests/reference.py -- independent itertools.permutations enumeration plus
    an independently coded staged Plackett-Luce probability (never imports the
    package);
  * hand-computed rational products for tiny candidate sets;
  * analytic identities (empty constraints partition the sample space; a
    total chain equals one full order; disjoint-event additivity).
"""
import itertools
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plackett_luce import (  # noqa: E402
    OrderConstraint,
    constrained_order_probability,
    ranking_log_probability,
    ranking_probability,
)
import reference as ref  # noqa: E402

C = OrderConstraint


def edges_from_pairs(pairs):
    return [C(a, b) for a, b in pairs]


class EmptyConstraintTests(unittest.TestCase):
    """With no constraints every full order is legal: m! extensions sum to 1."""

    def test_counts_are_factorials(self):
        theta_by_n = {
            1: (0.0,),
            2: (0.3, -0.3),
            3: (0.5, 0.2, -0.7),
            4: (0.4, -0.1, 0.25, -0.55),
            5: (0.9, 0.35, -0.1, -0.45, -0.7),
            6: (0.3, -0.1, 0.25, -0.55, 0.4, -0.3),
        }
        for n, theta in theta_by_n.items():
            with self.subTest(n=n):
                self.assertAlmostEqual(math.fsum(theta), 0.0)
                r = constrained_order_probability(
                    None, tuple(range(n)), theta)
                self.assertEqual(r.n_extensions, math.factorial(n))
                self.assertAlmostEqual(r.likelihood, 1.0, places=12)
                self.assertAlmostEqual(r.log_likelihood, 0.0, places=12)

    def test_independent_reference_agrees_empty(self):
        theta = (0.5, 0.2, -0.7)
        r = constrained_order_probability((), (0, 1, 2), theta)
        want = ref.ref_constrained_likelihood((), (0, 1, 2), theta)
        self.assertAlmostEqual(r.likelihood, want, places=14)


class ChainTests(unittest.TestCase):
    """A total chain fixes the whole order: exactly one extension."""

    def test_three_chain_is_one_full_order(self):
        theta = (0.5, 0.2, -0.7)
        r = constrained_order_probability(
            [C(0, 1), C(1, 2)], (0, 1, 2), theta)
        self.assertEqual(r.n_extensions, 1)
        want = ranking_probability((0, 1, 2), theta,
                                   participants=(0, 1, 2))
        self.assertAlmostEqual(r.likelihood, want, places=14)

    def test_chain_need_not_list_every_edge(self):
        # Transitivity: 0<1 and 1<2 already force 0<2; adding 0<2 changes
        # neither the extension count nor the likelihood.
        theta = (0.5, 0.2, -0.7)
        r1 = constrained_order_probability(
            [C(0, 1), C(1, 2)], (0, 1, 2), theta)
        r2 = constrained_order_probability(
            [C(0, 1), C(1, 2), C(0, 2)], (0, 1, 2), theta)
        self.assertEqual(r1.n_extensions, r2.n_extensions, 1)
        self.assertAlmostEqual(r1.likelihood, r2.likelihood, places=14)

    def test_hand_computed_six_for_equal_abilities(self):
        # All abilities equal -> every full order has probability 1/3!.
        theta = (0.0, 0.0, 0.0)
        r = constrained_order_probability(
            [C(0, 1), C(1, 2)], (0, 1, 2), theta)
        self.assertEqual(r.n_extensions, 1)
        self.assertAlmostEqual(r.likelihood, 1.0 / 6.0, places=14)

    def test_hand_computed_two_candidate_chain(self):
        # theta = (a, -a): P(0 ahead of 1) = e^a / (e^a + e^{-a}) = sigmoid(2a).
        a = 0.6
        theta = (a, -a)
        r = constrained_order_probability([C(0, 1)], (0, 1), theta)
        self.assertEqual(r.n_extensions, 1)
        self.assertAlmostEqual(
            r.likelihood, 1.0 / (1.0 + math.exp(-2.0 * a)), places=14)
        # The reverse chain is the complementary event.
        r_rev = constrained_order_probability([C(1, 0)], (0, 1), theta)
        self.assertAlmostEqual(r.likelihood + r_rev.likelihood, 1.0,
                               places=14)


class SingleEdgeTests(unittest.TestCase):
    def test_three_candidates_one_edge_count(self):
        theta = (0.5, 0.2, -0.7)
        r = constrained_order_probability(
            [C(0, 1)], (0, 1, 2), theta, return_extensions=True)
        want_ext = ref.legal_extensions([(0, 1)], (0, 1, 2))
        self.assertEqual(r.n_extensions, 3)
        self.assertEqual(tuple(sorted(r.extensions)), want_ext)

    def test_hand_computed_single_edge_three_equal(self):
        # Equal strengths: each of the 3! orders equally likely; exactly half
        # have 0 ahead of 1 -> 3/6 = 1/2.
        theta = (0.0, 0.0, 0.0)
        r = constrained_order_probability([C(0, 1)], (0, 1, 2), theta)
        self.assertEqual(r.n_extensions, 3)
        self.assertAlmostEqual(r.likelihood, 0.5, places=14)

    def test_single_edge_matches_independent_reference(self):
        # Pairwise marginal order under Plackett-Luce is logistic of the gap
        # (Luce axiom): gap 3.4 -> sigmoid(3.4) ~= 0.968 > 0.95.
        theta = (-1.5, -0.4, 1.9)
        r = constrained_order_probability(
            [C(2, 0)], (0, 1, 2), theta)
        want = ref.ref_constrained_likelihood([(2, 0)], (0, 1, 2), theta)
        self.assertAlmostEqual(r.likelihood, want, places=13)
        self.assertGreater(r.likelihood, 0.95)
        self.assertAlmostEqual(
            r.likelihood, 1.0 / (1.0 + math.exp(-3.4)), places=10)

    def test_complementary_events_sum_to_one(self):
        # "0 before 1" and "1 before 0" with a third free candidate partition
        # the sample space.
        theta = (0.5, 0.2, -0.7)
        p = constrained_order_probability(
            [C(0, 1)], (0, 1, 2), theta).likelihood
        q = constrained_order_probability(
            [C(1, 0)], (0, 1, 2), theta).likelihood
        self.assertAlmostEqual(p + q, 1.0, places=12)


class BruteForceCrossCheckTests(unittest.TestCase):
    """Exhaustive cross-check against the independent permutation reference."""

    # Fixed (non-random) theta vectors and edge sets, small enough to hand-audit.
    THETAS = [
        (0.0, 0.0, 0.0),
        (1.2, -0.3, -0.9),
        (0.5, 0.2, -0.7),
        (0.4, -0.1, 0.25, -0.55),
        (-0.8, 1.1, 0.2, -0.5),
        (0.9, 0.35, -0.1, -0.45, -0.7),
        (0.3, -0.1, 0.25, -0.55, 0.4, -0.3),
    ]

    def test_all_edge_subsets_cross_check(self):
        # Exhaustive over every subset of possible edges for n <= 4
        # (at most 2^6 = 64 subsets); deterministic fixed stride sample for
        # n = 5, 6 (2^10 / 2^15 subsets would be too slow).
        for theta in self.THETAS:
            n = len(theta)
            with self.subTest(n=n):
                self.assertAlmostEqual(math.fsum(theta), 0.0)
                cands = tuple(range(n))
                pairs = [(i, j) for i in cands for j in cands if i < j]
                n_pairs = len(pairs)
                if n <= 4:
                    masks = range(1 << n_pairs)
                else:
                    masks = range(1, 1 << n_pairs,
                                  1 << max(1, n_pairs - 6))
                for mask in masks:
                    chosen = [pairs[k] for k in range(n_pairs)
                              if (mask >> k) & 1]
                    try:
                        r = constrained_order_probability(
                            edges_from_pairs(chosen), cands, theta)
                    except Exception as exc:  # noqa: BLE001
                        self.fail(
                            f"unexpected error for edges {chosen}: {exc!r}")
                    want_ext = ref.legal_extensions(chosen, cands)
                    self.assertEqual(
                        r.n_extensions, len(want_ext),
                        msg=f"count mismatch edges={chosen}")
                    want_p = ref.ref_constrained_likelihood(chosen, cands, theta)
                    self.assertAlmostEqual(
                        r.likelihood, want_p, places=11,
                        msg=f"likelihood mismatch edges={chosen}")

    def test_non_sorted_candidate_subsets(self):
        theta = (0.4, -0.1, 0.25, -0.55)
        for cands in ((3, 0, 1), (2, 0), (1, 3, 2, 0), (0, 3, 2)):
            pairs = [(a, b) for a in cands for b in cands if a < b]
            for chosen in (pairs[::3], pairs[:2], (), pairs):
                edges = [(a, b) for a, b in chosen]
                r = constrained_order_probability(
                    edges_from_pairs(edges), cands, theta)
                want_ext = ref.legal_extensions(edges, cands)
                want_p = ref.ref_constrained_likelihood(edges, cands, theta)
                self.assertEqual(r.n_extensions, len(want_ext))
                self.assertAlmostEqual(r.likelihood, want_p, places=12)


class DisjointEventAdditivityTests(unittest.TestCase):
    def test_partial_orders_partitioned_by_one_edge(self):
        # The event "0 before 1" (3 extensions on 3 candidates) is the disjoint
        # union of "0<1 and 0<2" style refinements checked explicitly: build
        # two disjoint refinements whose union is exactly 0<1.
        theta = (0.5, 0.2, -0.7)
        # Refinement A: 0<1 and 2<0 -> order must be (2,0,1), 1 extension.
        a = constrained_order_probability(
            [C(0, 1), C(2, 0)], (0, 1, 2), theta).likelihood
        # Refinement B: 0<1 and 0<2 -> extensions (0,1,2),(0,2,1), 2 ext.
        b = constrained_order_probability(
            [C(0, 1), C(0, 2)], (0, 1, 2), theta).likelihood
        total = constrained_order_probability(
            [C(0, 1)], (0, 1, 2), theta).likelihood
        self.assertAlmostEqual(a + b, total, places=12)

    def test_full_order_probabilities_sum_to_constrained_event(self):
        theta = (0.4, -0.1, 0.25, -0.55)
        edges = [C(0, 3)]
        r = constrained_order_probability(
            edges, (0, 1, 2, 3), theta, return_extensions=True)
        direct = math.fsum(
            ranking_probability(ext, theta, participants=(0, 1, 2, 3))
            for ext in r.extensions)
        self.assertEqual(r.n_extensions, 12)  # half of 4!
        self.assertAlmostEqual(direct, r.likelihood, places=13)


class ExtensionEnumerationTests(unittest.TestCase):
    def test_enumerated_extensions_are_distinct_full_orders(self):
        theta = (0.5, 0.2, -0.7)
        r = constrained_order_probability(
            [C(0, 2)], (0, 1, 2), theta, return_extensions=True)
        ext = r.extensions
        self.assertEqual(len(ext), len(set(ext)))
        for e in ext:
            self.assertEqual(sorted(e), [0, 1, 2])

    def test_extensions_satisfy_edges(self):
        theta = (0.4, -0.1, 0.25, -0.55)
        pairs = [(0, 3), (1, 2)]
        r = constrained_order_probability(
            edges_from_pairs(pairs), (0, 1, 2, 3), theta,
            return_extensions=True)
        for e in r.extensions:
            self.assertLess(e.index(0), e.index(3))
            self.assertLess(e.index(1), e.index(2))

    def test_v_poset_count_six_of_24(self):
        # 0<2 and 1<2 among 4 candidates: object 2 must come after both 0 and
        # 1; among {0,1,2} alone 2 is last (2 of 6 orders), object 3 free
        # (factor 4 insertions) -> 8 extensions.
        theta = (0.4, -0.1, 0.25, -0.55)
        r = constrained_order_probability(
            [C(0, 2), C(1, 2)], (0, 1, 2, 3), theta)
        self.assertEqual(r.n_extensions, 8)


if __name__ == "__main__":
    unittest.main()
