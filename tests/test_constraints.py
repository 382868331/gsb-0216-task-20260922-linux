"""Tests for exact precedence-constraint (poset) ranking likelihood.

Expected values come from hand calculations (sigmoid of a gap, n! and
4!/(2!2!) extension counts) and from an INDEPENDENT brute-force enumeration:
itertools.permutations filtered by the stated pairs, scored with the
independent staged log probability in tests/reference.py. The package's own
enumeration is never used to produce an expected value.
"""
import itertools
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plackett_luce import (  # noqa: E402
    MAX_CANDIDATES,
    ConstrainedLikelihood,
    OrderingConstraint as OC,
    PLInputError,
    constrained_ranking_probability,
    ranking_log_probability,
)
import reference as ref  # noqa: E402


def brute_force(pairs, candidates, theta):
    """Independent reference: every permutation satisfying the precedence pairs.

    Returns (probability, log_probability, n_extensions). Scoring uses the
    independent ref.ref_log_prob, never the package under test.
    """
    required = set(pairs)
    terms = []
    for perm in itertools.permutations(candidates):
        position = {obj: j for j, obj in enumerate(perm)}
        if all(position[b] < position[a] for b, a in required):
            terms.append(ref.ref_log_prob(perm, tuple(candidates), list(theta)))
    m = max(terms)
    lp = m + math.log(math.fsum(math.exp(v - m) for v in terms))
    return math.exp(lp), lp, len(terms)


class EmptyAndChainTests(unittest.TestCase):
    def test_empty_constraints_sum_to_one(self):
        # No constraints: all n! full rankings are legal and their PL
        # probabilities partition the event space, so the likelihood is 1.
        for n in range(1, MAX_CANDIDATES + 1):
            theta = [0.1 * i - 0.05 * (n - 1) for i in range(n)]
            r = constrained_ranking_probability([], theta)
            self.assertEqual(r.n_extensions, math.factorial(n))
            self.assertAlmostEqual(r.probability, 1.0, places=12)
            self.assertAlmostEqual(r.log_probability, 0.0, places=12)

    def test_total_order_chain_is_one_extension(self):
        theta = [0.5, -0.2, -0.3]
        cons = [OC(0, 1), OC(1, 2)]
        r = constrained_ranking_probability(cons, theta)
        self.assertEqual(r.n_extensions, 1)
        # The unique extension is exactly the full ranking 0 > 1 > 2.
        self.assertAlmostEqual(
            r.log_probability,
            ranking_log_probability((0, 1, 2), theta), places=12)

    def test_transitive_edge_is_redundant(self):
        theta = [0.5, -0.2, -0.3]
        r = constrained_ranking_probability([OC(0, 1), OC(1, 2), OC(0, 2)],
                                            theta)
        self.assertEqual(r.n_extensions, 1)
        self.assertAlmostEqual(
            r.log_probability,
            ranking_log_probability((0, 1, 2), theta), places=12)
        # Redundant duplicate/transitive pairs are materialized once.
        self.assertEqual(r.constraints, ((0, 1), (0, 2), (1, 2)))

    def test_single_candidate(self):
        r = constrained_ranking_probability([], [0.0])
        self.assertEqual(r.n_extensions, 1)
        self.assertAlmostEqual(r.probability, 1.0)
        self.assertEqual(r.candidate_set, (0,))


class HandComputedTests(unittest.TestCase):
    def test_two_candidate_pair_is_sigmoid_of_gap(self):
        # theta = (a, -a): P(0 before 1) = e^a / (e^a + e^-a) = sigmoid(2a).
        for a in (-1.3, 0.0, 0.4, 1.7):
            theta = [a, -a]
            r = constrained_ranking_probability([OC(0, 1)], theta)
            self.assertEqual(r.n_extensions, 1)
            hand = 1.0 / (1.0 + math.exp(-2.0 * a))
            self.assertAlmostEqual(r.probability, hand, places=12)
            self.assertAlmostEqual(math.exp(r.log_probability), hand, places=12)

    def test_complementary_pair_events_partition(self):
        theta = [0.6, -0.1, -0.5]
        p_01 = constrained_ranking_probability([OC(0, 1)], theta).probability
        p_10 = constrained_ranking_probability([OC(1, 0)], theta).probability
        self.assertAlmostEqual(p_01 + p_10, 1.0, places=12)

    def test_extension_counts_small_posets(self):
        # 0 before 1 only, three candidates: legal perms 012, 021, 201 -> 3.
        r = constrained_ranking_probability([OC(0, 1)], [0.0, 0.0, 0.0])
        self.assertEqual(r.n_extensions, 3)
        # Two disjoint precedence pairs on four objects interleave freely:
        # 4! / (2! * 2!) = 6 linear extensions.
        r4 = constrained_ranking_probability(
            [OC(0, 1), OC(2, 3)], [0.0] * 4)
        self.assertEqual(r4.n_extensions, 6)
        # A "V": 0 before both 1 and 2 forces 0 first: 012, 021 -> 2.
        rv = constrained_ranking_probability(
            [OC(0, 1), OC(0, 2)], [0.0] * 3)
        self.assertEqual(rv.n_extensions, 2)

    def test_uniform_abilities_likelihood_is_count_over_factorial(self):
        # All abilities equal: every full permutation has probability 1/n!.
        n = 4
        cons = [OC(0, 1), OC(2, 3)]
        r = constrained_ranking_probability(cons, [0.0] * n)
        self.assertAlmostEqual(r.probability, r.n_extensions / math.factorial(n))
        self.assertAlmostEqual(r.log_probability,
                               math.log(r.n_extensions) - math.lgamma(n + 1))

    def test_restricting_constraints_decreases_likelihood(self):
        theta = [0.7, 0.1, -0.8]
        p_none = constrained_ranking_probability([], theta).probability
        p_one = constrained_ranking_probability([OC(0, 1)], theta).probability
        p_chain = constrained_ranking_probability(
            [OC(0, 1), OC(1, 2)], theta).probability
        self.assertAlmostEqual(p_none, 1.0, places=12)
        self.assertLess(p_chain, p_one)
        self.assertLess(p_one, p_none)


class BruteForceReferenceTests(unittest.TestCase):
    CASES = [
        # (n, candidate_set, pairs, theta)
        (3, (0, 1, 2), ((0, 1),), [0.4, -0.1, -0.3]),
        (3, (0, 1, 2), ((0, 1), (0, 2)), [1.1, -0.4, -0.7]),
        (4, (0, 1, 2, 3), ((0, 1), (2, 3)), [0.6, -0.5, 0.2, -0.3]),
        (4, (0, 1, 2, 3), ((3, 0), (0, 2), (2, 1)), [-0.2, 0.9, 0.3, -1.0]),
        (5, (0, 1, 2, 3, 4), ((4, 0), (1, 2)),
         [0.5, -0.5, 0.2, -0.1, -0.1]),
        (6, tuple(range(6)), ((0, 1), (1, 2), (3, 4)),
         [0.3 * i - 0.75 for i in range(6)]),
    ]

    def test_matches_independent_enumeration(self):
        for case in self.CASES:
            n, cand, pairs, theta = case
            with self.subTest(cand=cand, pairs=pairs):
                cons = [OC(b, a) for b, a in pairs]
                got = constrained_ranking_probability(cons, theta, cand)
                p, lp, count = brute_force(pairs, list(cand), theta)
                self.assertEqual(got.n_extensions, count)
                self.assertAlmostEqual(got.probability, p, places=12)
                self.assertAlmostEqual(got.log_probability, lp, places=12)
                self.assertGreaterEqual(got.probability, 0.0)
                self.assertLessEqual(got.probability, 1.0 + 1e-12)
                self.assertEqual(got.candidate_set, cand)
                self.assertEqual(got.constraints, tuple(sorted(pairs)))

    def test_subset_candidates_excludes_others_from_denominators(self):
        # Object 2 is extreme but absent from the candidate set: the pair
        # probability over {0, 1} depends only on the gap theta0 - theta1.
        cons = [OC(0, 1)]
        r1 = constrained_ranking_probability(cons, [0.4, -0.4, 0.0],
                                             candidate_set=(0, 1))
        r2 = constrained_ranking_probability(cons, [0.6, -0.2, -0.4],
                                             candidate_set=(0, 1))
        self.assertEqual(r1.n_extensions, 1)
        self.assertAlmostEqual(r1.log_probability, r2.log_probability, 12)
        hand = 1.0 / (1.0 + math.exp(-0.8))
        self.assertAlmostEqual(r1.probability, hand, places=12)
        # Independent brute force on the subset.
        p, _, count = brute_force(((0, 1),), [0, 1], [0.4, -0.4, 0.0])
        self.assertEqual(count, 1)
        self.assertAlmostEqual(r1.probability, p, places=12)

    def test_six_candidate_enumeration_bound(self):
        theta = [0.2 * i - 0.5 for i in range(MAX_CANDIDATES)]
        r = constrained_ranking_probability([], theta)
        self.assertEqual(r.n_extensions, math.factorial(MAX_CANDIDATES))
        self.assertAlmostEqual(r.probability, 1.0, places=12)
        chain = [OC(i, i + 1) for i in range(MAX_CANDIDATES - 1)]
        rc = constrained_ranking_probability(chain, theta)
        self.assertEqual(rc.n_extensions, 1)
        self.assertAlmostEqual(
            rc.log_probability,
            ranking_log_probability(tuple(range(MAX_CANDIDATES)), theta),
            places=12)

    def test_deterministic_and_does_not_mutate_inputs(self):
        theta = [0.3, -0.3, 0.0]
        cons = [OC(0, 1)]
        r1 = constrained_ranking_probability(cons, theta)
        r2 = constrained_ranking_probability(cons, theta)
        self.assertEqual(r1, r2)
        self.assertEqual(theta, [0.3, -0.3, 0.0])
        self.assertEqual(cons, [OC(0, 1)])
        self.assertIsInstance(r1, ConstrainedLikelihood)


class CyclicAndInvalidInputTests(unittest.TestCase):
    def test_two_cycle_rejected_before_enumeration(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_ranking_probability([OC(0, 1), OC(1, 0)],
                                            [0.0] * 3)
        self.assertIn("cycle", str(cm.exception))

    def test_three_cycle_rejected(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_ranking_probability(
                [OC(0, 1), OC(1, 2), OC(2, 0)], [0.0] * 3)
        self.assertIn("cycle", str(cm.exception))

    def test_self_loop_rejected(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_ranking_probability([OC(1, 1)], [0.0] * 3)
        self.assertIn("cycle", str(cm.exception))

    def test_contradictory_call_leaves_no_state(self):
        # The API is purely functional: after a rejected (cyclic) call a valid
        # call on the same interpreter must still work normally.
        try:
            constrained_ranking_probability([OC(0, 0)], [0.0] * 2)
        except PLInputError:
            pass
        r = constrained_ranking_probability([OC(0, 1)], [0.5, -0.5])
        self.assertAlmostEqual(r.probability,
                               1.0 / (1.0 + math.exp(-1.0)))

    def test_candidate_set_too_large(self):
        theta = [0.0] * (MAX_CANDIDATES + 1)
        with self.assertRaises(PLInputError) as cm:
            constrained_ranking_probability([], theta)
        self.assertIn("candidate_set", str(cm.exception))
        self.assertIn(str(MAX_CANDIDATES), str(cm.exception))

    def test_empty_candidate_set_rejected(self):
        with self.assertRaises(PLInputError):
            constrained_ranking_probability([], [0.0, 0.0],
                                            candidate_set=())

    def test_duplicate_candidate_rejected(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_ranking_probability([], [0.0] * 3,
                                            candidate_set=(0, 1, 1))
        self.assertIn("candidate_set", str(cm.exception))

    def test_endpoint_must_be_candidate(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_ranking_probability(
                [OC(0, 2)], [0.0] * 3, candidate_set=(0, 1))
        self.assertIn("not in the candidate set", str(cm.exception))

    def test_index_out_of_range_located(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_ranking_probability([OC(0, 3)], [0.0] * 3)
        self.assertIn("constraints[0].after", str(cm.exception))

    def test_wrong_constraint_element_type(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_ranking_probability([(0, 1)], [0.0] * 3)
        self.assertIn("constraints[0]", str(cm.exception))

    def test_bool_and_non_int_endpoints_rejected(self):
        with self.assertRaises(PLInputError):
            constrained_ranking_probability([OC(True, 1)], [0.0] * 3)
        with self.assertRaises(PLInputError):
            constrained_ranking_probability([OC(0, 1.0)], [0.0] * 3)

    def test_non_iterable_constraints_rejected(self):
        with self.assertRaises(PLInputError):
            constrained_ranking_probability(OC(0, 1), [0.0] * 2)

    def test_theta_validation_shared_with_existing_api(self):
        with self.assertRaises(PLInputError):
            constrained_ranking_probability([], [float("nan"), 0.0])
        with self.assertRaises(PLInputError):
            constrained_ranking_probability([], [1.0, 0.0])  # sum != 0
        with self.assertRaises(PLInputError):
            constrained_ranking_probability([], [])

    def test_generator_constraints_accepted(self):
        def gen():
            yield OC(0, 1)
            yield OC(1, 2)

        r = constrained_ranking_probability(gen(), [0.0] * 3)
        self.assertEqual(r.n_extensions, 1)


if __name__ == "__main__":
    unittest.main()
