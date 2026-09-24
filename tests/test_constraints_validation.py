"""Validation and error-semantics tests for precedence-constrained orders.

Covers the contract of ``constrained_order_probability``: candidate-set
bounds, edge validation, acyclicity, located PLInputError messages, and the
"no half-updated state" rule (the API is functional and never mutates its
inputs; a rejected call leaves everything reusable).
"""
import math
import unittest

from plackett_luce import (
    MAX_CONSTRAINED_CANDIDATES,
    OrderConstraint,
    PLInputError,
    constrained_order_probability,
)


THETA4 = (0.5, 0.2, -0.3, -0.4)  # sums to zero
C = OrderConstraint


class CandidateValidationTests(unittest.TestCase):
    def test_six_accepted_seven_rejected(self):
        theta6 = (0.3, -0.1, 0.25, -0.55, 0.4, -0.3)
        self.assertAlmostEqual(math.fsum(theta6), 0.0)
        r = constrained_order_probability(None, tuple(range(6)), theta6)
        self.assertEqual(r.n_extensions, 720)
        with self.assertRaises(PLInputError) as cm:
            constrained_order_probability(
                None, tuple(range(7)), [0.0] * 7)
        self.assertIn("candidates", str(cm.exception))
        self.assertIn(str(7), str(cm.exception))

    def test_bound_constant_is_six(self):
        self.assertEqual(MAX_CONSTRAINED_CANDIDATES, 6)

    def test_empty_candidate_set_rejected(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_order_probability(None, (), THETA4)
        self.assertIn("candidates", str(cm.exception))
        self.assertIn("at least one", str(cm.exception))

    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_order_probability(None, (0, 1, 0), THETA4)
        self.assertIn("candidates", str(cm.exception))
        self.assertIn("duplicated", str(cm.exception))

    def test_candidate_index_outside_theta(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_order_probability(None, (0, 4), THETA4)
        self.assertIn("candidates[1]", str(cm.exception))

    def test_candidate_wrong_types_rejected(self):
        for bad in ("012", 123, 3.5, [0.1], bytes()):
            with self.subTest(bad=bad):
                with self.assertRaises(PLInputError):
                    constrained_order_probability(None, bad, THETA4)

    def test_bool_candidate_rejected(self):
        with self.assertRaises(PLInputError):
            constrained_order_probability(None, (True, 1), THETA4)

    def test_theta_validation_is_reused(self):
        # NaN / non-finite / wrong length / sum != 0 are all refused exactly
        # as in the existing likelihood entry points.
        with self.assertRaises(PLInputError):
            constrained_order_probability(None, (0, 1), (float("nan"), 0.0))
        with self.assertRaises(PLInputError):
            constrained_order_probability(None, (0, 1), (1.0, 0.0))
        # theta shorter than the candidate indices it is asked about.
        with self.assertRaises(PLInputError):
            constrained_order_probability(None, (0, 1), (0.0,))
        with self.assertRaises(PLInputError):
            constrained_order_probability(None, (0,), ())

    def test_single_candidate_trivial(self):
        r = constrained_order_probability(None, (0,), (0.0,))
        self.assertEqual(r.n_extensions, 1)
        self.assertEqual(r.likelihood, 1.0)
        self.assertEqual(r.log_likelihood, 0.0)


class ConstraintValidationTests(unittest.TestCase):
    def test_direct_three_cycle_rejected(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_order_probability(
                [C(0, 1), C(1, 2), C(2, 0)], (0, 1, 2),
                (0.2, -0.1, -0.1))
        msg = str(cm.exception)
        self.assertIn("cycle", msg)
        # The offending objects are named, in cyclic order.
        self.assertIn("0", msg)
        self.assertIn("1", msg)
        self.assertIn("2", msg)

    def test_two_cycle_rejected(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_order_probability(
                [C(0, 1), C(1, 0)], (0, 1), (0.3, -0.3))
        self.assertIn("cycle", str(cm.exception))

    def test_transitive_cycle_rejected(self):
        # 0 -> 1, 1 -> 2, 0 -> 2 is fine; adding 2 -> 0 closes the loop even
        # though the contradiction is partly via transitivity.
        good = constrained_order_probability(
            [C(0, 1), C(1, 2), C(0, 2)], (0, 1, 2), (0.2, 0.0, -0.2))
        self.assertEqual(good.n_extensions, 1)
        with self.assertRaises(PLInputError) as cm:
            constrained_order_probability(
                [C(0, 1), C(1, 2), C(2, 0)], (0, 1, 2), (0.2, 0.0, -0.2))
        self.assertIn("cycle", str(cm.exception))

    def test_self_loop_rejected(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_order_probability(
                [C(1, 1)], (0, 1, 2), (0.5, 0.2, -0.7))
        self.assertIn("constraints[0]", str(cm.exception))
        self.assertIn("itself", str(cm.exception))

    def test_endpoint_not_in_candidates_located(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_order_probability(
                [C(0, 1), C(0, 3)], (0, 1, 2), THETA4)
        self.assertIn("constraints[1]", str(cm.exception))
        self.assertIn("3", str(cm.exception))
        self.assertIn("candidate set", str(cm.exception))

    def test_endpoint_index_out_of_range(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_order_probability(
                [C(0, -1)], (0, 1), THETA4)
        self.assertIn("constraints[0].after", str(cm.exception))
        with self.assertRaises(PLInputError):
            constrained_order_probability(
                [C(0, 9)], (0, 1), THETA4)

    def test_endpoint_wrong_type_and_bool(self):
        with self.assertRaises(PLInputError):
            constrained_order_probability(
                [C(0, 1.0)], (0, 1), THETA4)
        with self.assertRaises(PLInputError):
            constrained_order_probability(
                [C(False, 1)], (0, 1), THETA4)

    def test_element_must_be_order_constraint(self):
        with self.assertRaises(PLInputError) as cm:
            constrained_order_probability(
                [(0, 1)], (0, 1), THETA4)
        self.assertIn("constraints[0]", str(cm.exception))
        self.assertIn("OrderConstraint", str(cm.exception))

    def test_constraints_wrong_container_type(self):
        with self.assertRaises(PLInputError):
            constrained_order_probability(
                C(0, 1), (0, 1), THETA4)  # one edge, not an iterable of edges

    def test_none_and_empty_both_accepted(self):
        r1 = constrained_order_probability(None, (0, 1, 2), THETA4)
        r2 = constrained_order_probability((), (0, 1, 2), THETA4)
        self.assertEqual((r1.n_extensions, r2.n_extensions), (6, 6))

    def test_generators_accepted(self):
        def edges():
            yield C(0, 1)
            yield C(1, 2)
        r = constrained_order_probability(
            edges(), (x for x in (0, 1, 2)), (0.5, 0.2, -0.7))
        self.assertEqual(r.n_extensions, 1)


class NoPartialStateTests(unittest.TestCase):
    def test_inputs_not_mutated_by_success(self):
        cands = [0, 1, 2]
        edges = [C(0, 1)]
        constrained_order_probability(edges, cands, (0.5, 0.2, -0.7),
                                      return_extensions=True)
        self.assertEqual(cands, [0, 1, 2])
        self.assertEqual(edges, [C(0, 1)])

    def test_inputs_not_mutated_by_failure(self):
        cands = [0, 1, 2]
        edges = [C(0, 1), C(1, 0)]
        with self.assertRaises(PLInputError):
            constrained_order_probability(edges, cands, (0.5, 0.2, -0.7))
        self.assertEqual(cands, [0, 1, 2])
        self.assertEqual(edges, [C(0, 1), C(1, 0)])

    def test_rejected_call_leaves_reusable_state(self):
        # After a contradictory batch the same candidate set answers a valid
        # query normally (nothing is cached in a half-updated state).
        with self.assertRaises(PLInputError):
            constrained_order_probability(
                [C(0, 1), C(1, 2), C(2, 0)], (0, 1, 2),
                (0.5, 0.2, -0.7))
        r = constrained_order_probability(
            [C(0, 1)], (0, 1, 2), (0.5, 0.2, -0.7))
        self.assertEqual(r.n_extensions, 3)

    def test_validation_runs_before_any_probability(self):
        # A theta problem and a cycle in the same call surface the input
        # error deterministically (never a numerical result).
        with self.assertRaises(PLInputError):
            constrained_order_probability(
                [C(0, 1), C(1, 0)], (0, 1), (float("inf"), float("-inf")))


class ReturnShapeTests(unittest.TestCase):
    def test_extensions_opt_in_only(self):
        theta = (0.5, 0.2, -0.7)
        r = constrained_order_probability([C(0, 1)], (0, 1, 2), theta)
        self.assertEqual(r.extensions, ())
        r2 = constrained_order_probability(
            [C(0, 1)], (0, 1, 2), theta, return_extensions=True)
        self.assertEqual(len(r2.extensions), 3)
        self.assertEqual(r2.candidates, (0, 1, 2))
        for e in r2.extensions:
            self.assertIsInstance(e, tuple)

    def test_return_extensions_must_be_bool(self):
        with self.assertRaises(PLInputError):
            constrained_order_probability(
                None, (0, 1), THETA4, return_extensions=1)
        with self.assertRaises(PLInputError):
            constrained_order_probability(
                None, (0, 1), THETA4, return_extensions="yes")

    def test_log_and_probability_agree(self):
        r = constrained_order_probability(
            [C(0, 1)], (0, 1, 2), (0.5, 0.2, -0.7))
        self.assertAlmostEqual(math.exp(r.log_likelihood), r.likelihood,
                               places=14)
        self.assertLessEqual(r.likelihood, 1.0)
        self.assertGreater(r.likelihood, 0.0)

    def test_candidate_order_preserved(self):
        r = constrained_order_probability(
            [C(2, 0)], (2, 0, 1), THETA4)
        self.assertEqual(r.candidates, (2, 0, 1))


if __name__ == "__main__":
    unittest.main()
