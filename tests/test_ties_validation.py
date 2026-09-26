"""Contract / error-semantics tests for the two-way-tied model."""
import unittest

from plackett_luce import (
    PLInputError,
    TiedObservation,
    fit_tied_plackett_luce,
    tied_penalized_gradient_free,
    tied_ranking_log_probability,
)


def expect_error(testcase, fn, needle):
    with testcase.assertRaises(PLInputError) as ctx:
        fn()
    testcase.assertIn(needle, str(ctx.exception))


class TiedValidationTests(unittest.TestCase):
    def test_three_way_tie_rejected_up_front(self):
        expect_error(
            self,
            lambda: fit_tied_plackett_luce(
                [TiedObservation(((0, 1, 2),))], 3, 1.0),
            "exactly 1 or 2")

    def test_pair_repeating_same_object_rejected(self):
        expect_error(
            self,
            lambda: tied_ranking_log_probability(((1, 1),), [0.0, 0.0], 1.0),
            "repeats the same object")

    def test_object_appearing_in_two_groups_rejected(self):
        expect_error(
            self,
            lambda: fit_tied_plackett_luce(
                [TiedObservation(((0, 1), (1, 2)))], 3, 1.0),
            "two rank positions")

    def test_illegal_group_size_zero_rejected(self):
        expect_error(
            self,
            lambda: fit_tied_plackett_luce(
                [TiedObservation(((), (0,)))], 3, 1.0),
            "exactly 1 or 2")

    def test_index_out_of_range_rejected(self):
        expect_error(
            self,
            lambda: fit_tied_plackett_luce(
                [TiedObservation(((0, 3),))], 3, 1.0),
            "outside [0, 2]")

    def test_float_and_bool_indices_rejected(self):
        expect_error(
            self,
            lambda: tied_ranking_log_probability(((0.0,),), [0.0, 0.0], 1.0),
            "expected int")
        expect_error(
            self,
            lambda: tied_ranking_log_probability(((True,),), [0.0, 0.0], 1.0),
            "bool")

    def test_chosen_object_outside_participants_rejected(self):
        expect_error(
            self,
            lambda: fit_tied_plackett_luce(
                [TiedObservation(((0, 2),), participants=(0, 1))], 3, 1.0),
            "not in the participant set")

    def test_tau_must_be_positive_finite(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(PLInputError):
                fit_tied_plackett_luce(
                    [TiedObservation(((0,),))], 3, bad)

    def test_tau_bool_rejected(self):
        expect_error(
            self,
            lambda: fit_tied_plackett_luce([TiedObservation(((0,),))], 3, True),
            "bool")

    def test_non_tied_observation_rejected(self):
        expect_error(
            self,
            lambda: fit_tied_plackett_luce([(0, 1)], 3, 1.0),
            "TiedObservation")

    def test_theta_sum_constraint_reported_not_projected(self):
        expect_error(
            self,
            lambda: tied_penalized_gradient_free(
                [TiedObservation(((0, 1),))], [0.5, 0.0, 0.0], 1.0),
            "sum(theta) = 0")

    def test_duplicated_participants_rejected(self):
        expect_error(
            self,
            lambda: fit_tied_plackett_luce(
                [TiedObservation(((0,),), participants=(0, 0, 1))], 3, 1.0),
            "participants")

    def test_reversed_pair_is_the_same_canonical_outcome(self):
        # Not an error: (1, 0) denotes the same unordered joint outcome (0, 1).
        theta = [0.2, -0.1, -0.1]
        lp_forward = tied_ranking_log_probability(((0, 1),), theta, 1.0)
        lp_reverse = tied_ranking_log_probability(((1, 0),), theta, 1.0)
        self.assertEqual(lp_forward, lp_reverse)


if __name__ == "__main__":
    unittest.main()
