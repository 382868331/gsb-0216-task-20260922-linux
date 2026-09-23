"""Input validation and error-semantics tests."""
import math
import unittest

from plackett_luce import (
    ConvergenceStatus,
    Observation,
    PLInputError,
    fit_plackett_luce,
    ranking_log_probability,
)


class ValidationTests(unittest.TestCase):
    def test_nan_infinity_rejected(self):
        obs = [Observation((0,))]
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(PLInputError) as cm:
                    fit_plackett_luce(obs, 2, lam=1.0, initial=(bad, -bad))
                self.assertIn("initial", str(cm.exception))
            with self.assertRaises(PLInputError):
                fit_plackett_luce(obs, 2, lam=bad)

    def test_bool_rejected_as_integer(self):
        obs = [Observation((0,))]
        with self.assertRaises(PLInputError):
            fit_plackett_luce(obs, True)
        with self.assertRaises(PLInputError):
            fit_plackett_luce(obs, 2, max_iter=True)
        with self.assertRaises(PLInputError):
            fit_plackett_luce(obs, 2, lam=True)
        # bool inside an index sequence is not silently accepted as 0/1.
        with self.assertRaises(PLInputError):
            fit_plackett_luce([Observation((True, 1))], 2)

    def test_n_objects_bounds(self):
        obs = [Observation((0,))]
        for bad in (0, 21, -1):
            with self.assertRaises(PLInputError):
                fit_plackett_luce(obs, bad)
        with self.assertRaises(PLInputError):
            fit_plackett_luce(obs, 2, lam=0.0)
        with self.assertRaises(PLInputError):
            fit_plackett_luce(obs, 2, max_iter=0)
        with self.assertRaises(PLInputError):
            fit_plackett_luce(obs, 2, gtol=0.0)

    def test_index_out_of_range_is_located(self):
        with self.assertRaises(PLInputError) as cm:
            fit_plackett_luce([Observation((2,))], 2)
        msg = str(cm.exception)
        self.assertIn("observations[0].ranking[0]", msg)
        self.assertIn("2", msg)

    def test_duplicate_ranking_rejected(self):
        with self.assertRaises(PLInputError) as cm:
            fit_plackett_luce([Observation((0, 1, 0))], 3)
        self.assertIn("duplicated", str(cm.exception))
        self.assertIn("observations[0].ranking", str(cm.exception))

    def test_duplicate_participants_rejected(self):
        with self.assertRaises(PLInputError) as cm:
            fit_plackett_luce(
                [Observation((0,), participants=(0, 1, 1))], 3)
        self.assertIn("participants", str(cm.exception))

    def test_ranked_object_must_participate(self):
        with self.assertRaises(PLInputError) as cm:
            fit_plackett_luce(
                [Observation((1,), participants=(0, 2))], 3)
        self.assertIn("not in the participant set", str(cm.exception))

    def test_wrong_element_type(self):
        with self.assertRaises(PLInputError) as cm:
            fit_plackett_luce([(0, 1)], 2)
        self.assertIn("observations[0]", str(cm.exception))

    def test_initial_must_match_length_and_sum_zero(self):
        obs = [Observation((0,))]
        with self.assertRaises(PLInputError):
            fit_plackett_luce(obs, 3, initial=(0.0, 0.0))
        with self.assertRaises(PLInputError) as cm:
            fit_plackett_luce(obs, 2, initial=(1e-9, -1e-9 + 1.0))
        self.assertIn("sum(theta) = 0", str(cm.exception))
        # Within the documented relative tolerance: accepted.
        r = fit_plackett_luce(obs, 2, initial=(0.0, 1e-12))
        self.assertTrue(r.converged)

    def test_k0_accepted_empty_ranking(self):
        r = fit_plackett_luce([Observation(())], 2)
        self.assertEqual(r.status, ConvergenceStatus.CONVERGED)
        self.assertEqual(r.log_contributions, (0.0,))

    def test_empty_batch_pins_zero(self):
        r = fit_plackett_luce([], 3)
        self.assertTrue(r.converged)
        self.assertEqual(r.theta, (0.0, 0.0, 0.0))
        self.assertEqual(r.log_likelihood, 0.0)
        self.assertEqual(r.penalized_objective, 0.0)

    def test_generator_observations_accepted(self):
        def gen():
            yield Observation((0, 1))
            yield Observation((1,), participants=(0, 1))
        r = fit_plackett_luce(gen(), 2)
        self.assertTrue(r.converged)

    def test_one_object_trivial(self):
        r = fit_plackett_luce([Observation((0,)), Observation(())], 1)
        self.assertTrue(r.converged)
        self.assertEqual(r.theta, (0.0,))
        # k=0 contributes 0; a lone participant wins with probability 1.
        self.assertEqual(r.log_contributions, (0.0, 0.0))

    def test_ranking_probability_validates_theta(self):
        with self.assertRaises(PLInputError):
            ranking_log_probability((0,), (float("nan"), 0.0))
        with self.assertRaises(PLInputError):
            ranking_log_probability((0,), (1.0, 0.0))  # sum != 0

    def test_nan_theta_in_probability(self):
        with self.assertRaises(PLInputError):
            ranking_log_probability((0,), (0.0, float("nan")))


if __name__ == "__main__":
    unittest.main()
