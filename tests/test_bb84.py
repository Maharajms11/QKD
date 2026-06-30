import unittest

import numpy as np

from bb84 import (
    calculate_qber,
    calculate_qber_from_counts,
    encode_photons,
    expected_qber,
    generate_bases,
    generate_bits,
    predictable_measurement_bit,
    run_protocol,
    sample_sifted_key,
)


class BB84Tests(unittest.TestCase):
    def test_generation_and_encoding(self):
        rng = np.random.default_rng(1)
        bits = generate_bits(1000, rng)
        bases = generate_bases(1000, rng)
        self.assertEqual(len(bits), 1000)
        self.assertTrue(set(bits).issubset({0, 1}))
        self.assertTrue(set(bases).issubset({"+", "x"}))
        np.testing.assert_array_equal(
            encode_photons(
                np.array([0, 1, 0, 1]),
                np.array(["+", "+", "x", "x"]),
            ),
            np.array([0, 90, 45, 135]),
        )

    def test_no_eve_and_no_noise_has_zero_sifted_errors(self):
        rng = np.random.default_rng(2)
        bits = generate_bits(1000, rng)
        alice_bases = generate_bases(1000, rng)
        bob_bases = generate_bases(1000, rng)
        result = run_protocol(bits, alice_bases, bob_bases, 0.0, 0, rng)
        self.assertEqual(result.qber, 0.0)
        self.assertEqual(int(result.sifted_errors.sum()), 0)

    def test_full_intercept_resend_approaches_25_percent_qber(self):
        qbers = []
        for seed in range(30):
            rng = np.random.default_rng(seed)
            bits = generate_bits(1000, rng)
            alice_bases = generate_bases(1000, rng)
            bob_bases = generate_bases(1000, rng)
            result = run_protocol(bits, alice_bases, bob_bases, 0.0, 1000, rng)
            qbers.append(result.qber)
        self.assertAlmostEqual(float(np.mean(qbers)), 25.0, delta=2.0)

    def test_error_attribution_sums_to_observed_errors(self):
        rng = np.random.default_rng(42)
        bits = generate_bits(1000, rng)
        alice_bases = generate_bases(1000, rng)
        bob_bases = generate_bases(1000, rng)
        result = run_protocol(bits, alice_bases, bob_bases, 0.05, 600, rng)
        self.assertEqual(
            int(result.sifted_errors.sum()),
            result.noise_attributed_errors + result.eve_attributed_errors,
        )

    def test_qber_helpers(self):
        errors, qber = calculate_qber(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1]))
        self.assertEqual(errors, 2)
        self.assertEqual(qber, 50.0)
        self.assertEqual(calculate_qber_from_counts(2, 8), 25.0)
        self.assertEqual(expected_qber(1.0, 0.0), 25.0)
        self.assertAlmostEqual(expected_qber(0.0, 0.05), 5.0)

    def test_guided_measurement_prediction(self):
        self.assertEqual(predictable_measurement_bit(0, "+", "+"), 0)
        self.assertEqual(predictable_measurement_bit(1, "x", "x"), 1)
        self.assertIsNone(predictable_measurement_bit(0, "+", "x"))
        self.assertIsNone(predictable_measurement_bit(1, "x", "+"))

    def test_qber_count_validation(self):
        with self.assertRaises(ValueError):
            calculate_qber_from_counts(6, 5)
        with self.assertRaises(ValueError):
            calculate_qber_from_counts(-1, 5)

    def test_parameter_estimation_discards_revealed_sample(self):
        alice = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
        bob = np.array([0, 1, 1, 1, 1, 0, 1, 1], dtype=np.int8)
        first = sample_sifted_key(alice, bob, 0.25, np.random.default_rng(7))
        second = sample_sifted_key(alice, bob, 0.25, np.random.default_rng(7))

        np.testing.assert_array_equal(first.sample_indices, second.sample_indices)
        self.assertEqual(len(first.sample_indices), 2)
        self.assertEqual(len(first.remaining_alice_key), 6)
        self.assertEqual(
            len(first.sample_alice_bits) + len(first.remaining_alice_key),
            len(alice),
        )
        np.testing.assert_array_equal(
            np.delete(alice, first.sample_indices), first.remaining_alice_key
        )
        np.testing.assert_array_equal(
            np.delete(bob, first.sample_indices), first.remaining_bob_key
        )
        self.assertEqual(
            first.sample_qber,
            100.0 * int(first.sample_errors.sum()) / len(first.sample_errors),
        )

    def test_parameter_estimation_validates_fraction(self):
        with self.assertRaises(ValueError):
            sample_sifted_key(np.array([0]), np.array([0]), 0.0)
        with self.assertRaises(ValueError):
            sample_sifted_key(np.array([0]), np.array([0]), 1.0)


if __name__ == "__main__":
    unittest.main()
