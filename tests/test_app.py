import unittest

from streamlit.testing.v1 import AppTest


class TeachingInterfaceTests(unittest.TestCase):
    def test_wrong_measurement_answer_gives_a_hint_and_allows_retry(self):
        app = AppTest.from_file("app.py").run(timeout=30)
        alice_bit = next(
            metric.value for metric in app.metric if metric.label == "Alice's bit"
        )
        alice_basis = next(
            metric.value for metric in app.metric if metric.label == "Alice's basis"
        )
        bob_basis = next(
            metric.value for metric in app.metric if metric.label == "Bob's random basis"
        )
        wrong_answer = (
            "1" if alice_bit == "0" else "0"
        ) if alice_basis == bob_basis else "0"
        guided_radio = next(
            radio
            for radio in app.radio
            if radio.label.startswith("What kind of output should Bob obtain")
        )
        guided_radio.set_value(wrong_answer)
        next(button for button in app.button if button.label == "Check my answer").click()
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Not quite" in message.value for message in app.warning))
        self.assertTrue(
            any(
                radio.label.startswith("What kind of output should Bob obtain")
                for radio in app.radio
            )
        )

    def test_guided_photons_unlock_qber_checkpoints(self):
        app = AppTest.from_file("app.py").run(timeout=30)
        self.assertEqual(len(app.exception), 0)

        for completed in range(5):
            guided_radio = next(
                radio
                for radio in app.radio
                if radio.label.startswith("What kind of output should Bob obtain")
            )
            alice_bit = next(
                metric.value for metric in app.metric if metric.label == "Alice's bit"
            )
            alice_basis = next(
                metric.value for metric in app.metric if metric.label == "Alice's basis"
            )
            bob_basis = next(
                metric.value for metric in app.metric if metric.label == "Bob's random basis"
            )
            expected = alice_bit if alice_basis == bob_basis else "Random (0 or 1)"

            guided_radio.set_value(expected)
            check_button = next(
                button for button in app.button if button.label == "Check my answer"
            )
            check_button.click()
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)

        self.assertTrue(
            any("Checkpoint complete" in message.value for message in app.success)
        )
        checkpoint_labels = [expander.label for expander in app.expander]
        self.assertIn("5. Student QBER checkpoint", checkpoint_labels)
        self.assertIn("6. Student QBER checkpoint", checkpoint_labels)
        self.assertEqual(
            sum(button.label == "Check my QBER" for button in app.button),
            2,
        )

        sifted_counts = [
            int(metric.value.replace(",", "").replace(" bits", ""))
            for metric in app.metric
            if metric.label == "Sifted key"
        ]
        error_counts = [
            int(metric.value.replace(",", ""))
            for metric in app.metric
            if metric.label == "Sifted errors"
        ]
        qbers = [
            100.0 * errors / sifted
            for errors, sifted in zip(error_counts, sifted_counts)
        ]
        threshold = next(
            slider.value for slider in app.slider if slider.label == "Maximum accepted QBER"
        )

        for expected_qber in qbers:
            qber_input = next(
                widget
                for widget in app.number_input
                if widget.label == "Your calculated QBER (%)"
            )
            qber_input.set_value(expected_qber)
            next(
                button for button in app.button if button.label == "Check my QBER"
            ).click()
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)

            decision = (
                "Abort — QBER is too high"
                if expected_qber > threshold
                else "Proceed — QBER is acceptable"
            )
            next(
                radio
                for radio in app.radio
                if radio.label.startswith("The machine accepts at or below")
            ).set_value(decision)
            next(
                button for button in app.button if button.label == "Check my decision"
            ).click()
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)

        self.assertTrue(
            any(
                subheader.value == "What changes when Eve listens?"
                for subheader in app.subheader
            )
        )


if __name__ == "__main__":
    unittest.main()
