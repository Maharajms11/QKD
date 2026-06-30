"""Streamlit interface for the idealised BB84 teaching simulator."""

from __future__ import annotations

import secrets

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from bb84 import (
    calculate_qber,
    calculate_qber_from_counts,
    expected_qber,
    generate_bases,
    generate_bits,
    photon_state_labels,
    predictable_measurement_bit,
    run_protocol,
    sample_sifted_key,
)


N_BITS = 1000
SAMPLE_ROWS = 30
GUIDED_PHOTONS = 5
RANDOM_ANSWER = "Random (0 or 1)"


def prepare_exchange(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare reproducible Alice and Bob choices from a classroom seed."""

    seed_sequence = np.random.SeedSequence(seed)
    alice_bit_seed, alice_basis_seed, bob_basis_seed = seed_sequence.spawn(3)
    alice_bits = generate_bits(N_BITS, np.random.default_rng(alice_bit_seed))
    alice_bases = generate_bases(N_BITS, np.random.default_rng(alice_basis_seed))
    bob_bases = generate_bases(N_BITS, np.random.default_rng(bob_basis_seed))
    return alice_bits, alice_bases, bob_bases


def yes_no(values: np.ndarray) -> list[str]:
    return ["Yes" if bool(value) else "No" for value in values]


def conceptual_measurement_answer(bit: int, alice_basis: str, bob_basis: str) -> str:
    """Return the classroom answer for a BB84 measurement prediction."""

    predictable = predictable_measurement_bit(bit, alice_basis, bob_basis)
    return RANDOM_ANSWER if predictable is None else str(predictable)


def measurement_hint(bit: int, alice_basis: str, bob_basis: str, attempt: int) -> str:
    """Give progressively more explicit feedback without penalising randomness."""

    bases_match = alice_basis == bob_basis
    if bases_match:
        hints = [
            "Compare Alice's preparation basis with Bob's measurement basis. They match.",
            "With matching bases, Bob recovers the bit Alice encoded.",
            f"Alice encoded bit {bit}; use that same bit as the predictable output.",
        ]
    else:
        hints = [
            "Compare Alice's preparation basis with Bob's measurement basis. They differ.",
            "A measurement in the wrong basis has no single predictable bit value.",
            f"Choose “{RANDOM_ANSWER}”. The simulator will then reveal the sampled bit.",
        ]
    return hints[min(attempt - 1, len(hints) - 1)]


def guided_measurement_lab(
    alice_bits: np.ndarray,
    alice_bases: np.ndarray,
    bob_bases: np.ndarray,
    result,
    seed: int,
) -> bool:
    """Require students to reason through the first few measurements."""

    signature = f"{seed}"
    progress_key = f"guided_progress_{signature}"
    feedback_key = f"guided_feedback_{signature}"
    st.session_state.setdefault(progress_key, 0)
    progress = min(int(st.session_state[progress_key]), GUIDED_PHOTONS)

    st.subheader("Your turn: predict Bob's first measurements")
    st.write(
        "Work through the first five photons. Choose `0` or `1` when Bob's basis makes "
        "the result predictable; choose **Random (0 or 1)** when the bases differ. "
        "After each correct answer, the simulator reveals the sampled measurement."
    )
    st.progress(
        progress / GUIDED_PHOTONS,
        text=f"{progress} of {GUIDED_PHOTONS} photons completed",
    )

    if progress:
        completed = np.arange(progress)
        history = pd.DataFrame(
            {
                "Photon": completed + 1,
                "Alice bit": alice_bits[completed],
                "Alice basis": alice_bases[completed],
                "Bob basis": bob_bases[completed],
                "Conceptual answer": [
                    conceptual_measurement_answer(
                        int(alice_bits[i]), str(alice_bases[i]), str(bob_bases[i])
                    )
                    for i in completed
                ],
                "Sampled bit before channel error": result.bob.measured_bits_before_noise[
                    completed
                ],
                "Channel flipped it?": yes_no(result.bob.noise_flips[completed]),
                "Final measured bit": result.bob.measured_bits[completed],
            }
        )
        st.dataframe(history, hide_index=True, width="stretch")

    feedback = st.session_state.get(feedback_key)
    if feedback:
        if feedback["kind"] == "success":
            st.success(feedback["text"])
        else:
            st.warning(feedback["text"])

    if progress >= GUIDED_PHOTONS:
        st.success(
            "Checkpoint complete. The simulator has now generated the remaining "
            f"{N_BITS - GUIDED_PHOTONS:,} measurement outputs."
        )
        return True

    i = progress
    bit = int(alice_bits[i])
    alice_basis = str(alice_bases[i])
    bob_basis = str(bob_bases[i])
    state_label = str(photon_state_labels(np.array([result.alice_states[i]]))[0])

    st.markdown(f"#### Photon {i + 1}")
    prompt_columns = st.columns(4)
    prompt_columns[0].metric("Alice's bit", bit)
    prompt_columns[1].metric("Alice's basis", alice_basis)
    prompt_columns[2].metric("Photon state", state_label)
    prompt_columns[3].metric("Bob's random basis", bob_basis)

    answer_key = f"guided_answer_{signature}_{i}"
    attempt_key = f"guided_attempts_{signature}_{i}"
    answer = st.radio(
        "What kind of output should Bob obtain before channel error is applied?",
        options=["0", "1", RANDOM_ANSWER],
        index=None,
        horizontal=True,
        key=answer_key,
    )
    if st.button("Check my answer", type="primary", key=f"check_{signature}_{i}"):
        if answer is None:
            st.session_state[feedback_key] = {
                "kind": "hint",
                "text": "Choose an answer first, then check it.",
            }
        else:
            expected = conceptual_measurement_answer(bit, alice_basis, bob_basis)
            if answer == expected:
                before_noise = int(result.bob.measured_bits_before_noise[i])
                final_bit = int(result.bob.measured_bits[i])
                noise_note = (
                    f" The channel then flipped it to {final_bit}."
                    if bool(result.bob.noise_flips[i])
                    else " The channel did not flip it."
                )
                st.session_state[progress_key] = progress + 1
                st.session_state[feedback_key] = {
                    "kind": "success",
                    "text": (
                        f"Correct. The simulator sampled output {before_noise}."
                        f"{noise_note}"
                    ),
                }
            else:
                attempts = int(st.session_state.get(attempt_key, 0)) + 1
                st.session_state[attempt_key] = attempts
                st.session_state[feedback_key] = {
                    "kind": "hint",
                    "text": "Not quite. "
                    + measurement_hint(bit, alice_basis, bob_basis, attempts)
                    + " Try again.",
                }
        st.rerun()
    return False


def sampled_stream_indices(result, estimate) -> np.ndarray:
    """Map public-sample positions from the sifted key to the original stream."""

    return np.flatnonzero(result.sift_mask)[estimate.sample_indices]


def public_sample_table(result, estimate) -> pd.DataFrame:
    """Build the list of publicly compared bits that must be discarded."""

    stream_indices = sampled_stream_indices(result, estimate)
    return pd.DataFrame(
        {
            "Original stream position": stream_indices + 1,
            "Alice reveals": estimate.sample_alice_bits,
            "Bob reveals": estimate.sample_bob_bits,
            "Error?": yes_no(estimate.sample_errors),
            "After comparison": ["Discarded"] * len(stream_indices),
        }
    )


def sample_table(
    result,
    estimate=None,
    include_eve: bool = False,
) -> pd.DataFrame:
    """Build the first-30-position audit trail shown to students."""

    n = min(SAMPLE_ROWS, len(result.alice_bits))
    kept = result.sift_mask[:n]
    errors = np.where(
        kept,
        np.where(result.bob.measured_bits[:n] != result.alice_bits[:n], "Yes", "No"),
        "N/A",
    )
    data: dict[str, object] = {
        "Index": np.arange(1, n + 1),
        "Alice bit": result.alice_bits[:n],
        "Alice basis": result.alice_bases[:n],
        "Encoded photon state": photon_state_labels(result.alice_states[:n]),
    }
    if estimate is not None:
        public_sample = np.zeros(len(result.alice_bits), dtype=bool)
        public_sample[sampled_stream_indices(result, estimate)] = True
        data["Selected for public test?"] = yes_no(public_sample[:n])
    if include_eve:
        intercepted = result.eve.intercepted[:n]
        data.update(
            {
                "Eve intercepted?": yes_no(intercepted),
                "Eve basis": np.where(intercepted, result.eve.eve_bases[:n], "N/A"),
                "Eve measured bit": np.where(intercepted, result.eve.eve_bits[:n].astype(str), "N/A"),
                "Photon Bob receives": photon_state_labels(result.eve.resent_states[:n]),
            }
        )
    data.update(
        {
            "Bob basis": result.bob_bases[:n],
            "Bob measured bit": result.bob.measured_bits[:n],
            "Basis match?": yes_no(kept),
            "Kept?": yes_no(kept),
            "Error?": errors,
        }
    )
    return pd.DataFrame(data)


@st.cache_data(show_spinner=False)
def qber_sweep(
    channel_noise: float,
    sample_fraction: float,
    base_seed: int,
    trials: int = 16,
) -> pd.DataFrame:
    """Average repeated sample-QBER estimates for a readable teaching chart."""

    records: list[dict[str, float | str]] = []
    for percentage in range(0, 101, 5):
        simulated_qbers: list[float] = []
        for trial in range(trials):
            seed = base_seed + percentage * 10_000 + trial
            bits, alice_bases, bob_bases = prepare_exchange(seed)
            result = run_protocol(
                bits,
                alice_bases,
                bob_bases,
                channel_noise,
                round(N_BITS * percentage / 100),
                np.random.default_rng(seed + 91_337),
            )
            estimate = sample_sifted_key(
                result.sifted_alice_key,
                result.sifted_bob_key,
                sample_fraction,
                np.random.default_rng(seed + 48_271),
            )
            simulated_qbers.append(estimate.sample_qber)
        records.append(
            {
                "Eve interception (%)": float(percentage),
                "QBER (%)": float(np.mean(simulated_qbers)),
                "Series": "Simulated mean",
            }
        )
        records.append(
            {
                "Eve interception (%)": float(percentage),
                "QBER (%)": expected_qber(percentage / 100, channel_noise),
                "Series": "Theoretical expectation",
            }
        )
    return pd.DataFrame(records)


def metric_row(result, estimate, include_eve: bool = False) -> None:
    """Render the principal numbers for an exchange."""

    columns = st.columns(7 if include_eve else 6)
    columns[0].metric("Transmitted", f"{len(result.alice_bits):,}")
    columns[1].metric("Basis matches", f"{int(result.sift_mask.sum()):,}")
    columns[2].metric("Sifted key", f"{len(result.sifted_alice_key):,} bits")
    columns[3].metric("Public test sample", f"{len(estimate.sample_alice_bits):,} bits")
    columns[4].metric("Revealed errors", f"{int(estimate.sample_errors.sum()):,}")
    columns[5].metric(
        "Sample QBER",
        "Calculate it below",
    )
    if include_eve:
        columns[6].metric("Eve intercepted", f"{int(result.eve.intercepted.sum()):,}")


def detection_panel(
    qber: float,
    threshold: float,
    candidate_key_bits: int,
    step_number: int = 7,
) -> None:
    """Apply the user-selected classroom abort rule."""

    with st.expander(f"{step_number}. QBER-based detection", expanded=True):
        st.write(
            f"The publicly revealed sample has been discarded. The abort threshold is "
            f"**{threshold:.1f}%**, and the estimated sample QBER is **{qber:.2f}%**."
        )
        if qber > threshold:
            st.error(
                "Eavesdropping or excessive channel disturbance detected: abort and "
                "discard this entire exchange."
            )
        else:
            st.success(
                f"QBER below threshold: the {candidate_key_bits:,} unrevealed candidate "
                "bits proceed to error correction and privacy amplification."
            )
        st.caption(
            "Teaching simplification: this decision uses the sample percentage itself. "
            "A real finite-key protocol applies a statistical confidence margin because "
            "the unseen candidate key can have a different error rate from the sample."
        )


def qber_learning_panel(
    estimate,
    threshold: float,
    noise_percent: int,
    sample_percent: int,
    seed: int,
    scenario_key: str,
    step_number: int,
) -> bool:
    """Require a QBER calculation and threshold decision before revealing the answer."""

    error_count = int(estimate.sample_errors.sum())
    sample_count = len(estimate.sample_alice_bits)
    expected = calculate_qber_from_counts(error_count, sample_count)
    signature = (
        f"{scenario_key}_{seed}_{noise_percent}_{sample_percent}_{error_count}_{sample_count}_"
        f"{threshold:.1f}"
    ).replace(".", "_")
    calculation_key = f"qber_calculation_correct_{signature}"
    calculation_attempts_key = f"qber_calculation_attempts_{signature}"
    decision_key = f"qber_decision_correct_{signature}"
    decision_attempts_key = f"qber_decision_attempts_{signature}"
    st.session_state.setdefault(calculation_key, False)
    st.session_state.setdefault(decision_key, False)

    with st.expander(f"{step_number}. Student QBER checkpoint", expanded=True):
        st.info(
            f"Machine settings: **{noise_percent}% inherent channel error probability** "
            f"and **{threshold:.1f}% maximum accepted QBER**. Alice and Bob randomly "
            f"selected **{sample_percent}% of the sifted key** for public testing."
        )
        st.write(
            f"They found **{error_count} errors** among the **{sample_count} publicly "
            "compared sample bits**. Calculate the estimated quantum bit error rate."
        )
        st.latex(r"\mathrm{QBER}(\%)=\frac{\text{sample errors}}{\text{sample bits}}\times100")

        if not st.session_state[calculation_key]:
            learner_qber = st.number_input(
                "Your calculated QBER (%)",
                min_value=0.0,
                max_value=100.0,
                value=None,
                step=0.01,
                format="%.2f",
                key=f"qber_input_{signature}",
                placeholder="Enter a percentage",
            )
            if st.button(
                "Check my QBER",
                type="primary",
                key=f"check_qber_{signature}",
            ):
                if learner_qber is None:
                    st.warning("Enter your calculated percentage before checking it.")
                elif abs(float(learner_qber) - expected) <= 0.05:
                    st.session_state[calculation_key] = True
                    st.rerun()
                else:
                    attempts = int(st.session_state.get(calculation_attempts_key, 0)) + 1
                    st.session_state[calculation_attempts_key] = attempts
                    if attempts == 1:
                        st.warning(
                            f"Not quite. Substitute the counts: {error_count} ÷ "
                            f"{sample_count} × 100. Try again."
                        )
                    elif attempts == 2:
                        fraction = error_count / sample_count if sample_count else 0.0
                        st.warning(
                            f"First, {error_count} ÷ {sample_count} = {fraction:.4f}. "
                            "Now convert that decimal to a percentage and try again."
                        )
                    else:
                        st.warning(
                            "Multiply the decimal by 100 and round the result to two "
                            "decimal places. Try again."
                        )
            return False

        st.success(
            f"Correct: {error_count} ÷ {sample_count} × 100 = **{expected:.2f}%**. "
            f"All {sample_count} revealed sample bits are now discarded."
        )

        if not st.session_state[decision_key]:
            decision = st.radio(
                f"The machine accepts at or below {threshold:.1f}% QBER. What should it do?",
                options=[
                    "Proceed — QBER is acceptable",
                    "Abort — QBER is too high",
                ],
                index=None,
                key=f"decision_input_{signature}",
            )
            if st.button(
                "Check my decision",
                type="primary",
                key=f"check_decision_{signature}",
            ):
                should_abort = expected > threshold
                expected_decision = (
                    "Abort — QBER is too high"
                    if should_abort
                    else "Proceed — QBER is acceptable"
                )
                if decision == expected_decision:
                    st.session_state[decision_key] = True
                    st.rerun()
                else:
                    attempts = int(st.session_state.get(decision_attempts_key, 0)) + 1
                    st.session_state[decision_attempts_key] = attempts
                    relation = "greater than" if should_abort else "not greater than"
                    st.warning(
                        f"Compare the values directly: {expected:.2f}% is {relation} "
                        f"the {threshold:.1f}% limit. Try again."
                    )
            return False

        st.success("Correct security decision.")

    detection_panel(
        expected,
        threshold,
        len(estimate.remaining_alice_key),
        step_number + 1,
    )
    return True


st.set_page_config(
    page_title="BB84 Quantum Key Distribution Lab",
    page_icon="🔐",
    layout="wide",
)

st.title("🔐 BB84 Quantum Key Distribution Lab")
st.caption("A transparent, step-by-step 1,000-photon teaching simulator")
st.info(
    "This is an idealised BB84 teaching simulator, not a production QKD implementation. "
    "Every intermediate choice is exposed so that students can inspect how preparation, "
    "measurement, sifting, noise, and intercept-resend affect the key."
)

if "exchange_seed" not in st.session_state:
    st.session_state.exchange_seed = secrets.randbelow(2_000_000_000)

with st.sidebar:
    st.header("Experiment controls")
    if st.button("Generate a new 1,000-bit exchange", type="primary", width="stretch"):
        st.session_state.exchange_seed = secrets.randbelow(2_000_000_000)
        st.rerun()
    seed = st.number_input(
        "Reproducible experiment seed",
        min_value=0,
        max_value=2_147_483_647,
        value=int(st.session_state.exchange_seed),
        step=1,
        help="Use the same seed to reproduce Alice's and Bob's random choices.",
    )
    if int(seed) != st.session_state.exchange_seed:
        st.session_state.exchange_seed = int(seed)

    noise_percent = st.radio(
        "Inherent channel error probability",
        options=[0, 1, 2, 5, 10],
        horizontal=True,
        format_func=lambda value: f"{value}%",
        help="Independent probability that a measured bit is flipped.",
    )
    sample_percent = st.radio(
        "Sifted key revealed for public testing",
        options=[5, 10, 20, 25, 50],
        index=2,
        horizontal=True,
        format_func=lambda value: f"{value}%",
        help=(
            "Alice and Bob compare this random sample to estimate QBER, then discard "
            "every revealed sample bit."
        ),
    )
    eve_count = st.slider(
        "Photons Eve intercepts",
        min_value=0,
        max_value=N_BITS,
        value=500,
        step=10,
        help="Eve measures and resends exactly this many of the 1,000 photons.",
    )
    threshold = st.slider(
        "Maximum accepted QBER",
        min_value=0.0,
        max_value=25.0,
        value=11.0,
        step=0.5,
        format="%.1f%%",
    )
    st.caption(
        "The inherent error setting creates background errors. The QBER limit is the "
        "separate accept/abort rule. Public test bits are always discarded."
    )

channel_noise = noise_percent / 100.0
alice_bits, alice_bases, bob_bases = prepare_exchange(int(seed))
no_eve = run_protocol(
    alice_bits,
    alice_bases,
    bob_bases,
    channel_noise,
    0,
    np.random.default_rng(int(seed) + 10_001),
)
with_eve = run_protocol(
    alice_bits,
    alice_bases,
    bob_bases,
    channel_noise,
    eve_count,
    np.random.default_rng(int(seed) + 20_003),
)
sample_fraction = sample_percent / 100.0
sample_seed = int(seed) + 30_007
no_eve_estimate = sample_sifted_key(
    no_eve.sifted_alice_key,
    no_eve.sifted_bob_key,
    sample_fraction,
    np.random.default_rng(sample_seed),
)
with_eve_estimate = sample_sifted_key(
    with_eve.sifted_alice_key,
    with_eve.sifted_bob_key,
    sample_fraction,
    np.random.default_rng(sample_seed),
)

st.subheader("The BB84 journey")
stage_a, stage_b, stage_c = st.columns(3)
with stage_a:
    with st.expander("1. Alice preparation", expanded=True):
        st.write(
            "Alice independently generates 1,000 random bits and 1,000 random bases. "
            "Neither Bob nor Eve knows these choices during transmission."
        )
        st.code(
            "Bits:  " + "".join(map(str, alice_bits[:24])) + "\n"
            "Bases: " + "".join(alice_bases[:24]),
            language=None,
        )
with stage_b:
    with st.expander("2. Photon encoding", expanded=True):
        st.write("Each bit/basis pair determines one idealised polarization state:")
        st.markdown(
            "- `0` with `+` → **0° horizontal**\n"
            "- `1` with `+` → **90° vertical**\n"
            "- `0` with `x` → **45° diagonal**\n"
            "- `1` with `x` → **135° anti-diagonal**"
        )
with stage_c:
    with st.expander("3. Bob measurement", expanded=True):
        st.write(
            "Bob independently chooses + or x. If his basis matches the photon he "
            "receives, he recovers its encoded bit. Otherwise quantum measurement gives "
            "a random result. Channel noise may then flip that result."
        )
        st.code("First 24 bases:\n" + "".join(bob_bases[:24]), language=None)

st.divider()
guided_complete = guided_measurement_lab(
    alice_bits,
    alice_bases,
    bob_bases,
    no_eve,
    int(seed),
)
if not guided_complete:
    st.info(
        "Complete the five-photon checkpoint to unlock the automatically generated "
        "exchange, Eve scenario, QBER exercises, and comparison charts."
    )
    st.stop()

st.divider()
st.header("Choose your learning mode")
st.write(
    "Follow Modes 1 → 2 → 3 for the complete experiment. Mode 4 can be opened "
    "at any time to connect the classroom model to real QKD systems."
)
mode_columns = st.columns(4)
with mode_columns[0]:
    with st.container(border=True):
        st.markdown("### 🟢 1. No Eve")
        st.write(
            "Establish the baseline. Observe errors caused only by the selected channel "
            "noise, test a public sample, and calculate its QBER."
        )
with mode_columns[1]:
    with st.container(border=True):
        st.markdown("### 🕵️ 2. Eve attack")
        st.write(
            "Introduce intercept-resend eavesdropping. See how Eve's measurements "
            "disturb the photon stream and alter the sample QBER."
        )
with mode_columns[2]:
    with st.container(border=True):
        st.markdown("### 📊 3. Compare")
        st.write(
            "Compare the baseline and attack results. This mode unlocks after both QBER "
            "checkpoints are completed, so it cannot reveal answers early."
        )
with mode_columns[3]:
    with st.container(border=True):
        st.markdown("### 🌍 4. Real QKD")
        st.write(
            "Review what this idealised simulator teaches and which security, hardware, "
            "and post-processing steps a real QKD system still requires."
        )

tab_no_eve, tab_eve, tab_compare, tab_reality = st.tabs(
    [
        "🟢 1. No Eve",
        "🕵️ 2. Eve attack",
        "📊 3. Compare",
        "🌍 4. Real QKD",
    ]
)

with tab_no_eve:
    st.info(
        "Mode 1 — Baseline: no eavesdropper is present. Any observed errors come from "
        "the selected inherent channel error probability."
    )
    st.subheader(f"Alice → noisy channel ({noise_percent}%) → Bob")
    metric_row(no_eve, no_eve_estimate)
    with st.expander("4. Basis sifting", expanded=True):
        st.write(
            "After all measurements, Alice and Bob publicly compare bases—but never reveal "
            "their bit values. They keep only positions where their original bases match. "
            "About half of 1,000 positions should survive."
        )
        st.progress(len(no_eve.sifted_alice_key) / N_BITS, text=f"{len(no_eve.sifted_alice_key)} of 1,000 positions kept")
    with st.expander("5. Random public test sample", expanded=True):
        st.write(
            f"Alice and Bob randomly select **{len(no_eve_estimate.sample_alice_bits)} of "
            f"{len(no_eve.sifted_alice_key)} sifted bits** ({sample_percent}%). They "
            "publicly reveal both values at these positions, count disagreements, and "
            "permanently discard every revealed bit."
        )
        st.dataframe(
            public_sample_table(no_eve, no_eve_estimate),
            hide_index=True,
            width="stretch",
            height=300,
        )
        st.caption(
            f"After sampling, {len(no_eve_estimate.remaining_alice_key)} unrevealed "
            "candidate-key bits remain. Their exact error rate is not known yet."
        )
    st.markdown("#### First 30 transmitted positions (simulator audit view)")
    st.dataframe(
        sample_table(no_eve, no_eve_estimate),
        hide_index=True,
        width="stretch",
        height=560,
    )
    no_eve_qber_complete = qber_learning_panel(
        no_eve_estimate,
        threshold,
        int(noise_percent),
        int(sample_percent),
        int(seed),
        "no_eve",
        6,
    )

with tab_eve:
    st.info(
        "Mode 2 — Attack: Eve measures and resends selected photons. Compare this mode's "
        "sample QBER with the no-Eve baseline to see the disturbance she introduces."
    )
    st.subheader(
        f"Alice → Eve intercepts {eve_count:,} photons ({100 * eve_count / N_BITS:.1f}%) → Bob"
    )
    metric_row(with_eve, with_eve_estimate, include_eve=True)
    intercepted_sifted = int(np.count_nonzero(with_eve.sift_mask & with_eve.eve.intercepted))
    eve_matches = int(with_eve.eve.eve_basis_matches.sum())
    expected_eve_errors = intercepted_sifted * 0.25
    expected_overall_contribution = 25 * eve_count / N_BITS

    with st.expander("4. Eve intercept-resend attack", expanded=True):
        st.write(
            "For every intercepted photon, Eve randomly chooses + or x, measures, and "
            "prepares a replacement photon from her result. A wrong choice destroys the "
            "original state. Eve cannot quietly copy an unknown quantum state."
        )
        eve_cols = st.columns(5)
        eve_cols[0].metric("Intercepted", f"{eve_count:,} / {N_BITS:,}")
        eve_cols[1].metric("Interception", f"{100 * eve_count / N_BITS:.1f}%")
        eve_cols[2].metric("Eve basis matches", f"{eve_matches:,}")
        eve_cols[3].metric("Attacked sifted bits", f"{intercepted_sifted:,}")
        eve_cols[4].metric("Expected Eve errors", f"≈ {expected_eve_errors:.1f}")
        st.caption(
            f"Intercept-resend creates a 25% error rate among attacked sifted bits; at this "
            f"interception level its expected contribution to overall QBER before noise is "
            f"{expected_overall_contribution:.2f}%."
        )

    with st.expander("5. Basis sifting", expanded=True):
        st.write(
            f"Alice and Bob keep **{len(with_eve.sifted_alice_key)}** positions where their "
            "own bases match. They do not know which of those photons Eve touched."
        )
        st.progress(len(with_eve.sifted_alice_key) / N_BITS, text=f"{len(with_eve.sifted_alice_key)} of 1,000 positions kept")

    with st.expander("6. Random public test sample", expanded=True):
        st.write(
            f"Alice and Bob randomly select **{len(with_eve_estimate.sample_alice_bits)} "
            f"of {len(with_eve.sifted_alice_key)} sifted bits** ({sample_percent}%). "
            "They reveal and compare these values publicly, then discard all sampled bits."
        )
        st.dataframe(
            public_sample_table(with_eve, with_eve_estimate),
            hide_index=True,
            width="stretch",
            height=300,
        )
        st.caption(
            f"After sampling, {len(with_eve_estimate.remaining_alice_key)} unrevealed "
            "candidate-key bits remain. Eve hears the sample, but those bits will never "
            "be used as secret key material."
        )

    st.markdown("#### First 30 transmitted positions, including Eve's actions")
    st.dataframe(
        sample_table(with_eve, with_eve_estimate, include_eve=True),
        hide_index=True,
        width="stretch",
        height=560,
    )

    eve_qber_complete = qber_learning_panel(
        with_eve_estimate,
        threshold,
        int(noise_percent),
        int(sample_percent),
        int(seed),
        f"eve_{eve_count}",
        7,
    )

    if eve_qber_complete:
        with st.expander("9. Simulator-only error attribution", expanded=True):
            total_errors = int(with_eve.sifted_errors.sum())
            st.write(
                f"The observed sifted-key QBER is **{total_errors} / "
                f"{len(with_eve.sifted_alice_key)} = {with_eve.qber:.2f}%**. Because this is a "
                "simulation, we can inspect the hidden causal record: "
                f"**{with_eve.eve_attributed_errors}** final errors are attributable to Eve and "
                f"**{with_eve.noise_attributed_errors}** to channel flips."
            )
            if with_eve.eve_errors_cancelled_by_noise:
                st.caption(
                    f"An extra teaching subtlety: {with_eve.eve_errors_cancelled_by_noise} "
                    "Eve-induced error(s) were flipped back to the correct bit by channel noise. "
                    "Real Alice and Bob cannot label or causally separate individual errors."
                )

with tab_reality:
    st.info(
        "Mode 4 — Real-world context: use this mode at any time. It explains which parts "
        "of BB84 are represented here and which production requirements are deliberately omitted."
    )
    st.subheader("What this model teaches—and what it leaves out")
    st.markdown(
        """
This simulator isolates the central BB84 idea: measuring an unknown quantum state can
disturb it, and Alice and Bob can detect excess disturbance statistically. It is deliberately
not cryptographic production software.

A real QKD system also needs:

- **Authenticated classical communication** so Eve cannot impersonate Alice or Bob.
- **Error correction** to reconcile the sifted keys without leaking too much information.
- **Privacy amplification** to compress away Eve's possible information.
- **Finite-key analysis** because real security claims use finite statistical samples.
- **Decoy states** in practical weak-pulse systems to expose photon-number-splitting attacks.
- **Loss, detector, source, timing, and device models**, including side-channel assumptions.
- **Composable security proofs and certified randomness**, plus careful key management.

The familiar 11% threshold is a teaching reference associated with ideal one-way BB84
security analyses. A deployed system's acceptance rule must come from its complete protocol,
finite-key proof, hardware model, and security parameters—not from this slider alone.
        """
    )

with tab_compare:
    st.info(
        "Mode 3 — Comparison: place the baseline and Eve-attack results side by side. "
        "Complete the QBER checkpoint in Modes 1 and 2 to unlock the charts."
    )
    if not (no_eve_qber_complete and eve_qber_complete):
        st.info(
            "Complete the QBER checkpoint in both the No Eve and Eve tabs to unlock "
            "the comparison. This prevents the chart from revealing the answers."
        )
        st.stop()

    st.subheader("What changes when Eve listens?")
    comparison = pd.DataFrame(
        {
            "Scenario": ["No Eve", f"Eve: {100 * eve_count / N_BITS:.1f}% intercepted"],
            "Sifted bits": [len(no_eve.sifted_alice_key), len(with_eve.sifted_alice_key)],
            "Public sample bits": [
                len(no_eve_estimate.sample_alice_bits),
                len(with_eve_estimate.sample_alice_bits),
            ],
            "Candidate key bits": [
                len(no_eve_estimate.remaining_alice_key),
                len(with_eve_estimate.remaining_alice_key),
            ],
            "Sample QBER (%)": [
                no_eve_estimate.sample_qber,
                with_eve_estimate.sample_qber,
            ],
            "Sample errors": [
                int(no_eve_estimate.sample_errors.sum()),
                int(with_eve_estimate.sample_errors.sum()),
            ],
        }
    )
    st.dataframe(comparison, hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Candidate key after public sampling")
        length_chart = (
            alt.Chart(comparison)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("Scenario:N", title=None),
                y=alt.Y("Candidate key bits:Q", scale=alt.Scale(domain=[0, N_BITS])),
                color=alt.Color("Scenario:N", legend=None),
                tooltip=["Scenario", "Candidate key bits", "Public sample bits"],
            )
        )
        st.altair_chart(length_chart, width="stretch")
        st.caption(
            "Publicly tested bits are removed in both scenarios. Eve usually changes the "
            "sample error rate rather than the number of surviving basis matches."
        )
    with right:
        st.markdown("#### Hidden full-key errors: noise versus Eve")
        error_data = pd.DataFrame(
            {
                "Cause": ["Channel noise", "Eve intercept-resend"],
                "Errors": [with_eve.noise_attributed_errors, with_eve.eve_attributed_errors],
            }
        )
        error_chart = (
            alt.Chart(error_data)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("Cause:N", title=None),
                y=alt.Y("Errors:Q", title="Final sifted-key errors"),
                color=alt.Color("Cause:N", legend=None),
                tooltip=["Cause", "Errors"],
            )
        )
        st.altair_chart(error_chart, width="stretch")
        st.caption(
            "This causal split is visible only inside the simulator. Real systems estimate "
            "total disturbance and use a security proof rather than identifying each cause."
        )

    st.markdown("#### Estimated sample QBER versus Eve interception percentage")
    sweep = qber_sweep(channel_noise, sample_fraction, int(seed))
    lines = (
        alt.Chart(sweep)
        .mark_line(point=True)
        .encode(
            x=alt.X("Eve interception (%):Q", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("QBER (%):Q", scale=alt.Scale(domain=[0, 30])),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["Simulated mean", "Theoretical expectation"],
                    range=["#2563eb", "#f59e0b"],
                ),
            ),
            strokeDash=alt.StrokeDash("Series:N"),
            tooltip=["Eve interception (%):Q", alt.Tooltip("QBER (%):Q", format=".2f"), "Series:N"],
        )
    )
    threshold_rule = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(color="#dc2626", strokeDash=[7, 5], size=2)
        .encode(y="threshold:Q", tooltip=[alt.Tooltip("threshold:Q", title="Abort threshold")])
    )
    st.altair_chart(lines + threshold_rule, width="stretch")
    st.caption(
        f"Blue averages 16 independent 1,000-bit exchanges using a {sample_percent}% public "
        "sample at each point. The orange line is the ideal expectation, including the "
        "selected channel noise; the dashed red line is the current abort threshold. "
        "Smaller finite samples fluctuate more widely around the expectation."
    )

st.divider()
st.caption(
    "Educational model only · 1,000 idealised single-photon transmissions · "
    "No real secret key is generated or stored"
)
