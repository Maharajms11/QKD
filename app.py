"""Streamlit interface for the idealised BB84 teaching simulator."""

from __future__ import annotations

import secrets

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from bb84 import (
    calculate_qber,
    expected_qber,
    generate_bases,
    generate_bits,
    photon_state_labels,
    run_protocol,
)


N_BITS = 1000
SAMPLE_ROWS = 30


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


def sample_table(result, include_eve: bool = False) -> pd.DataFrame:
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
def qber_sweep(channel_noise: float, base_seed: int, trials: int = 16) -> pd.DataFrame:
    """Average repeated 1000-bit experiments for a readable teaching chart."""

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
            simulated_qbers.append(result.qber)
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


def metric_row(result, include_eve: bool = False) -> None:
    """Render the principal numbers for an exchange."""

    columns = st.columns(6 if include_eve else 5)
    columns[0].metric("Transmitted", f"{len(result.alice_bits):,}")
    columns[1].metric("Basis matches", f"{int(result.sift_mask.sum()):,}")
    columns[2].metric("Sifted key", f"{len(result.sifted_alice_key):,} bits")
    columns[3].metric("Sifted errors", f"{int(result.sifted_errors.sum()):,}")
    columns[4].metric("Observed QBER", f"{result.qber:.2f}%")
    if include_eve:
        columns[5].metric("Eve intercepted", f"{int(result.eve.intercepted.sum()):,}")


def detection_panel(qber: float, threshold: float) -> None:
    """Apply the user-selected classroom abort rule."""

    with st.expander("7. QBER-based detection", expanded=True):
        st.write(
            f"Alice and Bob compare a sample of their sifted key. This simulator "
            f"compares the complete sifted key for visibility. The abort threshold is "
            f"**{threshold:.1f}%**, and the observed QBER is **{qber:.2f}%**."
        )
        if qber > threshold:
            st.error("Eavesdropping or excessive channel disturbance detected: abort key.")
        else:
            st.success(
                "QBER below threshold: proceed to error correction and privacy amplification."
            )


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
        "Channel noise level",
        options=[0, 1, 2, 5, 10],
        horizontal=True,
        format_func=lambda value: f"{value}%",
        help="Independent probability that a measured bit is flipped.",
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
        "QBER abort threshold",
        min_value=0.0,
        max_value=25.0,
        value=11.0,
        step=0.5,
        format="%.1f%%",
    )
    st.caption("Controls update both scenarios immediately.")

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

tab_no_eve, tab_eve, tab_compare, tab_reality = st.tabs(
    ["No Eve", "Eve intercept-resend", "Compare and chart", "From classroom to real QKD"]
)

with tab_no_eve:
    st.subheader(f"Alice → noisy channel ({noise_percent}%) → Bob")
    metric_row(no_eve)
    with st.expander("4. Basis sifting", expanded=True):
        st.write(
            "After all measurements, Alice and Bob publicly compare bases—but never reveal "
            "their bit values. They keep only positions where their original bases match. "
            "About half of 1,000 positions should survive."
        )
        st.progress(len(no_eve.sifted_alice_key) / N_BITS, text=f"{len(no_eve.sifted_alice_key)} of 1,000 positions kept")
    st.markdown("#### First 30 transmitted positions")
    st.dataframe(sample_table(no_eve), hide_index=True, width="stretch", height=560)
    with st.expander("5. Error estimation", expanded=True):
        st.write(
            f"The sifted keys differ at **{int(no_eve.sifted_errors.sum())}** positions. "
            f"QBER = errors / sifted bits = {int(no_eve.sifted_errors.sum())} / "
            f"{len(no_eve.sifted_alice_key)} = **{no_eve.qber:.2f}%**. With no Eve, all "
            "simulated errors come from the selected channel noise."
        )
    detection_panel(no_eve.qber, threshold)

with tab_eve:
    st.subheader(
        f"Alice → Eve intercepts {eve_count:,} photons ({100 * eve_count / N_BITS:.1f}%) → Bob"
    )
    metric_row(with_eve, include_eve=True)
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

    st.markdown("#### First 30 transmitted positions, including Eve's actions")
    st.dataframe(sample_table(with_eve, include_eve=True), hide_index=True, width="stretch", height=560)

    with st.expander("6. Error estimation and attribution", expanded=True):
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
    detection_panel(with_eve.qber, threshold)

with tab_compare:
    st.subheader("What changes when Eve listens?")
    comparison = pd.DataFrame(
        {
            "Scenario": ["No Eve", f"Eve: {100 * eve_count / N_BITS:.1f}% intercepted"],
            "Sifted key length": [len(no_eve.sifted_alice_key), len(with_eve.sifted_alice_key)],
            "Observed QBER (%)": [no_eve.qber, with_eve.qber],
            "Total sifted errors": [int(no_eve.sifted_errors.sum()), int(with_eve.sifted_errors.sum())],
        }
    )
    st.dataframe(comparison, hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Sifted key length")
        length_chart = (
            alt.Chart(comparison)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("Scenario:N", title=None),
                y=alt.Y("Sifted key length:Q", scale=alt.Scale(domain=[0, N_BITS])),
                color=alt.Color("Scenario:N", legend=None),
                tooltip=["Scenario", "Sifted key length"],
            )
        )
        st.altair_chart(length_chart, width="stretch")
        st.caption(
            "Eve usually does not change the sifted length: sifting depends on Alice's and "
            "Bob's announced bases. Eve reveals herself mainly by increasing errors."
        )
    with right:
        st.markdown("#### Final errors: noise versus Eve")
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

    st.markdown("#### QBER versus Eve interception percentage")
    sweep = qber_sweep(channel_noise, int(seed))
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
        "Blue averages 16 independent 1,000-bit exchanges at each point. The orange line is "
        "the ideal expectation, including the selected channel noise; the dashed red line is "
        "the current abort threshold. Random finite samples fluctuate around the expectation."
    )

with tab_reality:
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

st.divider()
st.caption(
    "Educational model only · 1,000 idealised single-photon transmissions · "
    "No real secret key is generated or stored"
)
