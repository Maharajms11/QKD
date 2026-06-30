"""Core logic for an idealised BB84 teaching simulator.

This module models single photons as a bit plus a preparation basis.  It is
deliberately conceptual: it does not model optical loss, detector effects,
multi-photon pulses, timing, authentication, or finite-key security proofs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PLUS = "+"
CROSS = "x"
VALID_BASES = (PLUS, CROSS)


@dataclass(frozen=True)
class EveResult:
    """The photon stream after an intercept-resend attack."""

    intercepted: np.ndarray
    eve_bases: np.ndarray
    eve_bits: np.ndarray
    eve_basis_matches: np.ndarray
    resent_bits: np.ndarray
    resent_bases: np.ndarray
    resent_states: np.ndarray


@dataclass(frozen=True)
class BobResult:
    """Bob's raw measurement record, before basis sifting."""

    measured_bits_before_noise: np.ndarray
    measured_bits: np.ndarray
    receiver_basis_matches: np.ndarray
    noise_flips: np.ndarray


@dataclass(frozen=True)
class ProtocolResult:
    """Complete record and summary for one BB84 exchange."""

    alice_bits: np.ndarray
    alice_bases: np.ndarray
    alice_states: np.ndarray
    bob_bases: np.ndarray
    eve: EveResult
    bob: BobResult
    sift_mask: np.ndarray
    sifted_alice_key: np.ndarray
    sifted_bob_key: np.ndarray
    sifted_errors: np.ndarray
    qber: float
    noise_attributed_errors: int
    eve_attributed_errors: int
    eve_errors_before_noise: int
    eve_errors_cancelled_by_noise: int
    noise_flips_in_sifted_key: int


@dataclass(frozen=True)
class ParameterEstimationResult:
    """Public test sample and the unrevealed candidate key that remains."""

    sample_indices: np.ndarray
    sample_alice_bits: np.ndarray
    sample_bob_bits: np.ndarray
    sample_errors: np.ndarray
    sample_qber: float
    remaining_alice_key: np.ndarray
    remaining_bob_key: np.ndarray


def _rng(rng: np.random.Generator | None = None) -> np.random.Generator:
    """Return the supplied generator, or create a fresh one."""

    return rng if rng is not None else np.random.default_rng()


def generate_bits(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate *n* uniformly random classical bits."""

    if n < 0:
        raise ValueError("n must be non-negative")
    return _rng(rng).integers(0, 2, size=n, dtype=np.int8)


def generate_bases(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate *n* uniformly random BB84 bases, represented by + and x."""

    if n < 0:
        raise ValueError("n must be non-negative")
    return _rng(rng).choice(np.array(VALID_BASES), size=n)


def encode_photons(bits: np.ndarray, bases: np.ndarray) -> np.ndarray:
    """Encode BB84 bit/basis pairs as idealised polarization angles.

    Mapping: 0/+ -> 0 degrees, 1/+ -> 90 degrees,
    0/x -> 45 degrees, and 1/x -> 135 degrees.
    """

    bits = np.asarray(bits, dtype=np.int8)
    bases = np.asarray(bases)
    if bits.shape != bases.shape:
        raise ValueError("bits and bases must have the same shape")
    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError("bits must contain only 0 and 1")
    if not np.all(np.isin(bases, VALID_BASES)):
        raise ValueError("bases must contain only '+' and 'x'")

    return np.select(
        [
            (bits == 0) & (bases == PLUS),
            (bits == 1) & (bases == PLUS),
            (bits == 0) & (bases == CROSS),
            (bits == 1) & (bases == CROSS),
        ],
        [0, 90, 45, 135],
    ).astype(np.int16)


def photon_state_labels(states: np.ndarray) -> np.ndarray:
    """Return classroom-friendly labels for polarization angles."""

    labels = {
        0: "0 degrees - horizontal",
        90: "90 degrees - vertical",
        45: "45 degrees - diagonal",
        135: "135 degrees - anti-diagonal",
    }
    return np.array([labels[int(state)] for state in states], dtype=object)


def predictable_measurement_bit(
    received_bit: int,
    received_basis: str,
    measurement_basis: str,
) -> int | None:
    """Return the predictable bit, or ``None`` for a random measurement.

    In BB84 a measurement in the preparation basis returns the encoded bit.
    A measurement in the other basis can return either bit with equal
    probability, so asking a learner to guess one exact value would be
    misleading.
    """

    if received_bit not in (0, 1):
        raise ValueError("received_bit must be 0 or 1")
    if received_basis not in VALID_BASES:
        raise ValueError("received_basis must be '+' or 'x'")
    if measurement_basis not in VALID_BASES:
        raise ValueError("measurement_basis must be '+' or 'x'")
    return int(received_bit) if received_basis == measurement_basis else None


def calculate_qber_from_counts(error_count: int, sifted_count: int) -> float:
    """Calculate QBER percentage from classroom-visible aggregate counts."""

    if error_count < 0:
        raise ValueError("error_count must be non-negative")
    if sifted_count < 0:
        raise ValueError("sifted_count must be non-negative")
    if error_count > sifted_count:
        raise ValueError("error_count cannot exceed sifted_count")
    if sifted_count == 0:
        return 0.0
    return 100.0 * error_count / sifted_count


def sample_sifted_key(
    alice_key: np.ndarray,
    bob_key: np.ndarray,
    sample_fraction: float,
    rng: np.random.Generator | None = None,
) -> ParameterEstimationResult:
    """Reveal a random sifted-key sample and remove it from the candidate key.

    ``sample_indices`` are positions within the sifted key, not positions in the
    original transmitted stream. At least one bit is sampled when the sifted
    key is non-empty, while at least one bit is retained when possible.
    """

    alice_key = np.asarray(alice_key, dtype=np.int8)
    bob_key = np.asarray(bob_key, dtype=np.int8)
    if alice_key.shape != bob_key.shape:
        raise ValueError("sifted keys must have the same shape")
    if not 0.0 < sample_fraction < 1.0:
        raise ValueError("sample_fraction must be between 0 and 1")

    sifted_count = len(alice_key)
    if sifted_count == 0:
        sample_indices = np.array([], dtype=np.int64)
    elif sifted_count == 1:
        sample_indices = np.array([0], dtype=np.int64)
    else:
        sample_count = min(
            sifted_count - 1,
            max(1, round(sifted_count * sample_fraction)),
        )
        sample_indices = np.sort(
            _rng(rng).choice(sifted_count, size=sample_count, replace=False)
        )

    sample_mask = np.zeros(sifted_count, dtype=bool)
    sample_mask[sample_indices] = True
    sample_alice = alice_key[sample_mask]
    sample_bob = bob_key[sample_mask]
    sample_errors = sample_alice != sample_bob
    _, sample_qber = calculate_qber(sample_alice, sample_bob)

    return ParameterEstimationResult(
        sample_indices=sample_indices,
        sample_alice_bits=sample_alice,
        sample_bob_bits=sample_bob,
        sample_errors=sample_errors,
        sample_qber=sample_qber,
        remaining_alice_key=alice_key[~sample_mask],
        remaining_bob_key=bob_key[~sample_mask],
    )


def simulate_eve_intercept_resend(
    alice_bits: np.ndarray,
    alice_bases: np.ndarray,
    intercept_count: int,
    rng: np.random.Generator | None = None,
) -> EveResult:
    """Simulate Eve measuring exactly *intercept_count* photons and resending.

    Eve independently chooses + or x for each photon she intercepts.  A wrong
    basis gives her a random result.  She prepares a new photon from her result
    and her basis, destroying the original state.
    """

    alice_bits = np.asarray(alice_bits, dtype=np.int8)
    alice_bases = np.asarray(alice_bases)
    n = len(alice_bits)
    if alice_bits.shape != alice_bases.shape:
        raise ValueError("alice_bits and alice_bases must have the same shape")
    if not 0 <= intercept_count <= n:
        raise ValueError("intercept_count must be between 0 and n")

    rng = _rng(rng)
    intercepted = np.zeros(n, dtype=bool)
    if intercept_count:
        intercepted[rng.choice(n, size=intercept_count, replace=False)] = True

    eve_bases = np.full(n, "-", dtype="<U1")
    eve_bits = np.full(n, -1, dtype=np.int8)
    eve_basis_matches = np.zeros(n, dtype=bool)
    resent_bits = alice_bits.copy()
    resent_bases = alice_bases.copy()

    indices = np.flatnonzero(intercepted)
    if len(indices):
        chosen_bases = generate_bases(len(indices), rng)
        eve_bases[indices] = chosen_bases
        matches = chosen_bases == alice_bases[indices]
        eve_basis_matches[indices] = matches

        measured = generate_bits(len(indices), rng)
        measured[matches] = alice_bits[indices][matches]
        eve_bits[indices] = measured

        resent_bits[indices] = measured
        resent_bases[indices] = chosen_bases

    resent_states = encode_photons(resent_bits, resent_bases)
    return EveResult(
        intercepted=intercepted,
        eve_bases=eve_bases,
        eve_bits=eve_bits,
        eve_basis_matches=eve_basis_matches,
        resent_bits=resent_bits,
        resent_bases=resent_bases,
        resent_states=resent_states,
    )


def simulate_bob_measurement(
    received_bits: np.ndarray,
    received_bases: np.ndarray,
    bob_bases: np.ndarray,
    channel_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> BobResult:
    """Measure incoming photons and apply an independent bit-flip channel.

    A matching measurement basis returns the prepared bit.  A mismatched basis
    returns a uniformly random bit.  ``channel_noise`` is the probability that
    the resulting bit is flipped after measurement.
    """

    received_bits = np.asarray(received_bits, dtype=np.int8)
    received_bases = np.asarray(received_bases)
    bob_bases = np.asarray(bob_bases)
    if not (received_bits.shape == received_bases.shape == bob_bases.shape):
        raise ValueError("received bits, received bases, and Bob bases must align")
    if not 0.0 <= channel_noise <= 1.0:
        raise ValueError("channel_noise must be between 0 and 1")

    rng = _rng(rng)
    receiver_basis_matches = bob_bases == received_bases
    before_noise = generate_bits(len(received_bits), rng)
    before_noise[receiver_basis_matches] = received_bits[receiver_basis_matches]

    noise_flips = rng.random(len(received_bits)) < channel_noise
    measured_bits = np.bitwise_xor(before_noise, noise_flips.astype(np.int8))
    return BobResult(
        measured_bits_before_noise=before_noise,
        measured_bits=measured_bits,
        receiver_basis_matches=receiver_basis_matches,
        noise_flips=noise_flips,
    )


def sift_key(
    alice_bits: np.ndarray,
    alice_bases: np.ndarray,
    bob_bits: np.ndarray,
    bob_bases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep positions where Alice and Bob publicly announce matching bases."""

    alice_bits = np.asarray(alice_bits, dtype=np.int8)
    alice_bases = np.asarray(alice_bases)
    bob_bits = np.asarray(bob_bits, dtype=np.int8)
    bob_bases = np.asarray(bob_bases)
    if not (alice_bits.shape == alice_bases.shape == bob_bits.shape == bob_bases.shape):
        raise ValueError("all sifting arrays must have the same shape")
    keep = alice_bases == bob_bases
    return keep, alice_bits[keep], bob_bits[keep]


def calculate_qber(alice_key: np.ndarray, bob_key: np.ndarray) -> tuple[int, float]:
    """Return the error count and quantum bit error rate as a percentage."""

    alice_key = np.asarray(alice_key)
    bob_key = np.asarray(bob_key)
    if alice_key.shape != bob_key.shape:
        raise ValueError("sifted keys must have the same shape")
    if len(alice_key) == 0:
        return 0, 0.0
    errors = int(np.count_nonzero(alice_key != bob_key))
    return errors, 100.0 * errors / len(alice_key)


def run_protocol(
    alice_bits: np.ndarray,
    alice_bases: np.ndarray,
    bob_bases: np.ndarray,
    channel_noise: float,
    eve_intercept_count: int,
    rng: np.random.Generator | None = None,
) -> ProtocolResult:
    """Run one transparent BB84 exchange and retain every intermediate value."""

    rng = _rng(rng)
    alice_bits = np.asarray(alice_bits, dtype=np.int8)
    alice_bases = np.asarray(alice_bases)
    bob_bases = np.asarray(bob_bases)
    alice_states = encode_photons(alice_bits, alice_bases)

    eve = simulate_eve_intercept_resend(
        alice_bits, alice_bases, eve_intercept_count, rng
    )
    bob = simulate_bob_measurement(
        eve.resent_bits,
        eve.resent_bases,
        bob_bases,
        channel_noise,
        rng,
    )
    sift_mask, sifted_alice, sifted_bob = sift_key(
        alice_bits, alice_bases, bob.measured_bits, bob_bases
    )
    sifted_errors = sifted_alice != sifted_bob
    _, qber = calculate_qber(sifted_alice, sifted_bob)

    # This causal bookkeeping is possible because this is a simulation.  In a
    # real protocol Alice and Bob observe only aggregate errors; they cannot
    # label an individual error as "Eve" or "noise".
    error_before_noise = bob.measured_bits_before_noise != alice_bits
    final_error = bob.measured_bits != alice_bits
    eve_error_before_noise = sift_mask & eve.intercepted & error_before_noise
    noise_attributed = sift_mask & final_error & bob.noise_flips & ~error_before_noise
    eve_attributed = sift_mask & final_error & eve.intercepted & error_before_noise & ~bob.noise_flips
    cancelled = sift_mask & eve.intercepted & error_before_noise & bob.noise_flips

    return ProtocolResult(
        alice_bits=alice_bits,
        alice_bases=alice_bases,
        alice_states=alice_states,
        bob_bases=bob_bases,
        eve=eve,
        bob=bob,
        sift_mask=sift_mask,
        sifted_alice_key=sifted_alice,
        sifted_bob_key=sifted_bob,
        sifted_errors=sifted_errors,
        qber=qber,
        noise_attributed_errors=int(np.count_nonzero(noise_attributed)),
        eve_attributed_errors=int(np.count_nonzero(eve_attributed)),
        eve_errors_before_noise=int(np.count_nonzero(eve_error_before_noise)),
        eve_errors_cancelled_by_noise=int(np.count_nonzero(cancelled)),
        noise_flips_in_sifted_key=int(np.count_nonzero(sift_mask & bob.noise_flips)),
    )


def expected_qber(interception_fraction: float, channel_noise: float) -> float:
    """Expected QBER percentage for ideal intercept-resend plus bit-flip noise.

    Intercept-resend contributes p/4 before channel noise.  Independent binary
    flips can occasionally cancel, so the combined probability is
    e + p/4 - 2*e*(p/4).
    """

    if not 0.0 <= interception_fraction <= 1.0:
        raise ValueError("interception_fraction must be between 0 and 1")
    if not 0.0 <= channel_noise <= 1.0:
        raise ValueError("channel_noise must be between 0 and 1")
    eve_error = interception_fraction / 4.0
    return 100.0 * (channel_noise + eve_error - 2.0 * channel_noise * eve_error)
