
import numpy as np
from scipy.signal import upfirdn


def _get_rng(rng=None):
    """Return a NumPy random number generator."""
    if rng is None:
        rng = np.random.default_rng()
    return rng


def generate_bpsk(n_symbols=256, rng=None, phase_offset=0.0):
    """Generate BPSK complex symbols."""
    rng = _get_rng(rng)

    bits = rng.integers(0, 2, n_symbols)

    # Map bit 0 to -1 and bit 1 to +1
    symbols = 2 * bits - 1

    # Convert to complex baseband representation
    return symbols.astype(complex) * np.exp(1j * phase_offset)


def generate_qpsk(n_symbols=256, rng=None, phase_offset=0.0):
    """Generate QPSK complex symbols."""
    rng = _get_rng(rng)

    symbol_indices = rng.integers(0, 4, n_symbols)

    # Four equally spaced phase states, offset by 45 degrees
    phases = np.pi / 4 + symbol_indices * (2 * np.pi / 4)

    symbols = np.exp(1j * phases)

    return symbols * np.exp(1j * phase_offset)


def generate_8psk(n_symbols=256, rng=None, phase_offset=0.0):
    """Generate 8PSK complex symbols."""
    rng = _get_rng(rng)

    symbol_indices = rng.integers(0, 8, n_symbols)

    # Eight equally spaced phase states
    phases = symbol_indices * (2 * np.pi / 8)

    symbols = np.exp(1j * phases)

    return symbols * np.exp(1j * phase_offset)


def generate_16qam(n_symbols=256, rng=None, phase_offset=0.0):
    """Generate 16-QAM complex symbols."""
    rng = _get_rng(rng)

    levels = np.array([-3, -1, 1, 3])

    i_values = rng.choice(levels, n_symbols)
    q_values = rng.choice(levels, n_symbols)

    # Normalize average symbol power to approximately 1
    symbols = (i_values + 1j * q_values) / np.sqrt(10)

    return symbols * np.exp(1j * phase_offset)



def generate_rrc_filter(
    samples_per_symbol=4,
    rolloff=0.35,
    span_symbols=10
):
    """
    Generate Root Raised Cosine (RRC) filter coefficients.

    samples_per_symbol:
        Number of samples representing each symbol.

    rolloff:
        Excess bandwidth factor (0 to 1).

    span_symbols:
        Total filter length expressed in symbol durations.
    """

    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive.")

    if not 0 <= rolloff <= 1:
        raise ValueError("rolloff must be between 0 and 1.")

    if span_symbols <= 0:
        raise ValueError("span_symbols must be positive.")

    sps = samples_per_symbol
    beta = rolloff

    # Time values measured in symbol periods
    t = np.arange(
        -span_symbols * sps // 2,
        span_symbols * sps // 2 + 1
    ) / sps

    h = np.zeros_like(t, dtype=float)

    for i, time in enumerate(t):

        if np.isclose(time, 0.0):
            h[i] = 1 + beta * (4 / np.pi - 1)

        elif beta > 0 and np.isclose(
            abs(time), 1 / (4 * beta)
        ):
            h[i] = (
                beta / np.sqrt(2)
                * (
                    (1 + 2 / np.pi)
                    * np.sin(np.pi / (4 * beta))
                    +
                    (1 - 2 / np.pi)
                    * np.cos(np.pi / (4 * beta))
                )
            )

        else:
            numerator = (
                np.sin(np.pi * time * (1 - beta))
                +
                4 * beta * time
                * np.cos(np.pi * time * (1 + beta))
            )

            denominator = (
                np.pi * time
                * (1 - (4 * beta * time) ** 2)
            )

            h[i] = numerator / denominator

    # Normalize filter energy
    h = h / np.sqrt(np.sum(h ** 2))

    return h



def add_awgn(signal, snr_db, rng=None):
    """Add complex Additive White Gaussian Noise."""
    rng = _get_rng(rng)

    signal_power = np.mean(np.abs(signal) ** 2)

    if signal_power == 0:
        raise ValueError("Signal power must be greater than zero.")

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    # Each component receives half the total complex noise power
    noise = np.sqrt(noise_power / 2) * (
        rng.normal(size=signal.shape)
        + 1j * rng.normal(size=signal.shape)
    )

    return signal + noise




def pulse_shape(
    symbols,
    samples_per_symbol=4,
    rolloff=0.35,
    span_symbols=10
):
    """
    Convert modulation symbols into an oversampled,
    pulse-shaped complex baseband waveform.
    """

    if not isinstance(symbols, np.ndarray):
        symbols = np.asarray(symbols)

    if symbols.ndim != 1 or len(symbols) == 0:
        raise ValueError("symbols must be a non-empty 1D array.")

    h = generate_rrc_filter(
        samples_per_symbol=samples_per_symbol,
        rolloff=rolloff,
        span_symbols=span_symbols
    )

    waveform = upfirdn(
        h,
        symbols,
        up=samples_per_symbol
    )

    return waveform



import numpy as np

from src.signal_generator import (
    generate_qpsk,
    generate_rrc_filter,
    pulse_shape
)


def test_rrc_filter():

    h = generate_rrc_filter(
        samples_per_symbol=4,
        rolloff=0.35,
        span_symbols=10
    )

    assert len(h) == 41
    assert np.all(np.isfinite(h))
    assert np.isclose(np.sum(h ** 2), 1.0)

    print("RRC filter test passed.")


def test_pulse_shape():

    symbols = generate_qpsk(n_symbols=256)

    waveform = pulse_shape(
        symbols,
        samples_per_symbol=4,
        rolloff=0.35,
        span_symbols=10
    )

    expected_length = (len(symbols) - 1) * 4 + 41

    assert len(waveform) == expected_length
    assert np.iscomplexobj(waveform)
    assert np.all(np.isfinite(waveform))

    print("Pulse shaping test passed.")
    print("Input symbols:", len(symbols))
    print("Output samples:", len(waveform))


if __name__ == "__main__":
    test_rrc_filter()
    test_pulse_shape()
