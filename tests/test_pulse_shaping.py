
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
