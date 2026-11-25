
from src.signal_generator import (
    generate_bpsk,
    generate_qpsk,
    generate_8psk,
    generate_16qam,
    add_awgn
)

from src.predict import predict_modulation


def test_prediction():

    generators = {
        "BPSK": generate_bpsk,
        "QPSK": generate_qpsk,
        "8PSK": generate_8psk,
        "16QAM": generate_16qam
    }

    for modulation, generator in generators.items():

        signal = generator(n_symbols=256)

        noisy_signal = add_awgn(
            signal,
            snr_db=15
        )

        prediction = predict_modulation(noisy_signal)

        print(
            f"Actual: {modulation:6s} | "
            f"Predicted: {prediction}"
        )


if __name__ == "__main__":
    test_prediction()
