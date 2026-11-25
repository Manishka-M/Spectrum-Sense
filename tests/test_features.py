import numpy as np

from src.signal_generator import (
    generate_bpsk,
    generate_qpsk,
    generate_8psk,
    generate_16qam,
    add_awgn
)

from src.feature_extractor import extract_features


rng = np.random.default_rng(42)

generators = {
    "BPSK": generate_bpsk,
    "QPSK": generate_qpsk,
    "8PSK": generate_8psk,
    "16QAM": generate_16qam
}

for name, generator in generators.items():

    signal = generator(n_symbols=256, rng=rng)

    noisy_signal = add_awgn(signal, snr_db=10, rng=rng)

    features = extract_features(noisy_signal)

    print(f"\n{name}")
    for feature_name, value in features.items():
        print(f"{feature_name}: {value:.4f}")

    assert len(features) == 12
    assert all(np.isfinite(value) for value in features.values())

print("\nAll feature extraction tests passed!")