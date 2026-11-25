import numpy as np

from src.signal_generator import (
    generate_bpsk,
    generate_qpsk,
    generate_8psk,
    generate_16qam,
    add_awgn
)

rng = np.random.default_rng(42)

signals = {
    "BPSK": generate_bpsk(rng=rng),
    "QPSK": generate_qpsk(rng=rng),
    "8PSK": generate_8psk(rng=rng),
    "16QAM": generate_16qam(rng=rng)
}

for name, signal in signals.items():
    print(f"\n{name}")
    print("Number of symbols:", len(signal))
    print("Data type:", signal.dtype)
    print("Average power:", np.mean(np.abs(signal) ** 2))

    noisy_signal = add_awgn(signal, snr_db=10, rng=rng)

    print("Noisy signal power:", np.mean(np.abs(noisy_signal) ** 2))