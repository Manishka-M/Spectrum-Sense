
import numpy as np
import matplotlib.pyplot as plt

from src.signal_generator import (
    generate_bpsk,
    generate_qpsk,
    generate_8psk,
    generate_16qam,
    add_awgn
)

rng = np.random.default_rng(42)

generators = {
    "BPSK": generate_bpsk,
    "QPSK": generate_qpsk,
    "8PSK": generate_8psk,
    "16-QAM": generate_16qam
}

fig, axes = plt.subplots(2, 4, figsize=(15, 7))

for column, (name, generator) in enumerate(generators.items()):

    # Generate clean signal
    clean_signal = generator(n_symbols=500, rng=rng)

    # Add noise at 15 dB SNR
    noisy_signal = add_awgn(clean_signal, snr_db=15, rng=rng)

    # Plot clean constellation
    axes[0, column].scatter(
        clean_signal.real,
        clean_signal.imag,
        s=12,
        alpha=0.7
    )

    axes[0, column].set_title(f"{name} - Clean")
    axes[0, column].set_xlabel("In-phase (I)")
    axes[0, column].set_ylabel("Quadrature (Q)")
    axes[0, column].grid(True)
    axes[0, column].axis("equal")

    # Plot noisy constellation
    axes[1, column].scatter(
        noisy_signal.real,
        noisy_signal.imag,
        s=12,
        alpha=0.7
    )

    axes[1, column].set_title(f"{name} - 15 dB SNR")
    axes[1, column].set_xlabel("In-phase (I)")
    axes[1, column].set_ylabel("Quadrature (Q)")
    axes[1, column].grid(True)
    axes[1, column].axis("equal")

plt.tight_layout()
plt.show()
