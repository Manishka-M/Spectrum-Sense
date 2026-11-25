import os
import numpy as np
import pandas as pd
from src.signal_generator import (
    generate_bpsk,
    generate_qpsk,
    generate_8psk,
    generate_16qam,
    add_awgn
)

from src.feature_extractor import extract_features


def build_dataset(samples_per_class=1000):

    rng = np.random.default_rng(42)

    generators = {
        "BPSK": generate_bpsk,
        "QPSK": generate_qpsk,
        "8PSK": generate_8psk,
        "16QAM": generate_16qam
    }

    dataset = []

    for modulation, generator in generators.items():

        print(f"Generating {modulation} signals...")

        for _ in range(samples_per_class):

            # Random phase offset between 0 and 2*pi
            phase_offset = rng.uniform(0, 2 * np.pi)

            # Random SNR between -5 dB and 20 dB
            snr_db = rng.uniform(-5, 20)

            # Generate an independent signal burst
            signal = generator(
                n_symbols=256,
                rng=rng,
                phase_offset=phase_offset
            )

            # Add AWGN
            noisy_signal = add_awgn(
                signal,
                snr_db=snr_db,
                rng=rng
            )

            # Extract numerical features
            features = extract_features(noisy_signal)

            # Store features and labels
            features["modulation"] = modulation
            features["snr_db"] = snr_db

            dataset.append(features)

    # Convert list of dictionaries into a DataFrame
    df = pd.DataFrame(dataset)

    # Create output directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    # Save dataset
    output_path = "data/processed/modulation_dataset.csv"
    df.to_csv(output_path, index=False)

    print("\nDataset generated successfully!")
    print("Dataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["modulation"].value_counts())

    print("\nDataset saved at:", output_path)

    return df


if __name__ == "__main__":
    build_dataset()