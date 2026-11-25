import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("data/processed/modulation_dataset.csv")

feature_columns = [
    "amplitude_mean",
    "amplitude_std",
    "amplitude_skewness",
    "amplitude_kurtosis",
    "peak_amplitude",
    "papr",
    "moment_2",
    "moment_4",
    "moment_8",
]


# Recreate the same test split
X = df[feature_columns]
y = df["modulation"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

test_df = X_test.copy()
test_df["modulation"] = y_test.values
test_df["snr_db"] = df.loc[X_test.index, "snr_db"].values


# Select low-SNR samples
low_snr_df = test_df[test_df["snr_db"] <= 0]

print("Low-SNR samples:", len(low_snr_df))

print("\nMean feature values by modulation:")
print(
    low_snr_df.groupby("modulation")[feature_columns].mean().round(4)
)


# Plot feature distributions
for feature in feature_columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=low_snr_df,
        x="modulation",
        y=feature
    )

    plt.title(f"{feature} distribution at SNR <= 0 dB")
    plt.tight_layout()
    plt.show()