import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report


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

X = df[feature_columns]
y = df["modulation"]
snr = df["snr_db"]


# Same train-test split as earlier
X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
    X,
    y,
    snr,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train selected model
model = Pipeline([
    ("scaler", RobustScaler()),
    ("svm", SVC(C=5, gamma="auto"))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# Analyse performance at different SNR levels
test_results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": predictions,
    "snr": snr_test.values
})

bins = [-float("inf"), 0, 5, 10, 15, float("inf")]
labels = ["<=0 dB", "0-5 dB", "5-10 dB", "10-15 dB", ">15 dB"]

test_results["snr_group"] = pd.cut(
    test_results["snr"],
    bins=bins,
    labels=labels
)


for group in labels:

    subset = test_results[
        test_results["snr_group"] == group
    ]

    print("\n" + "=" * 50)
    print("SNR GROUP:", group)
    print("Samples:", len(subset))

    if len(subset) == 0:
        continue

    print(
        classification_report(
            subset["actual"],
            subset["predicted"],
            zero_division=0
        )
    )