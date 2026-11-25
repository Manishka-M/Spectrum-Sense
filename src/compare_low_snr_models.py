
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv("data/processed/modulation_dataset.csv")

all_features = [
    "amplitude_mean",
    "amplitude_std",
    "amplitude_skewness",
    "amplitude_kurtosis",
    "peak_amplitude",
    "papr",
    "phase_diff_sin_mean",
    "phase_diff_cos_mean",
    "phase_diff_std",
    "moment_2",
    "moment_4",
    "moment_8",
]

reduced_features = [
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

X = df[all_features]
y = df["modulation"]
snr = df["snr_db"]

X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
    X, y, snr,
    test_size=0.2,
    random_state=42,
    stratify=y
)

low_snr_mask = snr_test <= 0

X_low = X_test.loc[low_snr_mask]
y_low = y_test.loc[low_snr_mask]

models = {
    "9-feature model": (
        reduced_features,
        5,
        "auto"
    ),
    "12-feature model": (
        all_features,
        20,
        "scale"
    )
}

for name, (features, C, gamma) in models.items():

    model = Pipeline([
        ("scaler", RobustScaler()),
        ("svm", SVC(C=C, gamma=gamma))
    ])

    model.fit(X_train[features], y_train)

    predictions = model.predict(X_low[features])

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("Low-SNR accuracy:",
          accuracy_score(y_low, predictions))

    print(classification_report(
        y_low,
        predictions,
        zero_division=0
    ))
