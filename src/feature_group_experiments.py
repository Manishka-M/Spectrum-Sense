
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("data/processed/modulation_dataset.csv")

X = df.drop(columns=["modulation", "snr_db"])
y = df["modulation"]

# Reproduce the same split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Define feature groups
amplitude_features = [
    col for col in X.columns
    if col.startswith("amplitude_")
    or col in ["peak_amplitude", "papr"]
]

phase_features = [
    "phase_diff_sin_mean",
    "phase_diff_cos_mean",
    "phase_diff_std"
]

moment_features = [
    "moment_2",
    "moment_4",
    "moment_8"
]

experiments = {
    "Amplitude only": amplitude_features,

    "Phase difference only": phase_features,

    "Higher-order moments only": moment_features,

    "Phase + moments": phase_features + moment_features,

    "Amplitude + moments": amplitude_features + moment_features,

    "All features": list(X.columns)
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nFEATURE GROUP EXPERIMENTS")
print("-" * 60)

for name, selected_features in experiments.items():

    model = Pipeline([
        ("scaler", RobustScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=2,
            gamma="scale"
        ))
    ])

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    cv_scores = cross_val_score(
        model,
        X_train_selected,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    model.fit(X_train_selected, y_train)

    predictions = model.predict(X_test_selected)

    accuracy = accuracy_score(y_test, predictions)

    test_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print(f"\n{name}")
    print(f"Features: {len(selected_features)}")
    print(f"CV F1: {cv_scores.mean():.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
