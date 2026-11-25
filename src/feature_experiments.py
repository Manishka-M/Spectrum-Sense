
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("data/processed/modulation_dataset.csv")

X = df.drop(columns=["modulation", "snr_db"])
y = df["modulation"]

# Same reproducible train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Define feature experiments
experiments = {
    "All features": list(X.columns),

    "Without moment_8": [
        col for col in X.columns
        if col != "moment_8"
    ],

    "Amplitude features only": [
        col for col in X.columns
        if col.startswith("amplitude_")
        or col in ["peak_amplitude", "papr"]
    ]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nFEATURE ABLATION EXPERIMENT")
print("-" * 55)

for name, selected_features in experiments.items():

    X_train_subset = X_train[selected_features]
    X_test_subset = X_test[selected_features]

    model = Pipeline([
        ("scaler", RobustScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=2,
            gamma="scale"
        ))
    ])

    scores = cross_val_score(
        model,
        X_train_subset,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    model.fit(X_train_subset, y_train)

    predictions = model.predict(X_test_subset)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    print(f"\nExperiment: {name}")
    print(f"Number of features: {len(selected_features)}")
    print(f"CV Mean F1: {scores.mean():.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {f1:.4f}")
