
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("data/processed/modulation_dataset.csv")

X = df.drop(columns=["modulation", "snr_db"])
y = df["modulation"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

amplitude_features = [
    col for col in X.columns
    if col.startswith("amplitude_")
    or col in ["peak_amplitude", "papr"]
]

moment_features = [
    "moment_2",
    "moment_4",
    "moment_8"
]

feature_sets = {
    "All 12 features": list(X.columns),

    "Amplitude + moments": (
        amplitude_features + moment_features
    )
}

param_grid = {
    "svm__C": [0.5, 1, 2, 5, 10, 20, 50],
    "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, features in feature_sets.items():

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("svm", SVC(kernel="rbf"))
    ])

    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1
    )

    search.fit(X_train[features], y_train)

    predictions = search.predict(X_test[features])

    print("Best parameters:", search.best_params_)
    print("CV F1:", search.best_score_)

    print(
        "Test accuracy:",
        accuracy_score(y_test, predictions)
    )

    print(
        "Test macro F1:",
        f1_score(y_test, predictions, average="macro")
    )
