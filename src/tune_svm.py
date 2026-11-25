
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("data/processed/modulation_dataset.csv")

X = df.drop(columns=["modulation", "snr_db"])
y = df["modulation"]

# Preserve the existing train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Pipeline prevents scaling leakage across CV folds
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("svm", SVC(kernel="rbf"))
])

# Hyperparameter search space
param_grid = {
    "svm__C": [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
    "svm__gamma": [
        "scale",
        "auto",
        0.001,
        0.01,
        0.1,
        1
    ]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("\nStarting SVM hyperparameter optimization...")

search.fit(X_train, y_train)

print("\nBEST PARAMETERS")
print(search.best_params_)

print("\nBEST CROSS-VALIDATION F1")
print(search.best_score_)

# Evaluate the selected model once on the test set
best_model = search.best_estimator_

test_accuracy = best_model.score(X_test, y_test)

print("\nFINAL TEST ACCURACY")
print(test_accuracy)

from sklearn.metrics import classification_report

predictions = best_model.predict(X_test)

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        predictions
    )
)
