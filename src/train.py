
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

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
    "moment_8"
]

X = df[feature_columns]
y = df["modulation"]
snr_values = df["snr_db"]

labels = sorted(y.unique())


# --------------------------------------------------
# 2. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test, snr_train, snr_test = (
    train_test_split(
        X,
        y,
        snr_values,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 3. DEFINE SELECTED SVM MODEL
# --------------------------------------------------

model = Pipeline([
    ("scaler", RobustScaler()),
    ("classifier", SVC(
        kernel="rbf",
        C=5,
        gamma="auto"
    ))
])


# --------------------------------------------------
# 4. TRAIN MODEL
# --------------------------------------------------

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# --------------------------------------------------
# 5. EVALUATE MODEL
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

print("\nFINAL TEST RESULTS")
print("-" * 50)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Macro F1:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=labels,
    zero_division=0
))


# --------------------------------------------------
# 6. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

os.makedirs("models", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(cmap="Blues", values_format="d")
plt.title("SpectrumSense - SVM Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.close()


# --------------------------------------------------
# 7. SNR-WISE PERFORMANCE
# --------------------------------------------------

test_results = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": y_pred,
    "snr_db": snr_test.to_numpy()
})

test_results["snr_range"] = pd.cut(
    test_results["snr_db"],
    bins=[-5.000001, 0, 5, 10, 15, 20.000001],
    labels=["<=0 dB", "0-5 dB", "5-10 dB", "10-15 dB", ">15 dB"]
)

snr_accuracy = test_results.groupby(
    "snr_range",
    observed=False
).apply(
    lambda group: accuracy_score(
        group["actual"],
        group["predicted"]
    ) if len(group) > 0 else float("nan")
)

print("\nACCURACY BY SNR RANGE")
print(snr_accuracy)


plt.figure(figsize=(9, 5))

plt.plot(
    range(len(snr_accuracy)),
    snr_accuracy.values,
    marker="o"
)

plt.xticks(
    range(len(snr_accuracy)),
    snr_accuracy.index.astype(str)
)

plt.xlabel("SNR Range (dB)")
plt.ylabel("Classification Accuracy")
plt.title("SpectrumSense - SNR Robustness")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig("models/snr_robustness.png", dpi=150)
plt.close()


# --------------------------------------------------
# 8. SAVE MODEL AND METADATA
# --------------------------------------------------

model_artifact = {
    "model": model,
    "feature_columns": feature_columns,
    "classes": labels,
    "model_name": "RBF-SVM",
    "hyperparameters": {
        "C": 5,
        "gamma": "auto",
        "kernel": "rbf"
    },
    "metrics": {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1
    }
}

joblib.dump(
    model_artifact,
    "models/best_model.joblib"
)

print("\nModel saved successfully!")
print("Location: models/best_model.joblib")
print("Feature count:", len(feature_columns))
print("Model: RBF-SVM (C=5, gamma=auto)")
