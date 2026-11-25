
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset and trained model
df = pd.read_csv("data/processed/modulation_dataset.csv")
saved = joblib.load("models/best_model.joblib")

model = saved["model"]
feature_columns = saved["feature_columns"]

# Recreate the same test split
from sklearn.model_selection import train_test_split

X = df[feature_columns]
y = df["modulation"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Predict test samples
predictions = model.predict(X_test)

# Confusion matrix
labels = ["16QAM", "8PSK", "BPSK", "QPSK"]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("CONFUSION MATRIX")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(cmap="Blues", values_format="d")
plt.title("SpectrumSense - Confusion Matrix")
plt.tight_layout()
plt.savefig("models/detailed_confusion_matrix.png")
plt.show()

# Identify incorrect predictions
errors = pd.DataFrame({
    "actual": y_test.values,
    "predicted": predictions,
    "snr_db": df.loc[X_test.index, "snr_db"].values
})

incorrect = errors[errors["actual"] != errors["predicted"]]

print("\nTOTAL INCORRECT PREDICTIONS:", len(incorrect))

print("\nERRORS BY ACTUAL CLASS")
print(incorrect["actual"].value_counts())

print("\nERRORS BY PREDICTED CLASS")
print(incorrect["predicted"].value_counts())

print("\nERRORS BY SNR RANGE")

incorrect["snr_range"] = pd.cut(
    incorrect["snr_db"],
    bins=[-5, 0, 5, 10, 15, 20]
)

print(incorrect["snr_range"].value_counts().sort_index())

print("\nMOST COMMON CONFUSIONS")
print(
    incorrect.groupby(
        ["actual", "predicted"]
    ).size().sort_values(ascending=False)
)



# Detailed class-wise error analysis by SNR

print("\nCLASS-WISE ACCURACY BY SNR")

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": predictions,
    "snr_db": df.loc[X_test.index, "snr_db"].values
})

results["snr_range"] = pd.cut(
    results["snr_db"],
    bins=[-5, 0, 5, 10, 15, 20]
)

for modulation in ["16QAM", "8PSK", "BPSK", "QPSK"]:

    subset = results[results["actual"] == modulation]

    print(f"\n{modulation}")

    for snr_range, group in subset.groupby(
        "snr_range",
        observed=False
    ):
        if len(group) == 0:
            continue

        accuracy = (
            group["actual"] == group["predicted"]
        ).mean()

        print(
            f"{snr_range}: "
            f"{accuracy:.2%} "
            f"({len(group)} samples)"
        )


# Examine only QPSK and 16QAM confusion

print("\nQPSK AND 16QAM CONFUSION BY SNR")

confused = results[
    (
        (results["actual"] == "QPSK") &
        (results["predicted"] == "16QAM")
    ) |
    (
        (results["actual"] == "16QAM") &
        (results["predicted"] == "QPSK")
    )
]

print(
    confused.groupby(
        ["actual", "predicted", "snr_range"],
        observed=False
    ).size()
)
