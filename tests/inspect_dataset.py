import pandas as pd

df = pd.read_csv("data/processed/modulation_dataset.csv")

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["modulation"].value_counts())