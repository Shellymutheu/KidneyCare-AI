from ucimlrepo import fetch_ucirepo
import pandas as pd

print("Downloading Chronic Kidney Disease dataset from UCI...")

# UCI Chronic Kidney Disease dataset
ckd = fetch_ucirepo(id=336)

# Get features and target
X = ckd.data.features
y = ckd.data.targets

# Combine into one dataframe
df = pd.concat([X, y], axis=1)

# Clean column names
df.columns = [
    str(column).strip().lower().replace(" ", "_")
    for column in df.columns
]

# Save dataset
output_path = "data/kidney_disease.csv"
df.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("\nColumns:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df.iloc[:, -1].value_counts())