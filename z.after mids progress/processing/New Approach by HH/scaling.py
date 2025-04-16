import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Step 1: Load the dataset
input_file = "dataset_after_low_variance_removed.csv"
df = pd.read_csv(input_file)

# Step 2: Create a log for changes
log = []

# Step 3: Shift negative values to make them non-negative
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
shift_log = []

for col in numeric_cols:
    min_val = df[col].min()
    if pd.notnull(min_val) and min_val < 0:
        df[col] = df[col] + abs(min_val)
        shift_log.append(f"  ➤ Column '{col}': shifted by +{abs(min_val)}")
        log.append(f"↕️ Shifted column '{col}' to make values non-negative (min was {min_val})")

if not shift_log:
    log.append("✅ No negative values found to shift.")

# Step 4: Replace inf/-inf with NaN and drop them
inf_count = np.isinf(df[numeric_cols].values).sum()
if inf_count > 0:
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=numeric_cols, inplace=True)
    log.append(f"🧹 Removed {inf_count} infinity values by dropping rows containing them.")
else:
    log.append("✅ No infinity values found.")

# Step 5: Apply Min-Max Scaling to numeric columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
log.append(f"📏 Applied Min-Max scaling to numeric columns: {list(numeric_cols)}")

# Step 6: Round the scaled values to 4 decimal places
df[numeric_cols] = df[numeric_cols].round(4)
log.append("🔢 Rounded scaled values to 4 decimal places.")

# Step 7: Save the scaled dataset
output_file = "dataset_scaled.csv"
df.to_csv(output_file, index=False)
log.append(f"💾 Saved the cleaned and scaled dataset to '{output_file}'")

# Step 8: Save a change log
log_file = "scaling_log.txt"
with open(log_file, "w") as f:
    f.write("\n".join(log))

# Step 9: Display summary
print("🎉 Scaling completed successfully. Summary of changes:\n")
print("\n".join(log))
