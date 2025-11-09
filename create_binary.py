import pandas as pd
import numpy as np

print("Creating binary classification dataset...")

# Load original data
df = pd.read_csv('data/processed/causal_variables_final.csv')

# Create binary target: 0-1 → 0 (Non-severe), 2 → 1 (Severe)
df['Accident_Severity'] = (df['Accident_Severity'] >= 2).astype(int)

# Sample 5K
df_5k = df.sample(n=5000, random_state=42)

# Save
df_5k.to_csv('data/processed/causal_variables_binary.csv', index=False)

print(f"✅ Created binary dataset: {len(df_5k)} samples")
print(f"Distribution:\n{df_5k['Accident_Severity'].value_counts()}")
