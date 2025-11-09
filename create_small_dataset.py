"""
Create Small Dataset for Testing
=================================
Use only 5,000 samples instead of 132,000
"""

import pandas as pd

# Load full dataset
df = pd.read_csv('data/processed/causal_variables_final.csv')

print(f"Original size: {len(df)} samples")

# Take random sample
df_small = df.sample(n=5000, random_state=42)

print(f"New size: {len(df_small)} samples")

# Save small version
df_small.to_csv('data/processed/causal_variables_small.csv', index=False)

print("✅ Small dataset created: data/processed/causal_variables_small.csv")
