"""
Create Binary Classification Dataset
=====================================
Merge Low+Medium vs High for better separability
"""

import pandas as pd
import numpy as np

print("\n" + "="*70)
print("CREATING BINARY CLASSIFICATION DATASET")
print("="*70 + "\n")

# Load data
df = pd.read_csv('data/processed/causal_variables_final.csv')
print(f"Original size: {len(df)}")

# Convert to binary
# 0,1 → 0 (Low/Medium severity)
# 2 → 1 (High severity)
df['Accident_Severity_Binary'] = (df['Accident_Severity'] >= 2).astype(int)

# Show distribution
print("\nBinary class distribution:")
print(df['Accident_Severity_Binary'].value_counts())

# Sample 5K
df_binary = df.sample(n=5000, random_state=42)

# Save with binary target
df_binary_out = df_binary.drop('Accident_Severity', axis=1)
df_binary_out = df_binary_out.rename(columns={'Accident_Severity_Binary': 'Accident_Severity'})

df_binary_out.to_csv('data/processed/causal_variables_binary.csv', index=False)

print(f"\n✅ Saved binary dataset: {len(df_binary_out)} samples")
print("\nUpdate config:")
print('  causal_variables: "data/processed/causal_variables_binary.csv"')
print("\nAnd in configs/phase4_config.yaml:")
print('  model:')
print('    output_dim: 2  # Change from 3 to 2')
print("="*70 + "\n")
