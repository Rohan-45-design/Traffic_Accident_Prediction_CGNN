"""
Create TRULY Balanced Dataset (50/50)
======================================
Equal samples from each class
"""

import pandas as pd
from pathlib import Path

print("\n" + "="*70)
print("CREATING TRULY BALANCED DATASET (50/50)")
print("="*70 + "\n")

# Load binary data
df = pd.read_csv('data/processed/causal_variables_binary.csv')

print(f"Original: {len(df)} samples")
print(df['Accident_Severity'].value_counts())

# Find minimum class size
class_counts = df['Accident_Severity'].value_counts()
min_samples = min(class_counts)

print(f"\n✅ Will sample {min_samples} from each class (50/50 balance)")

# Sample equally from both classes
df_class0 = df[df['Accident_Severity'] == 0].sample(n=min_samples, random_state=42)
df_class1 = df[df['Accident_Severity'] == 1].sample(n=min_samples, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([df_class0, df_class1]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Balanced dataset: {len(df_balanced)} samples")
print(df_balanced['Accident_Severity'].value_counts())

for cls in [0, 1]:
    pct = (df_balanced['Accident_Severity'] == cls).sum() / len(df_balanced) * 100
    print(f"  Class {cls}: {pct:.1f}%")

# Save
df_balanced.to_csv('data/processed/causal_variables_balanced_5k.csv', index=False)
print(f"\n✅ Saved: causal_variables_balanced_5k.csv")
print("\nUpdate config:")
print('  causal_variables: "data/processed/causal_variables_balanced_5k.csv"')
print("="*70 + "\n")
