"""
Expand Feature Set - Add All Predictive Features
=================================================
Reads global_road_accidents.csv, encodes categoricals, saves full feature set
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("\n" + "="*70)
print("EXPANDING FEATURE SET FOR CGNN")
print("="*70 + "\n")

# ===== STEP 1: LOAD DATA =====
df_original = pd.read_csv('data/raw/global_road_accidents_dataset.csv')
print(f"✅ Loaded original dataset: {df_original.shape}")
print(f"   Columns: {list(df_original.columns)[:10]}... (showing first 10)\n")

# ===== STEP 2: DEFINE TARGET AND FEATURES =====
target_col = 'Accident Severity'

# Exclude non-predictive columns
exclude_cols = [
    'Country', 'Year', 'Month',  # Administrative metadata
    'Number of Injuries', 'Number of Fatalities',  # Outcomes (leak target info)
    'Insurance Claims', 'Medical Cost', 'Economic Loss',  # Post-accident outcomes
    'Region'  # Redundant with other location features
]

# Select all predictive features
all_features = [col for col in df_original.columns 
                if col not in exclude_cols and col != target_col]

print(f"📊 Selected {len(all_features)} predictive features")
print("   Features:")
for i, feat in enumerate(all_features, 1):
    print(f"   {i:2d}. {feat}")
print()

# ===== STEP 3: EXTRACT AND CLEAN =====
selected_cols = all_features + [target_col]
df_full = df_original[selected_cols].copy()

# Handle missing values
print(f"🔧 Cleaning data...")
before_rows = len(df_full)
df_full = df_full.dropna()
after_rows = len(df_full)
print(f"   Rows before: {before_rows:,}")
print(f"   Rows after:  {after_rows:,}")
print(f"   Dropped:     {before_rows - after_rows:,}\n")

# ===== STEP 4: IDENTIFY COLUMN TYPES =====
numeric_cols = df_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df_full.select_dtypes(include=['object']).columns.tolist()

# Remove target from lists if present
if target_col in numeric_cols:
    numeric_cols.remove(target_col)
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

print("="*70)
print("COLUMN TYPE ANALYSIS")
print("="*70)
print(f"\n✅ Numeric columns ({len(numeric_cols)}) - will be kept as-is:")
for col in numeric_cols:
    print(f"   - {col:40s} | dtype: {df_full[col].dtype}")

print(f"\n🔧 Categorical columns ({len(categorical_cols)}) - will be label encoded:")
for col in categorical_cols:
    unique = df_full[col].nunique()
    print(f"   - {col:40s} | {unique} unique values")

# ===== STEP 5: ENCODE CATEGORICAL COLUMNS =====
print("\n" + "="*70)
print("ENCODING CATEGORICAL FEATURES")
print("="*70 + "\n")

le = LabelEncoder()
for col in categorical_cols:
    unique_before = df_full[col].nunique()
    print(f"Encoding: {col:40s}", end="")
    df_full[col] = le.fit_transform(df_full[col].astype(str))
    print(f" → {unique_before} categories → integers 0-{unique_before-1}")

# Encode target if categorical
if df_full[target_col].dtype == 'object':
    print(f"\n🎯 Encoding target: {target_col}")
    unique_targets = df_full[target_col].nunique()
    df_full[target_col] = le.fit_transform(df_full[target_col].astype(str))
    print(f"   → {unique_targets} severity levels → integers 0-{unique_targets-1}")

# ===== STEP 6: VERIFY AND SAVE =====
print("\n" + "="*70)
print("FINAL DATASET SUMMARY")
print("="*70)
print(f"\n✅ Shape: {df_full.shape}")
print(f"✅ Features: {df_full.shape[1] - 1}")
print(f"✅ Samples: {df_full.shape[0]:,}")
print(f"✅ Target classes: {df_full[target_col].nunique()}")

print("\n📊 Data types after encoding:")
print(df_full.dtypes)

print("\n📊 First 3 rows:")
print(df_full.head(3))

print("\n📊 Class distribution:")
print(df_full[target_col].value_counts().sort_index())

# Save
output_file = 'data/processed/causal_variables_full_features.csv'
df_full.to_csv(output_file, index=False)

print("\n" + "="*70)
print(f"✅ SAVED: {output_file}")
print("="*70)

print(f"\n📊 Feature expansion complete:")
print(f"   Old: 6 features")
print(f"   New: {df_full.shape[1] - 1} features")
print(f"   Gain: +{df_full.shape[1] - 7} features")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("\n1. Update configs/phase4_config.yaml:")
print(f'   causal_variables: "data/processed/causal_variables_full_features.csv"')
print(f'   input_dim: {df_full.shape[1] - 1}')
print("\n2. Clear old model checkpoints:")
print("   Remove-Item -Path 'data\\neural_models\\checkpoints\\*' -Force")
print("\n3. Retrain model:")
print("   python main_phase4.py")
print("\n" + "="*70 + "\n")
