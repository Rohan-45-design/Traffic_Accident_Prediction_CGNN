"""
Diagnostic Script
=================
Find why model only predicts Class 2
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*70)
print("DIAGNOSTIC ANALYSIS")
print("="*70 + "\n")

# ============================================================
# CHECK 1: Data Quality
# ============================================================
print("CHECK 1: DATA QUALITY")
print("-"*70)

df = pd.read_csv('data/processed/causal_variables_small.csv')
print(f"Dataset size: {len(df)}")

# Find target column
if 'Accident_Severity' in df.columns:
    target = 'Accident_Severity'
else:
    target = df.columns[-1]

print(f"Target column: {target}")
print(f"\nClass distribution:")
print(df[target].value_counts().sort_index())

# Check if features differ by class
features = ['hour_of_day', 'temp_avg', 'precipitation', 'wind_speed', 
            'visibility_score', 'weather_severity_index']

print(f"\n📊 Feature statistics by class:")
for feat in features:
    if feat in df.columns:
        print(f"\n{feat}:")
        stats = df.groupby(target)[feat].agg(['mean', 'std', 'min', 'max'])
        print(stats)
        
        # Check if feature varies between classes
        means = df.groupby(target)[feat].mean()
        if means.std() < 0.01:
            print(f"⚠️ WARNING: {feat} has almost IDENTICAL values across classes!")

# ============================================================
# CHECK 2: Feature Variance
# ============================================================
print("\n" + "="*70)
print("CHECK 2: FEATURE VARIANCE")
print("-"*70)

for feat in features:
    if feat in df.columns:
        variance = df[feat].var()
        unique_ratio = len(df[feat].unique()) / len(df)
        print(f"{feat:25s} | Variance: {variance:8.4f} | Unique: {unique_ratio:.2%}")
        
        if variance < 0.01:
            print(f"  ⚠️ WARNING: Very low variance - feature might not be useful!")

# ============================================================
# CHECK 3: Correlation with Target
# ============================================================
print("\n" + "="*70)
print("CHECK 3: CORRELATION WITH TARGET")
print("-"*70)

for feat in features:
    if feat in df.columns:
        corr = df[feat].corr(df[target])
        print(f"{feat:25s} | Correlation: {corr:7.4f}")
        
        if abs(corr) < 0.05:
            print(f"  ⚠️ WARNING: Very weak correlation with target!")

# ============================================================
# CHECK 4: Class Separability
# ============================================================
print("\n" + "="*70)
print("CHECK 4: CLASS SEPARABILITY")
print("-"*70)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X = df[features].values
y = df[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simple logistic regression baseline
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y)
baseline_acc = lr.score(X_scaled, y)

print(f"\n📊 Logistic Regression Baseline Accuracy: {baseline_acc:.2%}")

if baseline_acc < 0.40:
    print("⚠️ CRITICAL: Even simple linear model can't separate classes!")
    print("   This suggests features don't contain enough information.")
else:
    print("✅ Classes are separable - neural network should work better than this!")

# Predict distribution
lr_preds = lr.predict(X_scaled)
print(f"\nLogistic Regression predictions:")
unique, counts = np.unique(lr_preds, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} ({count/len(lr_preds)*100:.1f}%)")

# ============================================================
# CHECK 5: Random Forest Baseline
# ============================================================
print("\n" + "="*70)
print("CHECK 5: RANDOM FOREST BASELINE")
print("-"*70)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_scaled, y)
rf_acc = rf.score(X_scaled, y)

print(f"\n📊 Random Forest Accuracy: {rf_acc:.2%}")

# Feature importance
print(f"\n📊 Feature Importances:")
for feat, imp in zip(features, rf.feature_importances_):
    print(f"  {feat:25s}: {imp:.4f}")

rf_preds = rf.predict(X_scaled)
print(f"\nRandom Forest predictions:")
unique, counts = np.unique(rf_preds, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} ({count/len(rf_preds)*100:.1f}%)")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY & RECOMMENDATIONS")
print("="*70 + "\n")

if baseline_acc < 0.35:
    print("🔴 CRITICAL PROBLEM: Features cannot separate classes")
    print("\nRecommendations:")
    print("  1. Add more features (day_of_week, rush_hour, location)")
    print("  2. Engineer features (interactions, polynomial)")
    print("  3. Get better/different data")
    print("\n⚠️ No amount of hyperparameter tuning will fix this!")
    
elif baseline_acc < 0.45:
    print("⚠️ MODERATE PROBLEM: Weak class separability")
    print("\nRecommendations:")
    print("  1. Try feature engineering")
    print("  2. Use ensemble methods")
    print("  3. Consider if 3-class is too granular (try binary?)")
    
else:
    print("✅ DATA IS GOOD: Classes are separable")
    print("\nYour CGNN should achieve > 50% accuracy")
    print("\nIf CGNN still fails, check:")
    print("  1. Model initialization")
    print("  2. Graph structure")
    print("  3. Training implementation")

print("\n" + "="*70 + "\n")
