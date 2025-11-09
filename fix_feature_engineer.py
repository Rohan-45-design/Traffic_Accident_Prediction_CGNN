"""
CORRECTED fix_feature_engineered.py
====================================
Fixes ONLY what needs fixing:
1. Converts age RANGES to averages (18-36 → 27) - ONLY if they're strings
2. Removes ID columns
3. Encodes categorical strings ONLY
4. PRESERVES all numerical columns (hour_of_day, etc.)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

def convert_age_range_to_average(age_value):
    """Convert age range to average: '18-36' → 27"""
    if pd.isna(age_value):
        return np.nan
    
    age_str = str(age_value).strip()
    
    # Handle range format "18-36" or "18 - 36"
    match = re.match(r'^(\d+)\s*-\s*(\d+)$', age_str)
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        return (low + high) / 2
    
    # Handle "Over 75" or "75+"
    if 'over' in age_str.lower() or '+' in age_str:
        numbers = re.findall(r'\d+', age_str)
        if numbers:
            return int(numbers[0]) + 5
    
    # Handle "Under 16"
    if 'under' in age_str.lower():
        numbers = re.findall(r'\d+', age_str)
        if numbers:
            return int(numbers[0]) / 2
    
    # Try direct number conversion
    try:
        return float(age_str)
    except:
        return np.nan

def fix_feature_engineered():
    """Complete data cleanup - PRESERVES numerical columns"""
    
    print("🚀 CORRECTED FEATURE_ENGINEERED.CSV CLEANUP")
    print("="*70)
    
    # Load data
    df = pd.read_csv("data/processed/feature_engineered.csv")
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # ============================================
    # STEP 1: REMOVE ID COLUMNS
    # ============================================
    print("\n📋 STEP 1: Removing ID columns")
    id_keywords = ['index', 'reference', 'easting', 'northing', 'lsoa']
    cols_to_remove = [col for col in df.columns if any(keyword in col.lower() for keyword in id_keywords)]
    
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)
        print(f"   Removed {len(cols_to_remove)} columns")
    
    # ============================================
    # STEP 2: CONVERT AGE RANGES (ONLY IF STRINGS)
    # ============================================
    print("\n🔢 STEP 2: Converting age ranges to averages (ONLY if string ranges)")
    
    # Find age columns
    age_columns = [col for col in df.columns if 'age' in col.lower()]
    
    for col in age_columns:
        # Check if column has string ranges (not already numeric)
        if df[col].dtype == 'object':  # Only process if it's text
            print(f"\n   Processing: {col}")
            sample_values = df[col].dropna().unique()[:5]
            print(f"   Sample before: {sample_values}")
            
            # Check if it actually has ranges
            has_ranges = any('-' in str(val) for val in df[col].dropna().head(100))
            
            if has_ranges:
                df[col] = df[col].apply(convert_age_range_to_average)
                
                # Fill NaN with median
                median_age = df[col].median()
                if pd.isna(median_age) or median_age == 0:
                    median_age = 40
                
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    df[col].fillna(median_age, inplace=True)
                    print(f"   Filled {missing_count} NaN with median: {median_age:.1f}")
                
                print(f"   Range after: {df[col].min():.1f} - {df[col].max():.1f}")
            else:
                print(f"   ⚠️ No ranges found, trying numeric conversion")
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median_age = df[col].median()
                df[col].fillna(median_age if not pd.isna(median_age) else 40, inplace=True)
        else:
            print(f"   ✅ {col} already numeric - PRESERVING AS-IS")
    
    # ============================================
    # STEP 3: ENCODE CATEGORICAL COLUMNS (NOT NUMERIC ONES!)
    # ============================================
    print("\n🔄 STEP 3: Encoding categorical columns (ONLY text columns)")
    
    encodings = {
        'Accident_Severity': {'Slight': 0, 'Serious': 1, 'Fatal': 2},
        'Casualty_Severity': {'Slight': 0, 'Serious': 1, 'Fatal': 2},
        'Light_Conditions': {
            'Daylight': 0,
            'Darkness - lights lit': 1,
            'Darkness - lights unlit': 2,
            'Darkness - no lighting': 2,
            'Darkness - lighting unknown': 2
        },
        'Weather_Conditions': {
            'Fine no high winds': 0, 'Fine + high winds': 0,
            'Raining no high winds': 1, 'Raining + high winds': 1,
            'Snowing no high winds': 2, 'Snowing + high winds': 2,
            'Fog or mist': 3, 'Other': 4, 'Unknown': 4
        },
        'Road_Surface_Conditions': {
            'Dry': 0, 'Wet or damp': 1, 'Frost or ice': 2,
            'Snow': 3, 'Flood over 3cm. deep': 4
        },
        'Urban_or_Rural_Area': {'Urban': 0, 'Rural': 1},
        'Sex_of_Driver': {'Male': 0, 'M': 0, 'Female': 1, 'F': 1},
        'Sex_of_Casualty': {'Male': 0, 'M': 0, 'Female': 1, 'F': 1}
    }
    
    encoded_count = 0
    for col, mapping in encodings.items():
        if col in df.columns and df[col].dtype == 'object':  # Only if text
            # Make case-insensitive
            case_map = {}
            for key, val in mapping.items():
                case_map[key] = val
                case_map[key.lower()] = val
            
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].map(lambda x: case_map.get(x, case_map.get(x.lower(), -1)))
            df[col] = df[col].astype(int)
            print(f"   ✅ {col}")
            encoded_count += 1
    
    print(f"   Total encoded: {encoded_count} columns")
    
    # ============================================
    # STEP 4: AUTO-ENCODE REMAINING TEXT COLUMNS
    # ============================================
    print("\n🔄 STEP 4: Auto-encoding remaining text columns")
    
    object_cols = df.select_dtypes(include=['object']).columns
    
    for col in object_cols:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 100:
            print(f"   Auto-encoding {col}: {len(unique_vals)} categories")
            mapping = {str(val): idx for idx, val in enumerate(sorted(unique_vals))}
            df[col] = df[col].astype(str).map(mapping)
            df[col].fillna(-1, inplace=True)
            df[col] = df[col].astype(int)
    
    # ============================================
    # STEP 5: HANDLE MISSING VALUES
    # ============================================
    print("\n💧 STEP 5: Handling remaining missing values")
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
    
    # ============================================
    # STEP 6: FINAL VALIDATION
    # ============================================
    print("\n✅ FINAL VALIDATION")
    print(f"   Shape: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    object_cols_final = df.select_dtypes(include=['object']).columns
    print(f"   Non-numerical columns: {len(object_cols_final)}")
    
    if len(object_cols_final) == 0:
        print("   🎉 ALL COLUMNS ARE NUMERICAL!")
    
    # Validate key columns NOT destroyed
    check_cols = ['hour_of_day', 'day_of_week', 'month', 'year']
    for col in check_cols:
        if col in df.columns:
            print(f"   {col}: min={df[col].min()}, max={df[col].max()}, unique={df[col].nunique()}")
    
    # ============================================
    # STEP 7: SAVE FILES
    # ============================================
    print("\n💾 STEP 7: Saving cleaned files")
    
    output_path = "data/processed/feature_engineered_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"   ✅ {output_path}")
    
    phase4_path = "data/processed/causal_variables_PHASE4_READY.csv"
    df.to_csv(phase4_path, index=False)
    print(f"   ✅ {phase4_path}")
    
    print("\n🎉 CLEANUP COMPLETE!")
    print("📄 Use 'causal_variables_PHASE4_READY.csv' for Phase 4")
    
    return df

if __name__ == "__main__":
    cleaned_df = fix_feature_engineered()
