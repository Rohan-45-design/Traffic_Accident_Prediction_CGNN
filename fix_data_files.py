"""
FIXED Data Cleaning Script
===========================
Properly fixes CSV files:
1. Removes ONLY unnecessary ID/text columns
2. Converts ONLY categorical strings to numbers
3. PRESERVES numerical columns (age, hour_of_day, temperature, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def get_columns_to_remove():
    """Get exact list of columns to remove"""
    return [
        # ID columns - ONLY these
        'accident_id', 'incident_id', 'report_id', 'case_number',
        'police_report_number', 'license_plate', 'vehicle_id',
        
        # Personal identifiable information
        'driver_name', 'passenger_names', 'officer_name',
        'witness_name', 'owner_name',
        
        # Location text (but keep lat/lon if they exist)
        'location_name', 'address', 'full_address', 'street_address',
        
        # Exact timestamps (but keep hour, day, month, year features)
        'timestamp', 'exact_time', 'datetime',
        
        # Text descriptions
        'description', 'detailed_description', 'notes',
        'comments', 'remarks', 'narrative'
    ]

def remove_unnecessary_columns(df):
    """Remove ONLY ID and text columns - keep all numerical data"""
    
    columns_to_remove = get_columns_to_remove()
    
    # Find columns that actually exist
    cols_to_drop = [col for col in columns_to_remove if col in df.columns]
    
    if cols_to_drop:
        print(f"   Removing {len(cols_to_drop)} unnecessary columns:")
        for col in cols_to_drop:
            print(f"      - {col}")
        df = df.drop(columns=cols_to_drop)
    else:
        print("   ℹ️  No unnecessary columns found (this is normal if already cleaned)")
    
    return df

def is_categorical_column(series, col_name):
    """
    Determine if a column is categorical (needs encoding)
    Returns True ONLY if it's an object/string type with categorical values
    """
    # Skip if already numeric
    if series.dtype in ['int64', 'float64', 'int32', 'float32']:
        return False
    
    # Skip if column name suggests it's already a number
    numeric_keywords = ['hour', 'day', 'month', 'year', 'age', 'speed', 
                       'temperature', 'humidity', 'count', 'number', 
                       'distance', 'volume', 'score', 'index', 'km', 'mph']
    
    col_lower = col_name.lower()
    if any(keyword in col_lower for keyword in numeric_keywords):
        return False
    
    # Check if it's object type with non-numeric values
    if series.dtype == 'object':
        # Try to see if values are actually numbers stored as strings
        try:
            pd.to_numeric(series.dropna().head(10))
            # If conversion works, it's numeric data stored as string
            return False
        except:
            # If conversion fails, it's truly categorical
            return True
    
    return False

def encode_categorical_columns(df):
    """
    Encode ONLY categorical string columns
    DO NOT touch numerical columns
    """
    
    print("\n   Encoding categorical columns...")
    
    # Predefined mappings for known categorical columns
    encoding_mappings = {
        'accident_severity': {
            'minor': 0, 'low': 0, 'slight': 0,
            'moderate': 1, 'medium': 1,
            'severe': 2, 'serious': 2, 'high': 2,
            'fatal': 3, 'critical': 3
        },
        'weather_conditions': {
            'clear': 0, 'sunny': 0,
            'cloudy': 1,
            'rain': 2, 'rainy': 2,
            'fog': 3, 'foggy': 3,
            'snow': 4, 'snowy': 4
        },
        'weather': {
            'clear': 0, 'rain': 1, 'fog': 2, 'snow': 3
        },
        'road_surface': {
            'dry': 0,
            'wet': 1,
            'icy': 2,
            'snow': 3
        },
        'light_conditions': {
            'daylight': 0, 'day': 0,
            'dawn': 1, 'dusk': 1,
            'darkness': 2, 'night': 2
        },
        'visibility': {
            'poor': 0, 'low': 0,
            'moderate': 1,
            'good': 2, 'high': 2
        },
        'urban_rural': {
            'urban': 0, 'city': 0,
            'rural': 1
        },
        'vehicle_type': {
            'car': 0, 'sedan': 0,
            'truck': 1,
            'motorcycle': 2,
            'bus': 3,
            'van': 4
        },
        'driver_gender': {
            'male': 0, 'm': 0,
            'female': 1, 'f': 1
        },
        'is_weekend': {
            'no': 0, 'false': 0,
            'yes': 1, 'true': 1
        }
    }
    
    encoded_count = 0
    skipped_numeric = []
    
    for col in df.columns:
        # Check if this column needs encoding
        if not is_categorical_column(df[col], col):
            if df[col].dtype == 'object':
                # It's object type but should be numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    skipped_numeric.append(col)
                except:
                    pass
            continue
        
        # Check if we have a predefined mapping
        col_lower = col.lower().replace('_', '')
        mapped = False
        
        for key, mapping in encoding_mappings.items():
            key_clean = key.replace('_', '')
            if col_lower == key_clean or col_lower.startswith(key_clean):
                # Apply the mapping
                original_unique = df[col].unique()
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].map({k.lower(): v for k, v in mapping.items()})
                
                unmapped = df[col].isna().sum()
                if unmapped > 0:
                    print(f"      ⚠️  {col}: {unmapped} unmapped values (setting to -1)")
                    df[col] = df[col].fillna(-1)
                
                df[col] = df[col].astype(int)
                print(f"      ✅ {col}: {original_unique} → {sorted(df[col].unique())}")
                encoded_count += 1
                mapped = True
                break
        
        # If no predefined mapping, create one automatically
        if not mapped:
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 20:  # Only if reasonable number of categories
                print(f"      🔄 Auto-encoding {col} ({len(unique_values)} categories)")
                value_map = {str(val): idx for idx, val in enumerate(sorted(unique_values))}
                df[col] = df[col].astype(str).map(value_map)
                df[col] = df[col].fillna(-1).astype(int)
                encoded_count += 1
    
    if skipped_numeric:
        print(f"\n   ℹ️  Preserved {len(skipped_numeric)} numerical columns:")
        for col in skipped_numeric[:5]:  # Show first 5
            print(f"      - {col}")
        if len(skipped_numeric) > 5:
            print(f"      ... and {len(skipped_numeric)-5} more")
    
    print(f"\n   ✅ Encoded {encoded_count} categorical columns")
    return df

def validate_numerical_columns(df):
    """Check which columns are still non-numeric"""
    
    print("\n   Validating data types...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    object_cols = df.select_dtypes(include=['object']).columns
    
    print(f"      ✅ Numerical columns: {len(numeric_cols)}")
    print(f"      ⚠️  Non-numerical columns: {len(object_cols)}")
    
    if len(object_cols) > 0:
        print(f"\n   Still non-numeric:")
        for col in object_cols:
            sample_vals = df[col].dropna().unique()[:3]
            print(f"      - {col}: {sample_vals}")
            
            # Try one more time to convert
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"        → Converted to numeric")
            except:
                print(f"        → Dropping this column")
                df = df.drop(columns=[col])
    
    return df

def handle_missing_values(df):
    """Handle missing values without destroying data"""
    
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        print(f"\n   Handling {missing_total} missing values...")
        
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Use median for numeric columns
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"      - {col}: Filled {missing} with median ({median_val})")
    
    return df

def show_column_summary(df, title):
    """Show summary of columns and their types"""
    
    print(f"\n   {title}:")
    print(f"      Total columns: {len(df.columns)}")
    print(f"      Numerical: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"      Non-numerical: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Show sample of preserved numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    important_numeric = [c for c in numeric_cols if any(k in c.lower() for k in 
                        ['hour', 'day', 'month', 'age', 'speed', 'temp'])]
    if important_numeric:
        print(f"\n      Important numerical columns preserved:")
        for col in important_numeric[:5]:
            sample_vals = df[col].dropna().unique()[:3]
            print(f"         ✅ {col}: {sample_vals}")

def fix_csv_file(input_path, output_path):
    """Fix a single CSV file properly"""
    
    print(f"\n{'='*70}")
    print(f"Processing: {input_path}")
    print(f"{'='*70}")
    
    try:
        # Load file
        df = pd.read_csv(input_path)
        print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        show_column_summary(df, "BEFORE cleaning")
        
        # Step 1: Remove unnecessary columns
        print(f"\n📋 Step 1: Removing unnecessary columns")
        df = remove_unnecessary_columns(df)
        print(f"   Result: {df.shape[1]} columns")
        
        # Step 2: Encode categorical columns ONLY
        print(f"\n🔄 Step 2: Encoding categorical columns")
        df = encode_categorical_columns(df)
        
        # Step 3: Validate and convert remaining objects
        print(f"\n🔍 Step 3: Validating data types")
        df = validate_numerical_columns(df)
        
        # Step 4: Handle missing values
        print(f"\n💧 Step 4: Handling missing values")
        df = handle_missing_values(df)
        
        show_column_summary(df, "AFTER cleaning")
        
        # Final check
        object_cols = df.select_dtypes(include=['object']).columns
        success = len(object_cols) == 0
        
        if success:
            print(f"\n✅ SUCCESS: All columns are numerical!")
        else:
            print(f"\n⚠️  WARNING: {len(object_cols)} columns still non-numeric")
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution"""
    
    print("🚀 FIXED DATA CLEANING SCRIPT")
    print("="*70)
    print("This script will:")
    print("  ✅ Remove ONLY ID/text columns (accident_id, driver_name, etc.)")
    print("  ✅ Convert ONLY categorical strings to numbers")
    print("  ✅ PRESERVE all numerical columns (hour, age, temperature, etc.)")
    print("="*70)
    
    files = [
        ('data/processed/causal_variables_FIXED.csv', 'data/processed/causal_variables_final.csv'),
        ('data/processed/causal_variables.csv', 'data/processed/causal_variables_cleaned.csv')
    ]
    
    success = 0
    for input_file, output_file in files:
        if Path(input_file).exists():
            if fix_csv_file(input_file, output_file):
                success += 1
        else:
            print(f"\n⚠️  Not found: {input_file}")
    
    print(f"\n{'='*70}")
    print(f"✅ Successfully processed {success} file(s)")
    print(f"{'='*70}")
    
    if success > 0:
        print("\n🎯 Use these cleaned files for Phase 4:")
        for _, output_file in files:
            if Path(output_file).exists():
                print(f"   ✅ {output_file}")

if __name__ == "__main__":
    main()
