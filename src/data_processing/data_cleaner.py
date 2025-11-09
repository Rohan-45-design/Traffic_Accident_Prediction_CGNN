import pandas as pd
import numpy as np
from datetime import datetime

class DataCleaner:
    """Clean and preprocess accident data"""
    
    def __init__(self):
        self.cleaning_stats = {}
    
    def clean_accident_data(self, df):
        """Clean the accident dataset"""
        print("🧹 Starting data cleaning...")
        original_count = len(df)
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # 1. Remove exact duplicates
        before_dedup = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        dedup_removed = before_dedup - len(cleaned_df)
        print(f"✅ Removed {dedup_removed} duplicate records")
        
        # 2. Handle missing values in critical columns
        critical_columns = ['latitude', 'longitude', 'date']
        before_missing = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=critical_columns)
        missing_removed = before_missing - len(cleaned_df)
        print(f"✅ Removed {missing_removed} records with missing critical data")
        
        # 3. Fix data types
        print("🔧 Fixing data types...")
        
        # Convert date column
        try:
            cleaned_df['date'] = pd.to_datetime(cleaned_df['date'], errors='coerce')
            invalid_dates = cleaned_df['date'].isna().sum()
            cleaned_df = cleaned_df.dropna(subset=['date'])
            print(f"✅ Converted dates, removed {invalid_dates} invalid dates")
        except Exception as e:
            print(f"⚠️ Date conversion issue: {e}")
        
        # Convert coordinates to numeric
        try:
            cleaned_df['latitude'] = pd.to_numeric(cleaned_df['latitude'], errors='coerce')
            cleaned_df['longitude'] = pd.to_numeric(cleaned_df['longitude'], errors='coerce')
            
            # Remove invalid coordinates
            before_coords = len(cleaned_df)
            cleaned_df = cleaned_df.dropna(subset=['latitude', 'longitude'])
            
            # Validate coordinate ranges
            valid_lat = (cleaned_df['latitude'] >= -90) & (cleaned_df['latitude'] <= 90)
            valid_lon = (cleaned_df['longitude'] >= -180) & (cleaned_df['longitude'] <= 180)
            cleaned_df = cleaned_df[valid_lat & valid_lon]
            
            coord_removed = before_coords - len(cleaned_df)
            print(f"✅ Fixed coordinates, removed {coord_removed} invalid coordinates")
            
        except Exception as e:
            print(f"⚠️ Coordinate conversion issue: {e}")
        
        # 4. Handle other missing values with appropriate strategies
        print("🔄 Handling remaining missing values...")
        
        # Fill categorical missing values with 'Unknown'
        categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in critical_columns:
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')
        
        # Fill numerical missing values with median
        numerical_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if col not in ['latitude', 'longitude']:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # Store cleaning statistics
        self.cleaning_stats = {
            'original_records': original_count,
            'final_records': len(cleaned_df),
            'duplicates_removed': dedup_removed,
            'missing_data_removed': missing_removed,
            'total_removed': original_count - len(cleaned_df),
            'retention_rate': len(cleaned_df) / original_count * 100
        }
        
        print(f"✅ Cleaning complete!")
        print(f"📊 Original: {original_count} → Final: {len(cleaned_df)} ({self.cleaning_stats['retention_rate']:.1f}% retained)")
        
        return cleaned_df
    
    def save_cleaned_data(self, cleaned_df, filename="cleaned_accidents.csv"):
        """Save cleaned data to processed directory"""
        filepath = f"data/processed/{filename}"
        cleaned_df.to_csv(filepath, index=False)
        print(f"💾 Cleaned data saved to {filepath}")
        return filepath

if __name__ == "__main__":
    cleaner = DataCleaner()
    print("🧪 Testing data cleaner...")
