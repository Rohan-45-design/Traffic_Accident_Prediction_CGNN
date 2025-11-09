import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_processing.feature_engineer import FeatureEngineer

def test_hour_fix():
    print("🔧 Testing hour_of_day fix...")
    
    # Load the existing cleaned data (skip API calls)
    cleaned_df = pd.read_csv('data/processed/cleaned_accidents.csv')
    
    # Load existing weather data (skip API calls)
    weather_df = pd.read_csv('data/raw/weather_data.csv')
    
    print(f"📊 Loaded {len(cleaned_df)} accident records")
    print(f"🌤️ Loaded {len(weather_df)} weather records")
    
    # Apply ONLY the feature engineering (with your fix)
    engineer = FeatureEngineer()
    enhanced_df = engineer.engineer_all_features(cleaned_df, weather_df)
    
    # Save the corrected files
    enhanced_df.to_csv('data/processed/feature_engineered_FIXED.csv', index=False)
    
    # Create corrected causal variables
    causal_columns = [
        'hour_of_day', 'day_of_week', 'season', 'is_weekend', 'rush_hour',
        'temp_avg', 'precipitation', 'wind_speed', 'visibility_score',
        'weather_severity_index', 'Accident Severity'
    ]
    
    available_columns = [col for col in causal_columns if col in enhanced_df.columns]
    causal_df = enhanced_df[available_columns]
    causal_df.to_csv('data/processed/causal_variables_FIXED.csv', index=False)
    
    print("✅ Fixed files created:")
    print("  - data/processed/feature_engineered_FIXED.csv")
    print("  - data/processed/causal_variables_FIXED.csv")
    
    # Check hour_of_day values
    print(f"\n🎯 Hour of day distribution:")
    print(causal_df['hour_of_day'].value_counts().sort_index().head(10))
    
    return causal_df

if __name__ == "__main__":
    result = test_hour_fix()
