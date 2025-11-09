import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from data_processing.data_loader import DataLoader
from data_processing.weather_fetcher import WeatherFetcher
from data_processing.data_cleaner import DataCleaner
from data_processing.feature_engineer import FeatureEngineer

def main_phase2_pipeline():
    """
    Complete Phase 2 execution pipeline for CGNN project
    """
    print("🚀 CGNN PROJECT - PHASE 2: DATA COLLECTION & PREPROCESSING")
    print("=" * 70)
    
    try:
        # Step 1: Load primary accident dataset
        print("\n📊 STEP 1: LOADING PRIMARY DATASET")
        print("-" * 40)
        loader = DataLoader()
        
        # Check if raw dataset exists
        raw_file = "data/raw/global_road_accidents_dataset.csv"
        if not os.path.exists(raw_file):
            print("❌ Primary dataset not found!")
            print("📥 Please download from: https://www.kaggle.com/datasets/ankushpanday1/global-road-accidents-dataset")
            print(f"💾 Save as: {raw_file}")
            return False
        
        accident_df = loader.load_accident_data()
        loader.validate_data(accident_df)
        
        # Step 2: Fetch weather data
        print("\n🌤️ STEP 2: FETCHING WEATHER DATA")
        print("-" * 40)
        weather_fetcher = WeatherFetcher()
        
        # Use smaller sample for testing (remove sample_size for full dataset)
        weather_df = weather_fetcher.fetch_bulk_weather(accident_df, sample_size=500)
        
        # Step 3: Clean accident data
        print("\n🧹 STEP 3: CLEANING ACCIDENT DATA")
        print("-" * 40)
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_accident_data(accident_df)
        cleaned_file = cleaner.save_cleaned_data(cleaned_df)
        
        # Step 4: Engineer features
        print("\n🔧 STEP 4: FEATURE ENGINEERING")
        print("-" * 40)
        engineer = FeatureEngineer()
        enhanced_df = engineer.engineer_all_features(cleaned_df, weather_df)
        engineered_file = engineer.save_engineered_features(enhanced_df)
        
        # Step 5: Prepare causal variables
        print("\n🔗 STEP 5: PREPARING CAUSAL VARIABLES")
        print("-" * 40)
        
        # Select key variables for causal discovery (Phase 3)
        causal_columns = [
            'hour_of_day', 'day_of_week', 'season', 'is_weekend', 'rush_hour',
            'temp_avg', 'precipitation', 'wind_speed', 'visibility_score',
            'weather_severity_index', 'distance_from_center'
        ]
        
        # Filter columns that actually exist in the data
        available_columns = [col for col in causal_columns if col in enhanced_df.columns]
        
        # Add target variable if it exists
        target_columns = ['severity', 'accident_severity', 'casualty_count']
        for target_col in target_columns:
            if target_col in enhanced_df.columns:
                available_columns.append(target_col)
                break
        
        causal_df = enhanced_df[available_columns].copy()
        
        # Save causal variables
        causal_file = "data/processed/causal_variables.csv"
        causal_df.to_csv(causal_file, index=False)
        print(f"✅ Causal variables prepared: {len(available_columns)} variables")
        print(f"💾 Saved to: {causal_file}")
        
        # Step 6: Generate summary report
        print("\n📋 PHASE 2 COMPLETION SUMMARY")
        print("=" * 70)
        
        summary = {
            "Original accident records": len(accident_df),
            "Cleaned accident records": len(cleaned_df),
            "Weather records fetched": len(weather_df) if not weather_df.empty else 0,
            "Final engineered features": len(enhanced_df.columns),
            "Causal variables prepared": len(available_columns),
            "Data retention rate": f"{len(cleaned_df)/len(accident_df)*100:.1f}%"
        }
        
        for key, value in summary.items():
            print(f"📊 {key}: {value}")
        
        print("\n📁 Generated Files:")
        files_generated = [
            "data/raw/weather_data.csv",
            "data/processed/cleaned_accidents.csv", 
            "data/processed/feature_engineered.csv",
            "data/processed/causal_variables.csv"
        ]
        
        for file_path in files_generated:
            if os.path.exists(file_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} (not created)")
        
        print("\n🎉 PHASE 2 COMPLETED SUCCESSFULLY!")
        print("🚀 Ready to proceed to Phase 3: Causal Discovery Implementation")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main_phase2_pipeline()
    if success:
        print("\n✅ Run this script after downloading the Kaggle dataset!")
    else:
        print("\n❌ Please fix the issues above and try again.")
