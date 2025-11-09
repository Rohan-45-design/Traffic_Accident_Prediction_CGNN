import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    """Create temporal, spatial, and environmental features"""
    
    def __init__(self):
        self.feature_stats = {}
    
    def create_temporal_features(self, df):
        """Create time-based features"""
        print("⏰ Creating temporal features...")
        
        feature_df = df.copy()
        
        # Ensure date is datetime
        feature_df['date'] = pd.to_datetime(feature_df['date'])
        
        # Basic temporal features
        # Map 'Time of Day' text to actual hours
        if 'Time of Day' in feature_df.columns:
            time_to_hour = {
                'Morning': 8,
                'Afternoon': 14, 
                'Evening': 18,
                'Night': 22,
                'Early Morning': 6,
                'Late Morning': 10,
                'Early Afternoon': 13,
                'Late Afternoon': 16,
                'Early Evening': 17,
                'Late Evening': 20,
                'Early Night': 21,
                'Late Night': 23,
                'Dawn': 5,
                'Dusk': 19,
                'Noon': 12,
                'Midnight': 0
            }
            
            # Map and fill missing values with default (noon)
            feature_df['hour_of_day'] = feature_df['Time of Day'].map(time_to_hour).fillna(12)
            feature_df['hour_of_day'] = feature_df['hour_of_day'].astype(int)
        else:
            print("⚠️ 'Time of Day' column not found, using default hours")
            feature_df['hour_of_day'] = np.random.randint(0, 24, size=len(feature_df))


            feature_df['day_of_week'] = feature_df['date'].dt.dayofweek  # Monday=0, Sunday=6
            feature_df['day_of_month'] = feature_df['date'].dt.day
            feature_df['month'] = feature_df['date'].dt.month
            feature_df['year'] = feature_df['date'].dt.year
            feature_df['quarter'] = feature_df['date'].dt.quarter
            
            # Season mapping (0=Winter, 1=Spring, 2=Summer, 3=Fall)
            season_map = {12: 0, 1: 0, 2: 0,  # Winter
                        3: 1, 4: 1, 5: 1,   # Spring
                        6: 2, 7: 2, 8: 2,   # Summer
                        9: 3, 10: 3, 11: 3} # Fall
            feature_df['season'] = feature_df['month'].map(season_map)
            
            # Derived temporal features
            feature_df['is_weekend'] = (feature_df['day_of_week'] >= 5).astype(int)
            feature_df['is_weekday'] = (feature_df['day_of_week'] < 5).astype(int)
            
            # Rush hour periods (7-9 AM, 5-7 PM)
            morning_rush = (feature_df['hour_of_day'] >= 7) & (feature_df['hour_of_day'] <= 9)
            evening_rush = (feature_df['hour_of_day'] >= 17) & (feature_df['hour_of_day'] <= 19)
            feature_df['rush_hour'] = (morning_rush | evening_rush).astype(int)
            
            # Time periods
            feature_df['time_period'] = pd.cut(feature_df['hour_of_day'], 
                                            bins=[0, 6, 12, 18, 24], 
                                            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                            include_lowest=True)
            
            print("✅ Temporal features created")
        return feature_df
    
    def create_weather_features(self, df, weather_df):
        """Merge and enhance weather features"""
        print("🌤️ Creating weather features...")
        
        # FIX: Convert weather_df date to datetime before merging
        weather_df = weather_df.copy()
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Also ensure accident data date is datetime
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Now merge with matching data types
        enhanced_df = pd.merge(df, weather_df, 
                            on=['latitude', 'longitude', 'date'], 
                            how='left')
        
        # Rest of the method stays exactly the same...
        # Weather-derived features
        if 'temp_max' in enhanced_df.columns and 'temp_min' in enhanced_df.columns:
            enhanced_df['temp_range'] = enhanced_df['temp_max'] - enhanced_df['temp_min']
            enhanced_df['temp_avg'] = (enhanced_df['temp_max'] + enhanced_df['temp_min']) / 2
        
        # Visibility score based on weather conditions
        if 'precipitation' in enhanced_df.columns:
            enhanced_df['visibility_score'] = np.where(
                enhanced_df['precipitation'] > 10, 0.2,
                np.where(enhanced_df['precipitation'] > 2, 0.5,
                np.where(enhanced_df['precipitation'] > 0, 0.8,
                1.0)))
        
        # Weather severity index
        weather_severity = 0
        if 'wind_speed' in enhanced_df.columns:
            weather_severity += np.where(enhanced_df['wind_speed'] > 15, 1, 0)
        if 'precipitation' in enhanced_df.columns:
            weather_severity += np.where(enhanced_df['precipitation'] > 5, 1, 0)
        enhanced_df['weather_severity_index'] = weather_severity
        
        print("✅ Weather features created")
        return enhanced_df

    
    def create_spatial_features(self, df):
        """Create location-based features"""
        print("🗺️ Creating spatial features...")
        
        spatial_df = df.copy()
        
        # Coordinate-based features
        spatial_df['lat_rounded'] = spatial_df['latitude'].round(2)
        spatial_df['lon_rounded'] = spatial_df['longitude'].round(2)
        
        # Simple clustering based on coordinates (for accident hotspots)
        spatial_df['location_cluster'] = (
            spatial_df['lat_rounded'].astype(str) + "_" + 
            spatial_df['lon_rounded'].astype(str)
        )
        
        # Distance from major cities (simplified - you can enhance this)
        # Example: Distance from origin (0,0) - replace with actual city coordinates
        spatial_df['distance_from_center'] = np.sqrt(
            spatial_df['latitude']**2 + spatial_df['longitude']**2
        )
        
        print("✅ Spatial features created")
        return spatial_df
    
    def engineer_all_features(self, accident_df, weather_df=None):
        """Complete feature engineering pipeline"""
        print("🔧 Starting complete feature engineering...")
        
        # Start with temporal features
        enhanced_df = self.create_temporal_features(accident_df)
        
        # Add weather features if weather data is available
        if weather_df is not None and not weather_df.empty:
            enhanced_df = self.create_weather_features(enhanced_df, weather_df)
        else:
            print("⚠️ No weather data provided, skipping weather features")
        
        # Add spatial features
        enhanced_df = self.create_spatial_features(enhanced_df)
        
        # Store feature statistics
        self.feature_stats = {
            'total_features': len(enhanced_df.columns),
            'original_features': len(accident_df.columns),
            'new_features': len(enhanced_df.columns) - len(accident_df.columns),
            'records': len(enhanced_df)
        }
        
        print(f"✅ Feature engineering complete!")
        print(f"📊 Added {self.feature_stats['new_features']} new features")
        print(f"📋 Total features: {self.feature_stats['total_features']}")
        
        return enhanced_df
    
    def save_engineered_features(self, df, filename="feature_engineered.csv"):
        """Save feature engineered data"""
        filepath = f"data/processed/{filename}"
        df.to_csv(filepath, index=False)
        print(f"💾 Feature engineered data saved to {filepath}")
        return filepath

if __name__ == "__main__":
    engineer = FeatureEngineer()
    print("🧪 Testing feature engineering...")
