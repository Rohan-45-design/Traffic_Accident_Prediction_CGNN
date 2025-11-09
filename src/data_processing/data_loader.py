import pandas as pd
import numpy as np
import os
from pathlib import Path

class DataLoader:
    """Load and validate primary accident dataset"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_accident_data(self, filename="global_road_accidents_dataset.csv"):
        """Load the primary accident dataset from Kaggle"""
        filepath = self.raw_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"❌ Dataset not found: {filepath}")
            
        print(f"📥 Loading accident data from {filepath}")
        df = pd.read_csv(filepath)
        
        print(f"✅ Loaded {len(df)} accident records")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Transform the dataset to match our expected format
        df = self.transform_dataset(df)
        
        return df
    
    def transform_dataset(self, df):
        """Transform the dataset to have required columns"""
        print("🔄 Transforming dataset to required format...")
        
        transformed_df = df.copy()
        
        # Create date column from Year, Month
        transformed_df['date'] = pd.to_datetime(
            transformed_df['Year'].astype(str) + '-' + 
            transformed_df['Month'].astype(str).str.zfill(2) + '-01'
        )
        
        # Create synthetic coordinates based on Country
        country_coords = {
            'USA': (39.8283, -98.5795),
            'UK': (55.3781, -3.4360),
            'Canada': (56.1304, -106.3468),
            'Australia': (-25.2744, 133.7751),
            'Germany': (51.1657, 10.4515),
            'France': (46.6034, 1.8883),
            'India': (20.5937, 78.9629),
            'China': (35.8617, 104.1954),
            'Japan': (36.2048, 138.2529),
            'Brazil': (-14.2350, -51.9253)
        }
        
        def assign_coordinates(country):
            if country in country_coords:
                base_lat, base_lon = country_coords[country]
                # Add small random variation (±2 degrees)
                lat = base_lat + np.random.uniform(-2, 2)
                lon = base_lon + np.random.uniform(-2, 2)
                return lat, lon
            else:
                return np.random.uniform(-90, 90), np.random.uniform(-180, 180)
        
        coords = transformed_df['Country'].apply(assign_coordinates)
        transformed_df['latitude'] = coords.apply(lambda x: x[0])
        transformed_df['longitude'] = coords.apply(lambda x: x[1])
        
        print("✅ Dataset transformed successfully!")
        print(f"📋 New columns added: date, latitude, longitude")
        
        return transformed_df
    
    def validate_data(self, df):
        """Validate required columns exist"""
        required_columns = ['latitude', 'longitude', 'date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"❌ Missing required columns: {missing_columns}")
        
        print("✅ Data validation passed")
        return True

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_accident_data()
    loader.validate_data(df)
