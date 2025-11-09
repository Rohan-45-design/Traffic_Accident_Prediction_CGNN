import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np

class WeatherFetcher:
    """Fetch weather data from Open-Meteo API"""
    
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/era5"
        self.weather_codes = {
            0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
            45: 'Fog', 48: 'Depositing rime fog',
            51: 'Light drizzle', 53: 'Moderate drizzle', 55: 'Dense drizzle',
            61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
            71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow',
            95: 'Thunderstorm', 96: 'Thunderstorm with hail'
        }
    
    def fetch_weather_for_location(self, lat, lon, date):
        """Fetch weather data for a specific location and date"""
        
        # Convert pandas timestamp to string date (YYYY-MM-DD format)
        if isinstance(date, pd.Timestamp):
            date_str = date.strftime('%Y-%m-%d')
        elif isinstance(date, str):
            try:
                parsed_date = pd.to_datetime(date)
                date_str = parsed_date.strftime('%Y-%m-%d')
            except:
                date_str = date
        else:
            date_str = str(date)
        
        # Ensure date is within reasonable range (ERA5 available from 1979)
        try:
            date_obj = pd.to_datetime(date_str)
            if date_obj.year < 1979:
                date_str = "2010-01-01"
            elif date_obj.year > 2023:
                date_str = "2023-01-01"
        except:
            date_str = "2010-01-01"
        
        params = {
            'latitude': float(lat),
            'longitude': float(lon),
            'start_date': date_str,
            'end_date': date_str,
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode',
            'timezone': 'UTC'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'daily' in data and data['daily']:
                daily = data['daily']
                return {
                    'latitude': lat,
                    'longitude': lon,
                    'date': date_str,
                    'temp_max': daily['temperature_2m_max'][0] if daily['temperature_2m_max'] else None,
                    'temp_min': daily['temperature_2m_min'][0] if daily['temperature_2m_min'] else None,
                    'precipitation': daily['precipitation_sum'][0] if daily['precipitation_sum'] else None,
                    'wind_speed': daily['windspeed_10m_max'][0] if daily['windspeed_10m_max'] else None,
                    'weather_code': daily['weathercode'][0] if daily['weathercode'] else None,
                    'weather_description': self.weather_codes.get(daily['weathercode'][0], 'Unknown') if daily['weathercode'] else 'Unknown'
                }
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                print(f"⚠️ Bad request for {lat:.2f}, {lon:.2f}, {date_str} - using default weather")
                return {
                    'latitude': lat,
                    'longitude': lon,
                    'date': date_str,
                    'temp_max': 20.0,
                    'temp_min': 10.0,
                    'precipitation': 0.0,
                    'wind_speed': 5.0,
                    'weather_code': 1,
                    'weather_description': 'Mainly clear'
                }
            else:
                print(f"❌ HTTP Error {e.response.status_code} for {lat}, {lon}: {e}")
                return None
        except Exception as e:
            print(f"❌ Weather fetch error for {lat}, {lon}: {e}")
            return None
    
    def fetch_bulk_weather(self, accident_df, sample_size=500):
        """Fetch weather data for accident locations"""
        print(f"🌤️ Fetching weather data for {min(sample_size, len(accident_df))} locations...")
        
        sample_df = accident_df.head(sample_size)
        unique_locations = sample_df[['latitude', 'longitude', 'date']].drop_duplicates()
        
        print(f"📍 Processing {len(unique_locations)} unique locations...")
        
        weather_data = []
        successful_requests = 0
        
        for idx, row in unique_locations.iterrows():
            weather_record = self.fetch_weather_for_location(
                row['latitude'], row['longitude'], row['date']
            )
            
            if weather_record:
                weather_data.append(weather_record)
                successful_requests += 1
            
            if idx % 50 == 0 and idx > 0:
                print(f"✅ Processed {idx} locations ({successful_requests} successful)...")
            
            time.sleep(0.2)
        
        weather_df = pd.DataFrame(weather_data)
        weather_df.to_csv('data/raw/weather_data.csv', index=False)
        print(f"✅ Weather data saved: {len(weather_df)} records ({successful_requests}/{len(unique_locations)} successful)")
        
        return weather_df

if __name__ == "__main__":
    fetcher = WeatherFetcher()
    print("🧪 Testing weather fetch...")
