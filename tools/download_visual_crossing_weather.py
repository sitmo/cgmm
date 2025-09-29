#!/usr/bin/env python3
"""
Download hourly weather data for Amsterdam from Visual Crossing Weather API.

Visual Crossing provides excellent timezone handling with explicit UTC option
and proper DST support. This script downloads historical weather data with
accurate timestamps.

Variables:
- Temperature [°C]
- Wind Speed [m/s] 
- Wind Direction [°]
- Solar Radiation [W/m²]
- Humidity [%]
- Pressure [hPa]

Usage:
  python3 download_visual_crossing_weather.py --start 2014-01-01 --end 2023-12-31 --outfile data/amsterdam_vc_hourly.csv --apikey YOUR_API_KEY

API Key: Get free at https://www.visualcrossing.com/weather-api
Free tier: 1000 requests/day
"""

import argparse
import datetime as dt
import os
import time
from typing import Dict, Any
import pandas as pd
import requests

API_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
AMSTERDAM_COORDS = "52.3676,4.9041"  # lat,lon


def fetch_weather_data(
    start_date: dt.date,
    end_date: dt.date,
    api_key: str,
    coords: str = AMSTERDAM_COORDS,
    timezone: str = "Europe/Amsterdam",
) -> Dict[str, Any]:
    """
    Fetch weather data from Visual Crossing API.

    Args:
        start_date: Start date for data
        end_date: End date for data
        api_key: Visual Crossing API key
        coords: Latitude,longitude coordinates
        timezone: Timezone for timestamps (Europe/Amsterdam or UTC)

    Returns:
        JSON response from API
    """
    # Visual Crossing API expects date range in format: YYYY-MM-DD/YYYY-MM-DD
    date_range = f"{start_date.isoformat()}/{end_date.isoformat()}"

    params = {
        "location": coords,
        "startDateTime": start_date.isoformat(),
        "endDateTime": end_date.isoformat(),
        "unitGroup": "metric",  # Metric units
        "include": "hours",  # Hourly data
        "elements": "datetime,temp,humidity,pressure,windspeed,winddir,solarradiation,uvindex",
        "timezone": timezone,  # Explicit timezone handling
        "key": api_key,
    }

    print(f"Fetching data for {date_range} in timezone {timezone}...")

    try:
        response = requests.get(API_URL, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        raise


def process_weather_data(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Process Visual Crossing API response into DataFrame.

    Args:
        data: JSON response from API

    Returns:
        DataFrame with weather data
    """
    if "days" not in data:
        raise ValueError("No weather data found in API response")

    # Extract hourly data from all days
    hourly_data = []
    for day in data["days"]:
        if "hours" in day:
            for hour in day["hours"]:
                hourly_data.append(
                    {
                        "datetime": pd.to_datetime(
                            f"{day['datetime']} {hour['datetime']}"
                        ),
                        "temp_c": hour.get("temp"),
                        "humidity": hour.get("humidity"),
                        "pressure": hour.get("pressure"),
                        "wind_ms": hour.get("windspeed"),
                        "wind_dir_deg": hour.get("winddir"),
                        "solarradiation": hour.get("solarradiation"),
                        "uvindex": hour.get("uvindex"),
                    }
                )

    df = pd.DataFrame(hourly_data)

    # Remove rows with missing critical data
    df = df.dropna(subset=["temp_c", "wind_ms", "solarradiation"])

    # Rename columns to match our existing format
    df = df.rename(
        columns={"solarradiation": "ghi_wm2", "wind_ms": "wind_ms"}  # Already in m/s
    )

    # Convert wind direction to degrees if needed
    if "wind_dir_deg" in df.columns:
        df["wind_dir_deg"] = pd.to_numeric(df["wind_dir_deg"], errors="coerce")

    return df


def download_in_chunks(
    start_date: dt.date, end_date: dt.date, api_key: str, chunk_days: int = 30
) -> pd.DataFrame:
    """
    Download data in chunks to respect API limits.

    Args:
        start_date: Start date
        end_date: End date
        api_key: API key
        chunk_days: Number of days per chunk

    Returns:
        Combined DataFrame
    """
    all_data = []
    current_date = start_date

    while current_date <= end_date:
        chunk_end = min(current_date + dt.timedelta(days=chunk_days - 1), end_date)

        print(f"Downloading chunk: {current_date} to {chunk_end}")

        try:
            data = fetch_weather_data(current_date, chunk_end, api_key)
            df_chunk = process_weather_data(data)

            if not df_chunk.empty:
                all_data.append(df_chunk)
                print(f"  Downloaded {len(df_chunk)} records")
            else:
                print("  No data for this period")

        except Exception as e:
            print(f"  Error downloading chunk: {e}")
            continue

        # Rate limiting - be nice to the API
        time.sleep(1)

        current_date = chunk_end + dt.timedelta(days=1)

    if not all_data:
        raise RuntimeError("No data downloaded")

    # Combine all chunks
    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.sort_values("datetime").reset_index(drop=True)

    return df_all


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the downloaded data.

    Args:
        df: DataFrame to validate
    """
    print("\nData validation:")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print("  Missing values:")
    for col in df.columns:
        if col != "datetime":
            missing = df[col].isnull().sum()
            print(f"    {col}: {missing} ({missing/len(df)*100:.1f}%)")

    # Check for reasonable values
    print("\nData ranges:")
    print(f"  Temperature: {df['temp_c'].min():.1f} to {df['temp_c'].max():.1f} °C")
    print(f"  Wind speed: {df['wind_ms'].min():.1f} to {df['wind_ms'].max():.1f} m/s")
    print(
        f"  Solar radiation: {df['ghi_wm2'].min():.1f} to {df['ghi_wm2'].max():.1f} W/m²"
    )

    # Check timezone handling
    print("\nTimezone analysis:")
    print("  Sample timestamps:")
    for i in range(min(5, len(df))):
        print(f"    {df.iloc[i]['datetime']}")

    # Check for DST transitions (should see 23 or 25 hour days)
    daily_counts = df.groupby(df["datetime"].dt.date).size()
    dst_days = daily_counts[(daily_counts == 23) | (daily_counts == 25)]
    print(f"  DST transition days (23/25 hours): {len(dst_days)}")

    if len(dst_days) > 0:
        print("  ✅ DST handling appears correct")
    else:
        print("  ⚠️  No DST transition days found - may indicate timezone issues")


def main():
    parser = argparse.ArgumentParser(
        description="Download Amsterdam weather data from Visual Crossing"
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--outfile", default="data/amsterdam_vc_hourly.csv", help="Output CSV file"
    )
    parser.add_argument("--apikey", required=True, help="Visual Crossing API key")
    parser.add_argument(
        "--timezone",
        default="Europe/Amsterdam",
        help="Timezone (Europe/Amsterdam or UTC)",
    )
    parser.add_argument(
        "--chunk-days", type=int, default=30, help="Days per API request (default: 30)"
    )

    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)

    if end_date < start_date:
        raise ValueError("End date must be >= start date")

    print("Downloading weather data for Amsterdam")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Timezone: {args.timezone}")
    print(f"API Key: {args.apikey[:8]}...")

    # Download data
    df = download_in_chunks(start_date, end_date, args.apikey, args.chunk_days)

    # Validate data
    validate_data(df)

    # Save to CSV
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    df.to_csv(args.outfile, index=False)

    print(f"\n✅ Data saved to {args.outfile}")
    print(f"   Records: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
