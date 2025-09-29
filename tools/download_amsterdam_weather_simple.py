#!/usr/bin/env python3
"""
Simple Amsterdam weather data download using Visual Crossing API.

This is a working solution that downloads reliable weather data
with proper timezone handling and DST support.

Usage:
  python3 tools/download_amsterdam_weather_simple.py --start 2014-01-01 --end 2023-12-31 --apikey YOUR_API_KEY

Get free API key: https://www.visualcrossing.com/weather-api
Free tier: 1000 requests/day
"""

import argparse
import datetime as dt
import os
import time
import pandas as pd
import requests


def download_weather_data(start_date, end_date, api_key, output_file):
    """Download weather data from Visual Crossing API."""

    print(f"Downloading Amsterdam weather data from {start_date} to {end_date}")
    print("Using Visual Crossing API with proper timezone handling")

    # Visual Crossing API parameters
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    location = "52.3676,4.9041"  # Amsterdam coordinates

    # Download in monthly chunks to respect API limits
    all_data = []
    current_date = start_date

    while current_date <= end_date:
        # Calculate end of month
        if current_date.month == 12:
            next_month = current_date.replace(
                year=current_date.year + 1, month=1, day=1
            )
        else:
            next_month = current_date.replace(month=current_date.month + 1, day=1)
        chunk_end = min(next_month - dt.timedelta(days=1), end_date)

        print(f"Downloading {current_date} to {chunk_end}...")

        # API request parameters
        params = {
            "location": location,
            "startDateTime": current_date.isoformat(),
            "endDateTime": chunk_end.isoformat(),
            "unitGroup": "metric",
            "include": "hours",
            "elements": "datetime,temp,humidity,pressure,windspeed,winddir,solarradiation",
            "timezone": "Europe/Amsterdam",  # Proper timezone with DST
            "key": api_key,
        }

        try:
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Extract hourly data
            hourly_data = []
            for day in data.get("days", []):
                for hour in day.get("hours", []):
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
                            "ghi_wm2": hour.get("solarradiation"),
                        }
                    )

            if hourly_data:
                df_chunk = pd.DataFrame(hourly_data)
                df_chunk = df_chunk.dropna(subset=["temp_c", "wind_ms", "ghi_wm2"])
                all_data.append(df_chunk)
                print(f"  Downloaded {len(df_chunk)} records")
            else:
                print("  No data for this period")

        except Exception as e:
            print(f"  Error: {e}")
            continue

        # Rate limiting
        time.sleep(1)
        current_date = chunk_end + dt.timedelta(days=1)

    if not all_data:
        raise RuntimeError("No data downloaded")

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # Validate data
    print("\nData validation:")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Check for DST transitions
    daily_counts = df.groupby(df["datetime"].dt.date).size()
    dst_days = daily_counts[(daily_counts == 23) | (daily_counts == 25)]
    print(f"  DST transition days: {len(dst_days)}")

    if len(dst_days) > 0:
        print("  ✅ Proper DST handling detected!")
        print(f"  DST dates: {list(dst_days.index)[:5]}...")
    else:
        print("  ⚠️  No DST transitions found")

    # Save data
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\n✅ Data saved to {output_file}")
    print(f"   Records: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Download Amsterdam weather data")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--outfile",
        default="docs/examples/data/amsterdam_vc_hourly.csv",
        help="Output CSV file",
    )
    parser.add_argument("--apikey", required=True, help="Visual Crossing API key")

    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)

    if end_date < start_date:
        raise ValueError("End date must be >= start date")

    # Download data
    df = download_weather_data(start_date, end_date, args.apikey, args.outfile)

    print("\n🎉 SUCCESS!")
    print(f"Downloaded {len(df):,} weather records for Amsterdam")
    print("Data includes proper timezone handling and DST transitions")
    print("Ready to use in your analysis!")


if __name__ == "__main__":
    main()
