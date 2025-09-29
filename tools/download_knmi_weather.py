#!/usr/bin/env python3
"""
Download hourly weather data from KNMI (Royal Netherlands Meteorological Institute).

KNMI is the official Dutch weather service and provides the most reliable
data for the Netherlands.

Data source: https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens
Station: Amsterdam Schiphol (240)

IMPORTANT: Data is in UTC (Universal Time)
- Amsterdam local time: UTC+1 (winter) / UTC+2 (summer)
- No DST transitions in UTC data (as expected)
- Hour range: 0-23 (converted from KNMI's 1-24 format)

Column meanings and units (after conversion):
- datetime: Timestamp in UTC
- temp_c: Temperature in °C (converted from 0.1°C, measured at 1.50m height)
- wind_ms: Wind speed in m/s (converted from 0.1 m/s, 10-minute average)
- ghi_wm2: Global horizontal irradiance in W/m² (converted from J/cm² per hour)

Raw KNMI columns:
- FF: Windsnelheid (in 0.1 m/s) gemiddeld over de laatste 10 minuten
- T: Temperatuur (in 0.1 graden Celsius) op 1.50 m hoogte
- Q: Globale straling (in J/cm2) per uurvak

Usage:
  python3 download_knmi_weather.py --start 2014-01-01 --end 2023-12-31 --outfile data/amsterdam_knmi_hourly.csv
"""

import argparse
import datetime as dt
import os
import time
import pandas as pd
import requests
from io import StringIO

KNMI_BASE_URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
AMSTERDAM_STATION = "240"  # Amsterdam Schiphol


def fetch_knmi_data(
    start_date: dt.date, end_date: dt.date, station: str = AMSTERDAM_STATION
) -> str:
    """
    Fetch hourly weather data from KNMI.

    Args:
        start_date: Start date for data
        end_date: End date for data
        station: KNMI station number (240 = Amsterdam Schiphol)

    Returns:
        CSV data as string
    """
    # KNMI API parameters
    params = {
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "stns": station,
        "vars": "ALL",  # All available variables
    }

    print(
        f"Fetching KNMI data for station {station} from {start_date} to {end_date}..."
    )

    try:
        response = requests.get(KNMI_BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching KNMI data: {e}")
        raise


def process_knmi_data(csv_data: str) -> pd.DataFrame:
    """
    Process KNMI CSV data into DataFrame.

    Args:
        csv_data: Raw CSV data from KNMI

    Returns:
        DataFrame with weather data
    """
    # KNMI CSV has specific format - skip header lines and parse
    lines = csv_data.strip().split("\n")

    # Find the data section (skip comments and headers)
    data_lines = []
    in_data_section = False
    header_line = None

    for line in lines:
        if line.startswith("# STN,YYYYMMDD,HH"):
            in_data_section = True
            header_line = line
            continue
        elif line.startswith("#"):
            continue
        elif in_data_section and line.strip():
            data_lines.append(line)

    if not data_lines:
        raise ValueError("No data found in KNMI response")

    if not header_line:
        raise ValueError("No header line found in KNMI response")

    # Extract column names from header
    columns = header_line.replace("# ", "").split(",")

    # Parse CSV with proper column names
    df_raw = pd.read_csv(
        StringIO("\n".join(data_lines)), sep=",", names=columns, low_memory=False
    )

    # Trim whitespace from column names
    df_raw.columns = df_raw.columns.str.strip()

    # Validate required columns exist
    required_columns = ["YYYYMMDD", "HH", "FH", "T", "Q"]
    missing_columns = [col for col in required_columns if col not in df_raw.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. Available: {list(df_raw.columns)}"
        )

    # Select only the columns we need
    selected_columns = ["YYYYMMDD", "HH", "FH", "T", "Q"]
    df_selected = df_raw[selected_columns].copy()

    # Rename to our standard names
    df_selected = df_selected.rename(
        columns={
            "YYYYMMDD": "date",
            "HH": "hour",
            "FH": "wind_ms",
            "T": "temp_c",
            "Q": "ghi_wm2",
        }
    )

    # Convert temperature from 0.1°C to °C (measured at 1.50m height)
    df_selected["temp_c"] = pd.to_numeric(df_selected["temp_c"], errors="coerce") / 10.0

    # Convert wind speed from 0.1 m/s to m/s (10-minute average)
    df_selected["wind_ms"] = (
        pd.to_numeric(df_selected["wind_ms"], errors="coerce") / 10.0
    )

    # Convert global radiation from J/cm² to W/m² (hourly)
    # J/cm² per hour → W/m²: multiply by 10000 (cm² to m²) and divide by 3600 (seconds in hour)
    df_selected["ghi_wm2"] = (
        pd.to_numeric(df_selected["ghi_wm2"], errors="coerce") * 10000 / 3600
    )
    # Round to 1 decimal place to avoid floating point precision issues
    df_selected["ghi_wm2"] = df_selected["ghi_wm2"].round(1)

    # Create datetime column
    df_selected["date"] = df_selected["date"].astype(str)
    df_selected["hour"] = df_selected["hour"].astype(int)

    # Convert hour range 1-24 to 0-23 by subtracting 1
    # This handles the KNMI data format where hours are 1-24 instead of 0-23
    df_selected["hour"] = df_selected["hour"] - 1

    # Convert hour to string with zero padding
    df_selected["hour"] = df_selected["hour"].astype(str).str.zfill(2)

    # Create datetime
    df_selected["datetime"] = pd.to_datetime(
        df_selected["date"] + df_selected["hour"], format="%Y%m%d%H", errors="coerce"
    )

    # Check for any failed datetime conversions
    failed_conversions = df_selected["datetime"].isna().sum()
    if failed_conversions > 0:
        print(f"Warning: {failed_conversions} datetime conversions failed")
        # Show some examples of failed conversions
        failed_examples = df_selected[df_selected["datetime"].isna()][
            ["date", "hour"]
        ].head()
        print("Failed conversion examples:")
        print(failed_examples)

    # Remove rows with missing critical data
    df_clean = df_selected.dropna(subset=["temp_c", "wind_ms", "ghi_wm2"])

    # Select final columns
    final_columns = ["datetime", "temp_c", "wind_ms", "ghi_wm2"]
    df_final = df_clean[final_columns]

    return df_final


def validate_knmi_data(df: pd.DataFrame) -> None:
    """
    Validate the KNMI data.

    Args:
        df: DataFrame to validate
    """
    print("\nKNMI Data validation:")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Check for reasonable values
    print("\nData ranges (after unit conversion):")
    print(
        f"  Temperature: {df['temp_c'].min():.1f} to {df['temp_c'].max():.1f} °C (at 1.50m height)"
    )
    print(
        f"  Wind speed: {df['wind_ms'].min():.1f} to {df['wind_ms'].max():.1f} m/s (10-min average)"
    )
    print(
        f"  Global irradiance: {df['ghi_wm2'].min():.1f} to {df['ghi_wm2'].max():.1f} W/m² (hourly)"
    )

    # Check data consistency (should be 24 hours per day in UTC)
    daily_counts = df.groupby(df["datetime"].dt.date).size()
    print("\nData consistency:")
    print("  Records per day (should be 24 for UTC data):")
    daily_counts_summary = daily_counts.value_counts().sort_index()
    for count, days in daily_counts_summary.head(5).items():
        print(f"    {count} hours: {days} days")

    # Note about time convention
    print("\nTime convention:")
    print("  Data is in UTC (Universal Time)")
    print("  Amsterdam local time: UTC+1 (winter) / UTC+2 (summer)")
    print("  No DST transitions in UTC data (as expected)")

    # Check data quality
    print("\nData quality:")
    daily_counts_summary = daily_counts.value_counts().sort_index()
    for count, days in daily_counts_summary.head(5).items():
        print(f"  {count} hours: {days} days")


def download_knmi_data(
    start_date: dt.date, end_date: dt.date, station: str = AMSTERDAM_STATION
) -> pd.DataFrame:
    """
    Download KNMI data for the specified date range.

    Args:
        start_date: Start date
        end_date: End date
        station: KNMI station number

    Returns:
        DataFrame with weather data
    """
    # KNMI has limits on date ranges, so we might need to chunk
    # But let's try the full range first
    try:
        csv_data = fetch_knmi_data(start_date, end_date, station)
        df = process_knmi_data(csv_data)
        return df
    except Exception as e:
        print(f"Error downloading full range: {e}")
        print("Trying to download in smaller chunks...")

        # Download in monthly chunks
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

            print(f"Downloading chunk: {current_date} to {chunk_end}")

            try:
                csv_data = fetch_knmi_data(current_date, chunk_end, station)
                df_chunk = process_knmi_data(csv_data)

                if not df_chunk.empty:
                    all_data.append(df_chunk)
                    print(f"  Downloaded {len(df_chunk)} records")

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"  Error downloading chunk: {e}")
                continue

            current_date = chunk_end + dt.timedelta(days=1)

        if not all_data:
            raise RuntimeError("No data downloaded")

        # Combine all chunks
        df_all = pd.concat(all_data, ignore_index=True)
        df_all = df_all.sort_values("datetime").reset_index(drop=True)

        return df_all


def main():
    parser = argparse.ArgumentParser(
        description="Download Amsterdam weather data from KNMI"
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--outfile", default="data/amsterdam_knmi_hourly.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--station", default="240", help="KNMI station number (240=Amsterdam Schiphol)"
    )

    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)

    if end_date < start_date:
        raise ValueError("End date must be >= start date")

    print("Downloading KNMI weather data for Amsterdam")
    print(f"Station: {args.station} (Amsterdam Schiphol)")
    print(f"Date range: {start_date} to {end_date}")

    # Download data
    df = download_knmi_data(start_date, end_date, args.station)

    # Validate data
    validate_knmi_data(df)

    # Save to CSV
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    df.to_csv(args.outfile, index=False)

    print(f"\n✅ Data saved to {args.outfile}")
    print(f"   Records: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
