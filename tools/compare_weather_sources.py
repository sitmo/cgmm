#!/usr/bin/env python3
"""
Compare weather data from different sources to validate timezone handling.

This script downloads a small sample from multiple sources and compares
their timezone handling and data quality.

Usage:
  python3 compare_weather_sources.py --start 2023-03-25 --end 2023-03-27 --vc-apikey YOUR_VC_KEY
"""

import argparse
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_timezone_handling(df: pd.DataFrame, source_name: str) -> dict:
    """
    Analyze timezone handling in weather data.

    Args:
        df: DataFrame with weather data
        source_name: Name of the data source

    Returns:
        Dictionary with analysis results
    """
    print(f"\n=== {source_name.upper()} ANALYSIS ===")

    # Basic info
    print(f"Records: {len(df)}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Check for DST transitions
    daily_counts = df.groupby(df["datetime"].dt.date).size()
    dst_days = daily_counts[(daily_counts == 23) | (daily_counts == 25)]

    print(f"DST transition days: {len(dst_days)}")
    if len(dst_days) > 0:
        print(f"  DST dates: {list(dst_days.index)}")
        print("  ✅ Proper DST handling detected")
    else:
        print("  ❌ No DST transitions - possible timezone issue")

    # Analyze sunrise/sunset patterns
    if "ghi_wm2" in df.columns:
        # Find sunrise/sunset based on solar radiation
        def find_sunrise_sunset(day_data, threshold=10):
            day_data = day_data.sort_values("hour")
            sunrise_hours = day_data[day_data["ghi_wm2"] > threshold]["hour"]
            sunrise = sunrise_hours.min() if not sunrise_hours.empty else None
            sunset_hours = day_data[day_data["ghi_wm2"] > threshold]["hour"]
            sunset = sunset_hours.max() if not sunset_hours.empty else None
            return sunrise, sunset

        # Add hour column if not present
        if "hour" not in df.columns:
            df["hour"] = df["datetime"].dt.hour

        # Analyze each day
        sunrise_times = []
        sunset_times = []

        for date in df["datetime"].dt.date.unique():
            day_data = df[df["datetime"].dt.date == date]
            sunrise, sunset = find_sunrise_sunset(day_data)
            if sunrise is not None and sunset is not None:
                sunrise_times.append(sunrise)
                sunset_times.append(sunset)

        if sunrise_times:
            avg_sunrise = np.mean(sunrise_times)
            avg_sunset = np.mean(sunset_times)
            print(f"Average sunrise: {avg_sunrise:.1f}:00")
            print(f"Average sunset: {avg_sunset:.1f}:00")
            print(f"Day length: {avg_sunset - avg_sunrise:.1f} hours")

            # Determine if this matches UTC or local time
            # Expected for Amsterdam in March:
            # UTC: sunrise ~06:30, sunset ~18:30
            # Local (CET): sunrise ~07:30, sunset ~19:30
            utc_sunrise_diff = abs(avg_sunrise - 6.5)
            local_sunrise_diff = abs(avg_sunrise - 7.5)

            if utc_sunrise_diff < local_sunrise_diff:
                print("  → Appears to be UTC time")
            else:
                print("  → Appears to be local time")

    return {
        "source": source_name,
        "records": len(df),
        "dst_days": len(dst_days),
        "avg_sunrise": (
            avg_sunrise if "sunrise_times" in locals() and sunrise_times else None
        ),
        "avg_sunset": (
            avg_sunset if "sunset_times" in locals() and sunset_times else None
        ),
    }


def create_comparison_plot(dfs: dict, output_file: str = None):
    """
    Create comparison plots for different data sources.

    Args:
        dfs: Dictionary of DataFrames {source_name: df}
        output_file: Output file for plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    colors = ["blue", "red", "green", "orange"]

    for i, (source_name, df) in enumerate(dfs.items()):
        color = colors[i % len(colors)]

        # Plot 1: Temperature over time
        axes[0, 0].plot(
            df["datetime"],
            df["temp_c"],
            color=color,
            alpha=0.7,
            label=source_name,
            linewidth=1,
        )
        axes[0, 0].set_title("Temperature Comparison")
        axes[0, 0].set_ylabel("Temperature (°C)")
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Plot 2: Solar radiation over time
        if "ghi_wm2" in df.columns:
            axes[0, 1].plot(
                df["datetime"],
                df["ghi_wm2"],
                color=color,
                alpha=0.7,
                label=source_name,
                linewidth=1,
            )
            axes[0, 1].set_title("Solar Radiation Comparison")
            axes[0, 1].set_ylabel("Solar Radiation (W/m²)")
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis="x", rotation=45)

        # Plot 3: Hourly patterns
        if "hour" not in df.columns:
            df["hour"] = df["datetime"].dt.hour

        hourly_temp = df.groupby("hour")["temp_c"].mean()
        axes[1, 0].plot(
            hourly_temp.index,
            hourly_temp.values,
            color=color,
            marker="o",
            label=source_name,
            linewidth=2,
        )
        axes[1, 0].set_title("Average Hourly Temperature Pattern")
        axes[1, 0].set_xlabel("Hour of Day")
        axes[1, 0].set_ylabel("Temperature (°C)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Solar radiation hourly patterns
        if "ghi_wm2" in df.columns:
            hourly_solar = df.groupby("hour")["ghi_wm2"].mean()
            axes[1, 1].plot(
                hourly_solar.index,
                hourly_solar.values,
                color=color,
                marker="o",
                label=source_name,
                linewidth=2,
            )
            axes[1, 1].set_title("Average Hourly Solar Radiation Pattern")
            axes[1, 1].set_xlabel("Hour of Day")
            axes[1, 1].set_ylabel("Solar Radiation (W/m²)")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare weather data sources")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--vc-apikey", help="Visual Crossing API key")
    parser.add_argument("--plot", help="Output file for comparison plot")

    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)

    print("Weather Data Source Comparison")
    print(f"Date range: {start_date} to {end_date}")
    print("Focus: DST transition period (March 25-27, 2023)")

    results = {}
    dfs = {}

    # Test 1: KNMI data
    try:
        print(f"\n{'='*50}")
        print("Testing KNMI data...")

        # Import the KNMI download function
        import sys

        sys.path.append(".")
        from tools.download_knmi_weather import download_knmi_data

        df_knmi = download_knmi_data(start_date, end_date)
        results["KNMI"] = analyze_timezone_handling(df_knmi, "KNMI")
        dfs["KNMI"] = df_knmi

    except Exception as e:
        print(f"Error downloading KNMI data: {e}")

    # Test 2: Visual Crossing data (if API key provided)
    if args.vc_apikey:
        try:
            print(f"\n{'='*50}")
            print("Testing Visual Crossing data...")

            # Import the VC download function
            from tools.download_visual_crossing_weather import download_in_chunks

            df_vc = download_in_chunks(
                start_date, end_date, args.vc_apikey, chunk_days=5
            )
            results["Visual Crossing"] = analyze_timezone_handling(
                df_vc, "Visual Crossing"
            )
            dfs["Visual Crossing"] = df_vc

        except Exception as e:
            print(f"Error downloading Visual Crossing data: {e}")

    # Test 3: Original Open-Meteo data (if available)
    try:
        print(f"\n{'='*50}")
        print("Testing original Open-Meteo data...")

        # Try to load existing data
        df_om = pd.read_csv("docs/examples/data/amsterdam_hourly.csv")
        df_om["datetime"] = pd.to_datetime(df_om["datetime"])

        # Filter to date range
        df_om = df_om[
            (df_om["datetime"].dt.date >= start_date)
            & (df_om["datetime"].dt.date <= end_date)
        ]

        if not df_om.empty:
            results["Open-Meteo"] = analyze_timezone_handling(df_om, "Open-Meteo")
            dfs["Open-Meteo"] = df_om
        else:
            print("No Open-Meteo data for this date range")

    except Exception as e:
        print(f"Error loading Open-Meteo data: {e}")

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")

    for source, result in results.items():
        print(f"\n{source}:")
        print(f"  Records: {result['records']}")
        print(f"  DST days: {result['dst_days']}")
        if result["avg_sunrise"]:
            print(f"  Avg sunrise: {result['avg_sunrise']:.1f}:00")
            print(f"  Avg sunset: {result['avg_sunset']:.1f}:00")

    # Create comparison plot
    if len(dfs) > 1:
        create_comparison_plot(dfs, args.plot)

    print("\nRecommendation:")
    best_source = max(results.keys(), key=lambda x: results[x]["dst_days"])
    print(
        f"  Best timezone handling: {best_source} ({results[best_source]['dst_days']} DST days)"
    )


if __name__ == "__main__":
    main()
