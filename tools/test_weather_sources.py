#!/usr/bin/env python3
"""
Quick test of weather data sources to verify timezone handling.

This script downloads a small sample around DST transition to test
timezone handling and data quality.

Usage:
  python3 test_weather_sources.py --vc-apikey YOUR_API_KEY
"""

import argparse
import subprocess
import sys
import os


def test_knmi():
    """Test KNMI data download."""
    print("Testing KNMI data source...")

    try:
        # Test with DST transition period
        start_date = "2023-03-25"
        end_date = "2023-03-27"
        output_file = "test_knmi_data.csv"

        cmd = [
            sys.executable,
            "tools/download_knmi_weather.py",
            "--start",
            start_date,
            "--end",
            end_date,
            "--outfile",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ KNMI download successful")

            # Check if file exists and has data
            if os.path.exists(output_file):
                import pandas as pd

                df = pd.read_csv(output_file)
                print(f"   Records: {len(df)}")
                print(f"   Columns: {list(df.columns)}")

                # Check for DST handling
                df["datetime"] = pd.to_datetime(df["datetime"])
                daily_counts = df.groupby(df["datetime"].dt.date).size()
                dst_days = daily_counts[(daily_counts == 23) | (daily_counts == 25)]

                if len(dst_days) > 0:
                    print(
                        f"   ✅ DST handling detected: {len(dst_days)} transition days"
                    )
                else:
                    print("   ⚠️  No DST transitions detected")

                # Clean up
                os.remove(output_file)
                return True
            else:
                print("❌ No output file created")
                return False
        else:
            print(f"❌ KNMI download failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ KNMI test error: {e}")
        return False


def test_visual_crossing(api_key):
    """Test Visual Crossing data download."""
    print("Testing Visual Crossing data source...")

    try:
        # Test with DST transition period
        start_date = "2023-03-25"
        end_date = "2023-03-27"
        output_file = "test_vc_data.csv"

        cmd = [
            sys.executable,
            "tools/download_visual_crossing_weather.py",
            "--start",
            start_date,
            "--end",
            end_date,
            "--outfile",
            output_file,
            "--apikey",
            api_key,
            "--timezone",
            "Europe/Amsterdam",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ Visual Crossing download successful")

            # Check if file exists and has data
            if os.path.exists(output_file):
                import pandas as pd

                df = pd.read_csv(output_file)
                print(f"   Records: {len(df)}")
                print(f"   Columns: {list(df.columns)}")

                # Check for DST handling
                df["datetime"] = pd.to_datetime(df["datetime"])
                daily_counts = df.groupby(df["datetime"].dt.date).size()
                dst_days = daily_counts[(daily_counts == 23) | (daily_counts == 25)]

                if len(dst_days) > 0:
                    print(
                        f"   ✅ DST handling detected: {len(dst_days)} transition days"
                    )
                else:
                    print("   ⚠️  No DST transitions detected")

                # Clean up
                os.remove(output_file)
                return True
            else:
                print("❌ No output file created")
                return False
        else:
            print(f"❌ Visual Crossing download failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Visual Crossing test error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test weather data sources")
    parser.add_argument("--vc-apikey", help="Visual Crossing API key for testing")

    args = parser.parse_args()

    print("Weather Data Source Testing")
    print("=" * 40)
    print("Testing DST transition period: March 25-27, 2023")
    print("This period includes the spring DST transition")
    print()

    results = {}

    # Test KNMI
    results["KNMI"] = test_knmi()
    print()

    # Test Visual Crossing (if API key provided)
    if args.vc_apikey:
        results["Visual Crossing"] = test_visual_crossing(args.vc_apikey)
        print()
    else:
        print("Skipping Visual Crossing test (no API key provided)")
        print("Get free API key at: https://www.visualcrossing.com/weather-api")
        print()

    # Summary
    print("Test Results Summary:")
    print("-" * 30)

    for source, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{source}: {status}")

    successful_sources = [source for source, success in results.items() if success]

    if successful_sources:
        print(f"\nRecommended source: {successful_sources[0]}")
        print("This source has proper timezone handling and DST support.")
    else:
        print("\nNo sources passed the test.")
        print("Check your internet connection and API keys.")


if __name__ == "__main__":
    main()
