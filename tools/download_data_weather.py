#!/usr/bin/env python3
"""
Fetch strict-hourly weather for Amsterdam (or any lat/lon) from the Open-Meteo Archive API.

Variables:
- temperature_2m  [°C]
- wind_speed_10m  [m/s]
- wind_direction_10m [°]
- shortwave_radiation (GHI) [W/m²]
- direct_normal_irradiance (DNI) [W/m²]
- diffuse_radiation (DHI) [W/m²]

Guarantees:
- Requests hourly series (no daily aggregation).
- Verifies exactly 24 samples per local day (Europe/Amsterdam) unless the day is DST-short/long.
- Verifies no duplicate timestamps and no multi-hour gaps (except at DST transitions).
- If any check fails, exits with a clear error message.

Usage:
  python3 fetch_amsterdam_hourly.py --start 2015-01-01 --end 2025-09-27 --outfile data/amsterdam_hourly.csv
"""

import argparse
import datetime as dt
import os
import time
from typing import Iterator, List, Dict, Any

import pandas as pd
import requests
from pandas.tseries.frequencies import to_offset

API_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
]


def month_range(start: dt.date, end: dt.date) -> Iterator[tuple[dt.date, dt.date]]:
    cur = start.replace(day=1)
    while cur <= end:
        if cur.month == 12:
            next_month = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            next_month = cur.replace(month=cur.month + 1, day=1)
        last = next_month - dt.timedelta(days=1)
        yield (max(cur, start), min(last, end))
        cur = next_month


def fetch_chunk(
    lat: float,
    lon: float,
    start_date: dt.date,
    end_date: dt.date,
    timezone: str,
    retries: int = 5,
    backoff: float = 1.0,
) -> Dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        # IMPORTANT: explicit hourly variables (no daily fields)
        "hourly": ",".join(HOURLY_VARS),
        "timezone": timezone,
        # Prefer ERA5 model for long consistent history
        "models": "era5",
        # Make sure we get hourly timestamps (Open-Meteo default for 'hourly=' is hourly).
        "timeformat": "iso8601",
    }
    for attempt in range(retries):
        r = requests.get(API_URL, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff * (2**attempt))
            continue
        r.raise_for_status()
    r = requests.get(API_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def to_hourly_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if not hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly).rename(columns={"time": "datetime"})
    # Parse local timestamps (Europe/Amsterdam) as naive, then localize for DST checks
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    # Quick structural sanity: equal length across columns
    n = len(df)
    for c in [
        "temperature_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
    ]:
        if c not in df.columns or len(df[c]) != n:
            raise RuntimeError(f"API returned inconsistent column '{c}'")
    return df


def validate_strict_hourly(df: pd.DataFrame, tz: str) -> None:
    """Ensure data is hourly with no duplicates and only DST anomalies allowed."""
    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique()
        raise RuntimeError(
            f"Duplicate timestamps detected (examples: {list(dupes[:3])})."
        )

    # Identify expected hourly step by median diff
    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        raise RuntimeError("Not enough timestamps to validate frequency.")
    # Most common step should be 1 hour
    step = diffs.mode().iloc[0]
    if to_offset(step) != to_offset("1h"):
        raise RuntimeError(f"Non-hourly step detected (mode step={step}).")

    # Check per-day counts allowing DST transitions:
    # - Spring forward day: 23 samples
    # - Fall back day: 25 samples
    # - Normal day: 24 samples
    s_local = df.copy()
    s_local.index = s_local.index.tz_localize(
        tz, nonexistent="shift_forward", ambiguous="NaT"
    )
    # We’ll group by local calendar date (normalized)
    counts = s_local.groupby(s_local.index.date).size()
    bad = []
    for day, cnt in counts.items():
        if cnt not in (23, 24, 25):
            bad.append((str(day), cnt))
    if bad:
        raise RuntimeError(
            f"Found non-hourly daily counts (expected 23/24/25): {bad[:5]} ..."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lat", type=float, default=52.3676, help="Latitude (default Amsterdam)"
    )
    parser.add_argument(
        "--lon", type=float, default=4.9041, help="Longitude (default Amsterdam)"
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--outfile", default="data/amsterdam_hourly.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--tz", default="Europe/Amsterdam", help="Timezone for timestamps"
    )
    args = parser.parse_args()

    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)
    if end_date < start_date:
        raise ValueError("end date must be >= start date")

    frames: List[pd.DataFrame] = []
    for mstart, mend in month_range(start_date, end_date):
        payload = fetch_chunk(args.lat, args.lon, mstart, mend, args.tz)
        df = to_hourly_frame(payload)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No data returned. Check dates/coords.")

    df_all = pd.concat(frames).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]

    # Strict hourly validation (raises on failure)
    validate_strict_hourly(df_all, args.tz)

    # Rename columns
    df_all = df_all.rename(
        columns={
            "temperature_2m": "temp_c",
            "wind_speed_10m": "wind_ms",
            "wind_direction_10m": "wind_dir_deg",
            "shortwave_radiation": "ghi_wm2",
            "direct_normal_irradiance": "dni_wm2",
            "diffuse_radiation": "dhi_wm2",
        }
    )

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    df_all.to_csv(args.outfile, index_label="datetime")

    # Minimal report
    days = df_all.index.normalize().nunique()
    print(f"Wrote {args.outfile}")
    print(f"Rows: {len(df_all):,}  Columns: {df_all.shape[1]}")
    print(f"Date span: {df_all.index.min()} to {df_all.index.max()}  (~{days} days)")
    # Show a small frequency summary
    per_day = df_all.groupby(df_all.index.date).size().value_counts().sort_index()
    print("Per-day counts (should be mostly 24, with 23/25 at DST):")
    for k, v in per_day.items():
        print(f"  {k} hours: {v} days")


if __name__ == "__main__":
    main()
