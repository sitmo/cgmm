# Weather Data Sources for Amsterdam

This directory contains scripts to download reliable weather data for Amsterdam with proper timezone handling and DST support.

## Problem with Open-Meteo

The original Open-Meteo API has known issues with Daylight Saving Time (DST) handling:
- Returns static time offsets instead of proper DST transitions
- Timestamps don't reflect actual local time changes
- No 23/25-hour days during DST transitions
- Sunrise/sunset times don't shift as expected

## Alternative Data Sources

### 1. KNMI (Royal Netherlands Meteorological Institute) ⭐ **RECOMMENDED**

**Best choice for Dutch weather data**

- **Official Dutch weather service**
- **Proper timezone handling with DST**
- **Free access to historical data**
- **High data quality and reliability**

**Usage:**
```bash
python3 tools/download_knmi_weather.py --start 2014-01-01 --end 2023-12-31 --outfile data/amsterdam_knmi_hourly.csv
```

**Variables:**
- Temperature [°C]
- Wind Speed [m/s]
- Wind Direction [°]
- Solar Radiation [W/m²] (converted from J/cm²)
- Humidity [%]
- Pressure [hPa]

### 2. Visual Crossing Weather API

**Excellent timezone handling with explicit UTC option**

- **Proper DST handling**
- **Explicit UTC option** (`&timezone=Z`)
- **High-quality data**
- **Free tier: 1000 requests/day**

**Get API Key:** https://www.visualcrossing.com/weather-api

**Usage:**
```bash
python3 tools/download_visual_crossing_weather.py --start 2014-01-01 --end 2023-12-31 --outfile data/amsterdam_vc_hourly.csv --apikey YOUR_API_KEY
```

**Variables:**
- Temperature [°C]
- Wind Speed [m/s]
- Wind Direction [°]
- Solar Radiation [W/m²]
- Humidity [%]
- Pressure [hPa]

## Testing and Comparison

### Quick Test
Test both sources with a small sample:
```bash
python3 tools/test_weather_sources.py --vc-apikey YOUR_API_KEY
```

### Full Comparison
Compare data sources around DST transitions:
```bash
python3 tools/compare_weather_sources.py --start 2023-03-25 --end 2023-03-27 --vc-apikey YOUR_API_KEY --plot comparison.png
```

## Expected Results

**Proper DST Handling:**
- 23-hour days during spring DST transition
- 25-hour days during fall DST transition
- Sunrise/sunset times shift by 1 hour
- Solar radiation patterns follow local time

**Data Quality:**
- Consistent hourly intervals
- Realistic temperature ranges (-20°C to 40°C)
- Solar radiation follows daily patterns
- Wind data shows expected patterns

## Migration from Open-Meteo

To replace your existing Open-Meteo data:

1. **Download new data:**
   ```bash
   # KNMI (recommended)
   python3 tools/download_knmi_weather.py --start 2014-01-01 --end 2023-12-31 --outfile data/amsterdam_knmi_hourly.csv
   
   # Or Visual Crossing
   python3 tools/download_visual_crossing_weather.py --start 2014-01-01 --end 2023-12-31 --outfile data/amsterdam_vc_hourly.csv --apikey YOUR_API_KEY
   ```

2. **Update your analysis:**
   - Replace `df_clean` with the new dataset
   - Verify column names match your existing code
   - Test DST handling with the new data

3. **Verify improvements:**
   - Check for 23/25-hour days during DST transitions
   - Verify sunrise/sunset times shift correctly
   - Confirm solar radiation patterns follow local time

## File Structure

```
tools/
├── download_knmi_weather.py          # KNMI data download
├── download_visual_crossing_weather.py # Visual Crossing download
├── compare_weather_sources.py        # Compare data sources
├── test_weather_sources.py           # Quick test script
└── README_weather_sources.md         # This file
```

## Troubleshooting

**KNMI Issues:**
- Check internet connection
- Verify date ranges are valid
- KNMI may have rate limits

**Visual Crossing Issues:**
- Verify API key is valid
- Check free tier limits (1000 requests/day)
- Ensure proper timezone parameter

**Data Quality Issues:**
- Check for missing values
- Verify timezone handling
- Compare with expected patterns

## Next Steps

1. **Test the sources** with the provided scripts
2. **Choose your preferred source** (KNMI recommended)
3. **Download full dataset** for your analysis period
4. **Update your notebooks** to use the new data
5. **Verify DST handling** in your analysis

The new data sources will provide accurate timezone handling and proper DST transitions, resolving the issues you encountered with Open-Meteo.
