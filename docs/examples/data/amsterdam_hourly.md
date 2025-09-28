# Amsterdam Hourly Weather Data

This dataset contains **hourly historical weather data** for Amsterdam, The Netherlands, retrieved via the [Open-Meteo Historical API](https://open-meteo.com/).  
The data is derived from the **ERA5 reanalysis** (ECMWF Copernicus Climate Data Store), which provides consistent global coverage since 1950. ERA5 combines satellite observations, weather stations, and models into a continuous dataset.

- **Location:** Amsterdam, NL (lat 52.3676, lon 4.9041)  
- **Time zone:** Europe/Amsterdam (local time, DST-adjusted)  
- **Resolution:** Hourly  

## Columns

| Column         | Unit   | Description |
|----------------|--------|-------------|
| `datetime`     | ISO8601 (local time) | Hourly timestamp in Europe/Amsterdam time zone. |
| `temp_c`       | °C     | Air temperature at 2 m above ground. |
| `wind_ms`      | m/s    | Wind speed at 10 m above ground. |
| `wind_dir_deg` | °      | Wind direction at 10 m above ground. Meteorological convention (0° = North, 90° = East). |
| `ghi_wm2`      | W/m²   | Global Horizontal Irradiance (shortwave solar radiation on a flat horizontal surface). |
| `dni_wm2`      | W/m²   | Direct Normal Irradiance (solar radiation received per unit area on a surface always perpendicular to the sun’s rays). |
| `dhi_wm2`      | W/m²   | Diffuse Horizontal Irradiance (solar radiation received from the sky after scattering, on a horizontal surface). |

## Notes

- All radiation values are **instantaneous hourly averages** in watts per square meter.  
- Missing values (NaN) can occur during nighttime or when model variables are unavailable.  
- Data source: ERA5 via Open-Meteo Archive API (processed into CSV by `fetch_amsterdam_hourly.py`).  

