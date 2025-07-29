# 📁 download_data/

This module handles the **automated downloading and preprocessing of ACCESS-S2 and ERA5 climate data** (e.g. SST, latent heat flux, radiation variables).

It consists of two key Scripts:
1. Downloading hourly ERA5 NetCDF files
2. Converting them to daily-averaged files, which are later used for interpolation

---

## 🧩 Scripts

| Script | Description |
|--------|-------------|
| `era5_downloader.py` | Downloads hourly ERA5 single-level climate variables (NetCDF) for specified years and stores them in "data/era5/raw/". |
| `era5_hourly_to_daily.py` | Converts hourly NetCDF files to daily averages and saves to "data/era5/daily/". |

---

## 📦 Dependencies and requirements

Required:
cdsapi
xarray
numpy

You will also need a CDS API key to use the downloader (see below).

## 🔐 Setup: CDS API Key:
To authenticate with the Copernicus API:

Register at https://cds.climate.copernicus.eu/

Get your API key from your user profile

### Save it in a file at:
	~/.cdsapirc (Linux/macOS)
	or
	C:\\Users\\<you>\\.cdsapirc (Windows)

### Example .cdsapirc:
	url: https://cds.climate.copernicus.eu/api/v2
	key: <uid>:<api_key>

---

## 🛠️ How to Use

### 📥 1. Download Data

python era5_downloader.py

You can modify the variables, years, and area inside the script.

Output:
→ data/era5/raw/era5_<variable>_hourly_<year>.nc

### 📆 2. Convert to Daily Averages

python era5_hourly_to_daily.py

You can change the frequency of average as per your requirement. But then you might have to adjust the interpolation routine as well accordingly. 

Output:
→ data/era5/daily/era5_<variable>_daily_<year>.nc

### 🗂️ Output Structure

data/
├── era5/raw/        # Output from era5_downloader.py
└── era5/daily/      # Output from era5_hourly_to_daily.py




