# 📁 data/

This directory serves as the "shared data hub" across the entire ML SST Downscaling pipeline. It stores downloaded, intermediate, and processed data files that flow from raw ERA5 downloads to final ".p", i.e., pickle files used by the ML model.

## 📂 Structure

├── access/ # Raw access-s2 NetCDF files for SST and MLD
├── era5/ # Raw ERA5 NetCDF files downloaded via CDS API
	├── raw/ # hourly NetCDF files
	├── daily/ # Daily-averaged NetCDF files (from hourly data)
└── processed/ # Interpolated and formatted .p files (used for ML training)

## 📥 How Data Flows

1. "download_data/era5_downloader.py"  
   Downloads hourly ERA5 data into:  
   → "data/era5/raw/"

2. "download_data/era5_hourly_to_daily.py"  
   Converts hourly to daily NetCDF:  
   → "data/era5/daily/"

3. "interpolation-engine/utils/data_generator.py"  
   Interpolates all variables and generates ".p" files (e.g. "Data2021_gcm.p"):  
   → "data/processed/"

4. "rcnn_model/scripts/data_utils.py"  
   Loads ".p" files from "data/processed/" for ML model training and inference.

## 📌 Notes

- All directories are **created automatically** at runtime if they don’t exist.
- Filenames follow the convention:  
  - Raw NetCDF: "era5_<variable>_hourly_<year>.nc"  
  - Processed pickle: "Data<year>_gcm.p"
- Make sure this folder is accessible to all modules when running the pipeline.

## ✅ Example

After running the full pipeline for 2021:

data/
├── era5/raw/
│ └── era5_sst_hourly_2021.nc
├── era5/daily/
│ └── era5_sst_daily_2021.nc
└── processed/
└── Data2021_gcm.p
