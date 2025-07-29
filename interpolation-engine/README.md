# 📁 interpolation-engine/

This module performs **interpolation** by interpolating global climate data onto a high-resolution regional grid (e.g. ROMS). The output is a ".p" file that contains downscaled variables ready for use in ML model training.

There are three interpolation routines/scripts, each for access-s2, era5 and mix layer depth:
These scripts interpolate the respective datasets. Each scrtipt can also generate global climate model data for a given region and a desired variable if needed for postprocessing. You can choose the interpolation method of your choice as demonstrated in access_interpolator.py (method='random_forest')

It acts as a bridge between:
- "download_data/" (raw NetCDF)
- "rcnn_model/" (ML training on preprocessed `.p` files)

---

## 🧩 Components

| File/Folder | Description |
|-------------|-------------|
| "main.py" | Entrypoint to run interpolation for a given year. |
| "utils/data_generator.py" | Core script that reads ERA5 & ACCESS-s2 data, interpolates, and saves a `Data{year}_gcm.p` file. |
| "interpolation/" | Optional: stores custom interpolation kernels (RF, RBF, RF, etc) for a respective dataset. |

---

## 📥 Inputs

From:
- "data/era5/daily/" — daily ERA5 NetCDF files
- "data/access/" — access-s2 NetCDF files

## Variables interpolated:
- SST
- Salt
- Latent/Sensible Heat Flux
- Shortwave/Longwave Radiation
- MLD (mixed layer depth)

---

## 📤 Output

Creates `.p` file:

---
## How to use:

from utils.data_generator import interpolationroutine
interpolationroutine(2021)

---
## 🛠️ Requirements

Required packages:
numpy
xarray
scikit-learn
matplotlib

## 📎 Example Output (Pickle)
[SST_array, Salt_array, slhf, ssr, str, sshf, mld1]
→ shape: [#days, 640 * 480]
