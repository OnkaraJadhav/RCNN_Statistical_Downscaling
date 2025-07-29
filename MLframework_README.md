# 🌊 ML SST Downscaling Framework

A repository for machine learning pipeline to **downscale Sea Surface Temperature (SST)** data using a hybrid interpolation + deep learning approach.

### Abstract:
The large-scale oceanic and atmospheric forecasts provided by global climate models typically lack sufficient resolution to accurately capture the response of the coastal ocean. Dynamical downscaling is computationally prohibitive, especially when applied to extensive coastlines, to many predictive ensembles, and to longer time periods. Therefore, this work presents a statistical downscaling of sea surface temperature (SST) from the seasonal coupled ocean-atmosphere forecast system (ACCESS-S2) based on machine learning techniques.

This study proposes a novel two-stage deep learning framework that combines a U-Net convolutional neural network for SST prediction with a Residual Corrective Neural Network (RCNN) for iterative refinement toward high-resolution outputs. The target SST fields are derived from the Regional Ocean Modeling System (ROMS).
The RCNN progressively refines its predictions by incorporating dynamically scaled residuals at each step, enabling accurate capture of both broad patterns and fine-grained features such as eddies and fronts. To improve performance during extreme events, which may be absent in the training data, a custom loss-assisted RCNN variant is also introduced.

The developed framework efficiently downscales the SST along the west coast of Australia. A specific case study involving the 2011 marine heat wave event shows that the RCNN improves the predictions of ACCESS-S2 SST by increasing horizontal resolution from 25 km to 2 km, allowing the identification of finer-scale anomalies during marine heat waves that are not resolved in the ACCESS-S2 dataset. By applying the developed framework, this work achieves a balance between computational efficiency and the accuracy required for capturing local oceanic variations, ultimately improving forecasting capabilities for coastal management and marine ecosystem studies.

Keywords: Statistical Downscaling, Sea Surface Temperature (SST), Deep Learning, Residual Corrective Neural Network (RCNN), ACCESS-S2, ROMS (Regional Ocean Modeling System), Marine Heatwave, Climate Modeling


This project integrates:
- ✅ data download & preprocessing (`download_data/`)
- ✅ Statistical interpolation to ROMS-like grid (`interpolation-engine/`)
- ✅ RCNN-based fine-scale SST prediction using U-Net (`rcnn_model/`)

---

## 📁 Repository Structure

```
MLframework/
├── download_data/         # Step 1: Download & process raw data
│   ├── era5_downloader.py
│   ├── era5_hourly_to_daily.py
│   └── README.md
│
├── interpolation-engine/  # Step 2: Interpolate ERA5 onto ROMS-like grid
│   ├── utils/data_generator.py
│   ├── interpolation/
│   └── main.py
│
├── rcnn_model/            # Step 3: Train & evaluate rcnn
│   ├── scripts/train.py
│   ├── scripts/test_full_inference.py
│   ├── scripts/data_utils.py
│   ├── scripts/model.py
│   └── scripts/config.py
│
├── data/                  # Shared data storage (auto-created)
│   ├── raw/               # Raw NetCDF downloads
│   ├── daily/             # Optional: daily-averaged NetCDF
│   └── processed/         # Final .p files used by model
│
└── README.md              # Top-level project overview
```

---

## 🔁 Full Pipeline Overview

| Step | Folder | Script | Output |
|------|--------|--------|--------|
| 1️⃣ Download ERA5 | [`download_data/`](download_data/README.md) | `era5_downloader.py` | `data/raw/*.nc` |
| 2️⃣ Convert to Daily | [`download_data/`](download_data/README.md) | `era5_hourly_to_daily.py` | `data/daily/*.nc` |
| 3️⃣ Interpolate & Format | [`interpolation-engine/`](interpolation-engine/README.md) | `main.py` or `utils/data_generator.py` | `data/processed/Data{year}_gcm.p` |
| 4️⃣ Train Model | [`rcnn_model/scripts/`](rcnn_model/README.md) | `train.py` | `output/unet_weights.h5` |
| 5️⃣ Run Inference | [`rcnn_model/scripts/`](rcnn_model/README.md) | `test_full_inference.py` | `output/test_full_results.pkl` |

---

## 🚀 Quickstart
---

### ✅ 1. Configure CDS API Key

See 📄 [`download_data/README.md`](download_data/README.md#🔐-setup-cds-api-key)

---

### ✅ 2. Run the Full Pipeline (Example)

```
# Step 1: Download data for year 2021
python download_data/era5_downloader.py

# Step 2: Convert hourly NetCDFs to daily
python download_data/era5_hourly_to_daily.py

# Step 3: Interpolate global data to ROMS-like grid
python interpolation-engine/main.py

# Step 4: Train U-Net RCNN model
python rcnn_model/scripts/train.py

# Step 5: Evaluate full-image inference
python rcnn_model/scripts/test_full_inference.py
```

---

## 📂 Shared Data Directory

```
data/
├── access/         ← NetCDF from access-s2
├── era5/raw/         ← NetCDF from downloader
├── era5/daily/       ← Daily-averaged ERA5
└── processed/   ← Final .p files used for ML training
```

These folders are created automatically when needed.

---

## 📈 Features

- U-Net with mask-aware regression
- Full-image inference pipeline
- Multi-year training support
- Compatible with cloud storage & batch jobs
- Auto-handled directory creation

---

## 👥 Authors

- Onkar Jadhav, Tim French, Ivica Janekovic, Nicole Jones, Matt Rayson

---

## 🧭 Navigation Links

- [📁 download_data](download_data/README.md)
- [📁 interpolation-engine](interpolation-engine/README.md)
- [📁 rcnn_model](rcnn_model/README.md)
- [📁 data](data/README.md)
