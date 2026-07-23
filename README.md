# 🌊 Predicting Flash Flood Impacts Through Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NOAA Hollings](https://img.shields.io/badge/Program-NOAA%20Hollings%20Scholarship-005596.svg)](https://www.noaa.gov/office-education/hollings-scholarship)

This repository contains the end-to-end data processing pipelines, machine learning models, local LLM classification modules, and geospatial visualization scripts developed as part of the **NOAA Ernest F. Hollings Undergraduate Scholarship Program**. 

The framework automates terrain processing, classifies text-based storm reports using local LLMs (Ollama/Gemma), extracts dynamic hydro-meteorological precipitation features across multiple NOAA data streams (AORC, MRMS, HRRR, IEM), trains balanced Random Forest models to predict flood threat levels, and renders spatial time-series animations overlaid with real-time NWS warnings.

---

## 🏗️ Repository Workflow Architecture

```text
                               ┌──────────────────────────────────────────────┐
                               │     1. DATA ACQUISITION & TERRAIN SETUP      │
                               │  - Copernicus 30m DEM Tiles (STAC API)        │
                               │  - Watershed Boundaries & Spatial Buffers     │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │       2. LLM REMARK CLASSIFICATION           │
                               │  - Severity: NUISANCE, MODERATE, SEVERE       │
                               │  - Type: FLUVIAL, INFRASTRUCTURE              │
                               │  - Local Inference via Ollama (Gemma 4:e4b)  │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │     3. FEATURE ENGINEERING & MODELING        │
                               │  - NOAA AORC 1km Cloud Zarr Extractions      │
                               │  - NOAA MRMS QPE (1h, 3h, 24h, 48h)          │
                               │  - NOAA HRRR 3h Cumulative Matrices (AWS)     │
                               │  - IEM ASOS Rolling Rainfall Accumulations   │
                               │  - Static Terrain: HAND, TWI, Slope, Imp, TCC│
                               │  - Random Forest Classifiers (Model 1 & 2)   │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │        4. SPATIAL INFERENCE & MAPS           │
                               │  - Spatial Grid Prediction (GeoTIFF)         │
                               │  - Contextual Validation vs LSRs & FFWs      │
                               │  - Animated Time-Series GIF Compilation      │
                               └──────────────────────────────────────────────┘
```

---

## 📂 Directory Structure

```text
hollings_project/
├── classifications/           # LLM text classification & HTML auditing tools
├── data/                      # DEM, QPE, HRRR, IEM, & buffer feature pipelines
├── models/                    # Random Forest training & evaluation scripts
├── visuals/                   # Spatial map prediction engines & GIF generators
└── README.md
```

---

## 📜 Comprehensive Script Catalog

### 📁 `data/` — Data Acquisition & Feature Extraction Pipelines
Scripts responsible for querying elevation data and extracting dynamic hydro-meteorological precipitation features across NOAA repositories.

| Script / File Name | Description |
| :--- | :--- |
| `data/dems.py` | Queries the Microsoft Planetary Computer STAC API for 30m Copernicus DEM tiles covering target watershed boundaries, mosaics them, and reprojects the output to `EPSG:5070`. |
| `data/download_qpe.py` | Downloader targeting NOAA S3 buckets for MRMS CONUS MultiSensor QPE products (1H, 3H, 24H, 48H) using event-based date matching derived from LSR shapefiles. |
| `data/download_qpe_25.py` | Time-series downloader for raw MRMS QPE compressed GRIB2 data across continuous seasonal windows (e.g., April 1 to September 30, 2025). |
| `data/hrrr.py` | Downloads high-resolution HRRR surface forecasts via `Herbie`, reprojects curvilinear grids to `EPSG:5070` using GDAL, and calculates 3-hour cumulative precipitation matrices via `NumPy`. |
| `data/iem_csv.py` | Connects to Iowa Environmental Mesonet (IEM) ASOS station streams across 48 CONUS states, computes 3-hour rolling rainfall totals, and transforms coordinates to `EPSG:5070`. |
| `data/pcp_buffers.py` | Cloud-native extraction pipeline using `xarray` and `s3fs` to query NOAA AORC 1km Zarr datasets (`s3://noaa-nws-aorc-v1-1-1km`). Generates random spatial buffer controls, extracts 1h and 3h core rainfall, and calculates 24h/48h antecedent precipitation. |
| `data/precip_workflow_test.py` | Integrated pipeline unit test. Simulates spatial nearest-neighbor lookups, pulls RTMA/Stage IV data via Herbie, and verifies feature matrices for precipitation accuracy. |

---

### 📁 `classifications/` — Local LLM Classification & Web Verification
Scripts that pipe National Weather Service Local Storm Reports (LSR) remark text through a locally running **Gemma 4:e4b** model via Ollama to derive structured training labels, alongside interactive web auditing dashboards.

| Script / File Name | Description |
| :--- | :--- |
| `classifications/2025_lsrs_classification.py` | Runs 2025 LSR text remarks through LLM severity rules (`NUISANCE`, `MODERATE`, `SEVERE`) with progress-checkpointing to CSV output. |
| `classifications/classification.py` | Core baseline script that evaluates LSR remark text against FEMA/NWS criteria to assign severity classes using deterministic sampling (`temperature=0.0`). |
| `classifications/flood_type_classification.py` | Advanced flood-type classifier featuring Python-level keyword rule overrides for bridges, low-water crossings, and urban infrastructure to prevent LLM hallucination. |
| `classifications/severity_classification_script.py` | Alternative pipeline configuration for batch-processing CSV datasets through Ollama severity prompts. |
| `classifications/shapefile_classification.py` | Extends text classification directly into spatial vector attributes, safely appending `fld_sv_cls` into shapefile DBF tables. |
| `classifications/shapeifle.py` | Spatial shapefile LLM classifier featuring incremental 100-row saving safeguards for 2026 LSR datasets to prevent data loss during long processing runs. |
| `classifications/updated_verification.html` | Interactive Tailwind-based web dashboard for auditing severity class mismatches between AI predictions (`fld_sv_cls`) and human ground truth annotations (`picked_class`). |
| `classifications/verifcation_type.html` | Interactive verification tool for inspecting, filtering, and auditing LLM flood type assignments (`FLUVIAL` vs `INFRASTRUCTURE`). |

---

### 📁 `models/` — Machine Learning Model Training & Evaluation
This core module merges dynamic rainfall features with static terrain metrics and trains supervised Random Forest classifiers.

* **Model 1 (Binary):** `ANY_IMPACT` vs `NO_IMPACTS`
* **Model 2 (Binary Filtered):** `CONSIDERABLE` vs `NO_IMPACTS` (drops Nuisance)

| Script / File Name | Description |
| :--- | :--- |
| `models/305_model.py` | Dedicated modeling script for HUC 0305. Features automated visualization generators for feature importances, confusion matrix heatmaps, and decision sub-tree architectures. |
| `models/classed_models.py` | Model trainer configured with balanced 2025 unseen evaluation benchmarks and automated joblib model payload serialization. |
| `models/huc1029_model.py` | Trains Random Forests using SMOTE resampling to handle extreme target class imbalances within HUC 1029 watersheds. |
| `models/model_correction.py` | Production training script that serializes compiled model instances (`.joblib`) specifically formatted for operational raster inference engines. |
| `models/models.py` | Primary, generalized model pipeline. Combines historical and 2025 test datasets, extracts rainfall acceleration/land-use interactions, trains Random Forests, and evaluates holdout accuracy. |

---

### 📁 `visuals/` — Spatial Map Prediction & Visual Validation
Scripts that ingest live QPE weather layers and static terrain layers to execute spatial predictions cell-by-cell, alongside tools that compile aerial basemaps, prediction rasters, LSR points, and Flash Flood Warning boundaries into animated timelines.

| Script / File Name | Description |
| :--- | :--- |
| `visuals/2025_maps.py` | Operational spatial raster predictor restricted to May–July event timelines within the HUC 1028 watershed. |
| `visuals/april_gifs.py` | Renders Web Mercator time-series map frames overlaid on Esri World Imagery basemaps with active +/-3-hour LSR rolling validation windows for April data. |
| `visuals/april_maps.py` | Time-series spatial prediction engine bound specifically to April 2025 events with 3-hour lag handling logic. |
| `visuals/considerable_gif.py` | Compiles chronological PNG frame assets generated by other scripts into smooth, infinitely looping validation GIFs. |
| `visuals/considerable_pngs.py` | Generates individual high-definition PNG map frames tailored for Model 2 (Considerable Threat) with active +/-3-hour warning and storm report windows. |
| `visuals/final_2025_gifs.py` | Advanced frame renderer supporting dynamic multi-line title banners, variable-width legend bounding boxes, and severity-coded vector dots for 2025 events. |
| `visuals/gifs.py` | Regional context animation compiler covering HUC 1028 with active warning outlines and storm report buffers. |
| `visuals/google_slides_gif.py` | Asset optimization utility that resizes, downsamples, and applies 64-color adaptive palettes to compile lightweight, presentation-ready GIFs for slide decks. |
| `visuals/maps_2025.py` | Master continuous time-series raster predictor. Features critical on-the-fly cell-corner transform fixes for MRMS GRIB2 metadata to align with terrain data. |
| `visuals/pngs.py` | Base generator for individual high-resolution PNG frame assets for time-series predictions with custom spatial legends and watershed boundary outlines. |

---

## 🛠️ Installation & Dependencies

### 1. Clone the Repository
```bash
git clone [https://github.com/gulisanoj/hollings_project.git](https://github.com/gulisanoj/hollings_project.git)
cd hollings_project
```

### 2. Environment Setup
It is highly recommended to use Conda due to complex GIS C-library dependencies (GDAL, GEOS, PROJ):
```bash
conda create -n flood_ml python=3.10 -y
conda activate flood_ml
conda install -c conda-forge gdal geopandas rasterio rioxarray xarray pystac-client stackstac contextily s3fs zarr
pip install imbalanced-learn ollama herbie-data pillow seaborn
```

### 3. Ollama Setup (For Text Classification Scripts)
To run the LLM classification pipeline locally:
```bash
# Install Ollama ([https://ollama.com](https://ollama.com)) and pull the Gemma model
ollama pull gemma4:e4b
```

---

## 🤝 Citation & Acknowledgments

* **NOAA Ernest F. Hollings Undergraduate Scholarship Program** for project support and research funding.
* **National Weather Service (NWS) & FEMA** for storm report records and warning datasets.
* **Iowa Environmental Mesonet (IEM)** and **NOAA Big Data Program (AWS)** for public hydro-meteorological data access (AORC, MRMS, HRRR).

If you use or reference this work, please cite:
```bibtex
@misc{gulisano2026flashflood,
  author = {Gulisano, J.},
  title = {Predicting Flash Flood Impacts Through Machine Learning},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{[https://github.com/gulisanoj/hollings_project](https://github.com/gulisanoj/hollings_project)}}
}
```
