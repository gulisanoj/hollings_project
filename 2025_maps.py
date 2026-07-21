import os
import glob
import re
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.transform import Affine
import rioxarray
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# =========================================================================
# 1. PATHS & INITIAL DATA PREPARATION
# =========================================================================
shp_path = "/Projects/flood_threat/HUC1028/buffers_with_precip.shp"
output_2025_csv = "/Projects/HUC1028/2025_lsrs.csv"

# Raster Directory Paths
raster_dir = "/Projects/flood_threat/HUC1028"
qpe_dir = os.path.join(raster_dir, "QPE")
output_dir = os.path.join(raster_dir, "predictions")
os.makedirs(output_dir, exist_ok=True)

print("Loading shapefile...")
gdf = gpd.read_file(shp_path, engine="fiona")

# Feature engineering matching updated pipeline configurations
gdf["precip_acceleration"] = gdf["P_HR_0"] - gdf["P_HR_MIN1"]

print("Extracting and combining 2025 test datasets...")
is_standard_2025 = gdf['VALID'].astype(str).str.strip().str.startswith('2025')
gdf_standard_2025 = gdf[is_standard_2025].copy()

valid_str = gdf['VALID'].astype(str).str.strip()
is_valid_null = gdf['VALID'].isna() | (valid_str == "") | (valid_str == "None") | (valid_str == "nan")
ends_with_25 = gdf['RND_TIME'].astype(str).str.contains(r'25\s*$', regex=True)
gdf_null_2025 = gdf[is_valid_null & ends_with_25].copy()

n_rows_to_add = min(100, len(gdf_null_2025))
gdf_null_2025_sample = gdf_null_2025.head(n_rows_to_add)

gdf_2025 = pd.concat([gdf_standard_2025, gdf_null_2025_sample], axis=0)
gdf_historical = gdf.drop(gdf_2025.index).copy()

# UPDATED FEATURE COLUMNS: Explicitly including P_3HR_CUM and new terrain naming scheme
feature_cols = [
    "P_HR_0",
    "P_HR_MIN1",
    "P_HR_MIN2",
    "P_3HR_CUM",
    "P_24HR_TOT",
    "P_48HR_TOT",
    "med_HAND",
    "mea_imper",
    "med_tcc",
    "max_twi",
    "min_slope",
    "precip_acceleration"
]

# Ensure the target map setup cleans FLD_SV_CLS strings into standard IMPACTS tags
gdf_historical["fld_sv_cls_clean"] = gdf_historical["FLD_SV_CLS"].astype(str).str.strip().str.upper()
conditions = [
    (gdf_historical["fld_sv_cls_clean"] == "SEVERE") | (gdf_historical["fld_sv_cls_clean"] == "MODERATE"),
    (gdf_historical["fld_sv_cls_clean"] == "NUISANCE"),
    (gdf_historical["fld_sv_cls_clean"] == "NO_IMPACTS") | (gdf_historical["fld_sv_cls_clean"] == "") | (gdf_historical["fld_sv_cls_clean"] == "NONE") | (gdf_historical["FLD_SV_CLS"].isna())
]
choices = ["CONSIDERABLE", "MINIMAL", "NO_IMPACTS"]
gdf_historical["IMPACTS"] = np.select(conditions, choices, default="NO_IMPACTS")

df_hist_clean = gdf_historical.dropna(subset=feature_cols + ["IMPACTS"]).copy()

# =========================================================================
# 2. MODEL TRAINING
# =========================================================================
def run_rf_and_evaluate_2025(X_hist, y_hist, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X_hist, y_hist, test_size=0.2, random_state=42, stratify=y_hist)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True, class_weight="balanced")
    rf.fit(X_train, y_train)
    print(f">>> Trained {model_name} (OOB Score: {rf.oob_score_:.2%})")
    return rf

# Model 1 target map alignment
m1_map = {"CONSIDERABLE": "ANY_IMPACT", "MINIMAL": "ANY_IMPACT", "NO_IMPACTS": "NO_IMPACTS"}
rf_model1 = run_rf_and_evaluate_2025(df_hist_clean[feature_cols], df_hist_clean["IMPACTS"].map(m1_map), "Model 1")

# Model 2 target map alignment (Drops MINIMAL/Nuisance completely)
df_hist_m2 = df_hist_clean[df_hist_clean["IMPACTS"] != "MINIMAL"].copy()
m2_map = {"CONSIDERABLE": "MOD_OR_SEV_IMPACT", "NO_IMPACTS": "NO_IMPACTS"}
rf_model2 = run_rf_and_evaluate_2025(df_hist_m2[feature_cols], df_hist_m2["IMPACTS"].map(m2_map), "Model 2")

# =========================================================================
# 3. REGISTER FILE MAPS BY ACCUMULATION TYPE & FIX PROJECTION
# =========================================================================
print("\nMapping individual QPE subproduct files...")
all_files = glob.glob(os.path.join(qpe_dir, "*.grib2.gz"))

prod_maps = {"01H": {}, "03H": {}, "24H": {}, "48H": {}}

for f in all_files:
    fname = os.path.basename(f)
    ts_match = re.search(r'\d{8}-\d{6}', fname)
    if not ts_match:
        continue
    timestamp = ts_match.group(0)
    
    if "_QPE_48H_" in fname:
        prod_maps["48H"][timestamp] = f
    elif "_QPE_24H_" in fname:
        prod_maps["24H"][timestamp] = f
    elif "_QPE_03H_" in fname:
        prod_maps["03H"][timestamp] = f
    elif "_QPE_01H_" in fname:
        prod_maps["01H"][timestamp] = f

all_timestamps = sorted(list(prod_maps["01H"].keys()))
print(f"Registered {len(all_timestamps)} baseline hourly sequence timestamps.")

if len(all_timestamps) == 0:
    raise FileNotFoundError("Could not find any standard MRMS_MultiSensor_QPE_01H files in the directory.")

# Load the base grid from the first compressed GRIB2 file
first_qpe_path = f"/vsigzip/{prod_maps['01H'][all_timestamps[0]]}"
base_grid = rioxarray.open_rasterio(first_qpe_path, engine="rasterio").squeeze()

# CRITICAL PROJECTION FIXES FOR MRMS GRIB2 METADATA:
base_grid.rio.write_crs("EPSG:4326", inplace=True)

t = base_grid.rio.transform()
res_x, res_y = t.a, t.e 
corrected_transform = t * Affine.translation(-res_x / 2.0, -res_y / 2.0)
base_grid.rio.write_transform(corrected_transform, inplace=True)

raw_shape = base_grid.shape
n_pixels = raw_shape[0] * raw_shape[1]
mrms_nodata = base_grid.rio.nodata if base_grid.rio.nodata is not None else -9999

# =========================================================================
# 4. LOAD & WARP STATIC TERRAIN LAYERS ONCE IN RAM
# =========================================================================
print("\nLoading and temporary-warping static terrain layers to match MRMS Lat/Lon grid...")
static_files = {
    "med_HAND": os.path.join(raster_dir, "HAND.tif"),
    "mea_imper": os.path.join(raster_dir, "impervious.tif"),
    "med_tcc": os.path.join(raster_dir, "tcc.tif"),
    "max_twi": os.path.join(raster_dir, "twi.tif"),
    "min_slope": os.path.join(raster_dir, "slope.tif"),
}

static_arrays = {}
for name, path in static_files.items():
    print(f"-> Structuring {os.path.basename(path)} into RAM memory as {name}...")
    da = rioxarray.open_rasterio(path).squeeze()
    da_warped = da.rio.reproject_match(base_grid)
    static_arrays[name] = da_warped.values

static_nodata = rioxarray.open_rasterio(static_files["med_HAND"]).rio.nodata
if static_nodata is None:
    static_nodata = -9999

# =========================================================================
# 5. SUBPRODUCT MULTI-FILE TIME-SERIES LOOP (MAY TO JULY ONLY)
# =========================================================================
def load_native_qpe(file_path):
    vsi_path = f"/vsigzip/{file_path}"
    with rasterio.open(vsi_path) as src:
        return src.read(1)

cache_01h = {}
cache_03h = {}
cache_24h = {}
cache_48h = {}

target_intervals = ('000000', '030000', '060000', '090000', '120000', '150000', '180000', '210000')
allowed_months = ('05', '06', '07')  # May, June, July

color_table = {
    0: (0, 0, 0, 0),        # 0 = No Impact -> Transparent
    1: (255, 0, 0, 255)     # 1 = Impact -> Red
}

print(f"\nStarting continuous directory processing loop across all {len(all_timestamps)} files...")
print(f"-> Filtering active operations to strictly include months: May, June, July")

for i in range(5, len(all_timestamps)):
    current_ts = all_timestamps[i]
    
    # Filter 1: Only produce maps for explicit 3-hour intervals 
    if not current_ts.endswith(target_intervals):
        continue
        
    # Filter 2: Extract month from timestamp sequence (Format: YYYYMMDD-HHMMSS -> index 4:6 is MM)
    ts_month = current_ts[4:6]
    if ts_month not in allowed_months:
        continue
        
    ts_now = current_ts                         
    ts_min1 = all_timestamps[i-1]               
    ts_min2 = all_timestamps[i-2]               
    ts_lag_3h = all_timestamps[i-3]             
    
    file_01h_0 = prod_maps["01H"].get(ts_now)
    file_01h_1 = prod_maps["01H"].get(ts_min1)
    file_01h_2 = prod_maps["01H"].get(ts_min2)
    file_03h   = prod_maps["03H"].get(ts_now)
    file_24h   = prod_maps["24H"].get(ts_lag_3h) 
    file_48h   = prod_maps["48H"].get(ts_lag_3h) 
    
    if not all([file_01h_0, file_01h_1, file_01h_2, file_03h, file_24h, file_48h]):
        print(f"[-] SKIPPING interval {ts_now}: Missing paired product step configuration files.")
        continue
        
    print(f"\n[+] Processing 3-Hour Interval: {ts_now} (Month: {ts_month})")
    
    try:
        if ts_now not in cache_01h: cache_01h[ts_now] = load_native_qpe(file_01h_0)
        if ts_min1 not in cache_01h: cache_01h[ts_min1] = load_native_qpe(file_01h_1)
        if ts_min2 not in cache_01h: cache_01h[ts_min2] = load_native_qpe(file_01h_2)
        
        if ts_now not in cache_03h: cache_03h[ts_now] = load_native_qpe(file_03h)
        if ts_lag_3h not in cache_24h: cache_24h[ts_lag_3h] = load_native_qpe(file_24h)
        if ts_lag_3h not in cache_48h: cache_48h[ts_lag_3h] = load_native_qpe(file_48h)
        
        feature_arrays = static_arrays.copy()
        
        feature_arrays["P_HR_0"] = cache_01h[ts_now]
        feature_arrays["P_HR_MIN1"] = cache_01h[ts_min1]
        feature_arrays["P_HR_MIN2"] = cache_01h[ts_min2]
        feature_arrays["P_3HR_CUM"] = cache_03h[ts_now]
        feature_arrays["P_24HR_TOT"] = cache_24h[ts_lag_3h] 
        feature_arrays["P_48HR_TOT"] = cache_48h[ts_lag_3h] 
        
        feature_arrays["precip_acceleration"] = feature_arrays["P_HR_0"] - feature_arrays["P_HR_MIN1"]
        
        spatial_matrix = np.zeros((n_pixels, len(feature_cols)))
        for idx, col in enumerate(feature_cols):
            spatial_matrix[:, idx] = feature_arrays[col].ravel()
            
        valid_pixel_mask = (feature_arrays["med_HAND"].ravel() != static_nodata) & \
                           (feature_arrays["P_HR_0"].ravel() != mrms_nodata) & \
                           (~np.isnan(spatial_matrix).any(axis=1))
        
        # --- Generate Raster Predictions ---
        models_to_run = {
            "Model1_Any_vs_None": (rf_model1, ["NO_IMPACTS", "ANY_IMPACT"]),
            "Model2_Considerable_vs_None": (rf_model2, ["NO_IMPACTS", "MOD_OR_SEV_IMPACT"])
        }
        
        for out_name, (model, class_labels) in models_to_run.items():
            prediction_array = np.zeros((n_pixels,), dtype=np.uint8)
            
            if len(valid_pixel_mask) > 0:
                valid_predictions_str = model.predict(spatial_matrix[valid_pixel_mask])
                label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
                numeric_predictions = np.vectorize(label_to_idx.get)(valid_predictions_str)
                prediction_array[valid_pixel_mask] = numeric_predictions
            
            prediction_grid = prediction_array.reshape(raw_shape)
            
            output_tif_path = os.path.join(output_dir, f"{ts_now}_{out_name}.tif")
            profile = {
                'driver': 'GTiff',
                'height': raw_shape[0],
                'width': raw_shape[1],
                'count': 1,
                'dtype': 'uint8',
                'crs': base_grid.rio.crs,
                'transform': base_grid.rio.transform(),
                'nodata': 0  
            }
            
            with rasterio.open(output_tif_path, 'w', **profile) as dst:
                dst.write(prediction_grid, 1)
                dst.write_colormap(1, color_table)
            
        print(f"-> Maps written cleanly: {ts_now}")
        
        # Clean cache records older than 5 timeline indexes back to prevent RAM exhaustion
        old_keys = [k for k in list(cache_01h.keys()) if k < all_timestamps[i-5]]
        for k in old_keys: 
            del cache_01h[k]
        del cache_03h[ts_now], cache_24h[ts_lag_3h], cache_48h[ts_lag_3h]
        
    except Exception as e:
        print(f"Skipping timestamp {ts_now} due to unexpected processing error: {e}")

print("\nTime-series script sequence finished processing completely!")