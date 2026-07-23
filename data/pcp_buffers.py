import os
import random
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import s3fs
from shapely.geometry import Point

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
BASE_DIR = "/Projects/flood_threat/MERIT/HUC1029"
SHP_PATH = os.path.join(BASE_DIR, "buffers.shp")
CHEESE_PATH = os.path.join(BASE_DIR, "swiss_cheese.shp")

# Unified multi-state precipitation reference file
COMBINED_PRECIP_PATH = "/Projects/flood_threat/MERIT/HUC1029/combined_states_3hr_precipitation.csv"

OUTPUT_PATH = os.path.join(BASE_DIR, "final_combined_buffers_with_precip_final.shp")
TARGET_CRS = "EPSG:5070"
NUM_RANDOM_SAMPLES = 2700

AORC_S3_BUCKET = "s3://noaa-nws-aorc-v1-1-1km"
fs = s3fs.S3FileSystem(anon=True)
AORC_YEAR_CACHE = {}

# ==============================================================================
# CLOUD OPTIMIZED AORC ZARR EXTRACTOR
# ==============================================================================
def get_aorc_dataset_for_year(year):
    """Lazily loads and caches the annual AORC Zarr dataset from Amazon S3."""
    if year in AORC_YEAR_CACHE:
        return AORC_YEAR_CACHE[year]
    
    s3_path = f"{AORC_S3_BUCKET}/{year}.zarr"
    print(f"[AORC CONNECT] Initializing connection to cloud cube: {s3_path}...")
    
    try:
        store = s3fs.S3Map(root=s3_path, s3=fs, check=False)
        ds = xr.open_zarr(store, consolidated=False)
        AORC_YEAR_CACHE[year] = ds
        return ds
    except Exception as e:
        print(f"[ERROR] Failed to open AORC Zarr cube for year {year}: {e}")
        return None

def extract_aorc_precip(target_time, poly_x, poly_y, is_projected=False):
    """Extracts hourly precipitation directly from S3 Zarr cubes."""
    ds = get_aorc_dataset_for_year(target_time.year)
    if ds is None:
        return np.nan
    
    try:
        if is_projected:
            point_df = gpd.GeoSeries([Point(poly_x, poly_y)], crs=TARGET_CRS).to_crs("EPSG:4326")
            native_lon = point_df.geometry.iloc[0].x
            native_lat = point_df.geometry.iloc[0].y
        else:
            native_lon = float(poly_x)
            native_lat = float(poly_y)
        
        target_timestamp = pd.Timestamp(target_time).floor('h')
        val = ds['APCP_surface'].sel(
            time=target_timestamp, longitude=native_lon, latitude=native_lat, method='nearest'
        ).values.item()
        
        return float(val)
    except Exception:
        return np.nan

def extract_aorc_accumulation_block(end_history_time, duration_hours, poly_x, poly_y, is_projected=False):
    """Sums continuous hours backward from an anchor time using rapid array slicing."""
    try:
        if is_projected:
            point_df = gpd.GeoSeries([Point(poly_x, poly_y)], crs=TARGET_CRS).to_crs("EPSG:4326")
            native_lon = point_df.geometry.iloc[0].x
            native_lat = point_df.geometry.iloc[0].y
        else:
            native_lon = float(poly_x)
            native_lat = float(poly_y)
        
        end_stamp = pd.Timestamp(end_history_time).floor('h')
        start_stamp = end_stamp - pd.Timedelta(hours=duration_hours - 1)
        time_range = pd.date_range(start=start_stamp, end=end_stamp, freq='h')
        
        total_precip = 0.0
        has_data = False
        
        for year, group in time_range.groupby(time_range.year).items():
            ds = get_aorc_dataset_for_year(year)
            if ds is None:
                continue
                
            year_times = [pd.Timestamp(t) for t in group]
            slice_vals = ds['APCP_surface'].sel(
                time=year_times, longitude=native_lon, latitude=native_lat, method='nearest'
            ).values
            
            valid_vals = slice_vals[~np.isnan(slice_vals)]
            if len(valid_vals) > 0:
                total_precip += sum(valid_vals)
                has_data = True
                
        return total_precip if has_data else np.nan
    except Exception:
        return np.nan

# ==============================================================================
# SPATIAL SAMPLING ENGINE FOR NEW RANDOM BUFFERS
# ==============================================================================
def generate_inset_sample_buffers(cheese_path, total_needed):
    """Creates 500m buffers around random points generated 1000m away from cheese boundaries."""
    print("🎨 Processing geometry for spatial sampling constraints...")
    cheese_gdf = gpd.read_file(cheese_path).to_crs(TARGET_CRS)
    unified_cheese = cheese_gdf.unary_union
    
    # Negative buffer strips away edges and shrinks from inner holes
    safe_zone = unified_cheese.buffer(-1000)
    
    if safe_zone.is_empty:
        raise ValueError("❌ Error: Swiss cheese geometry completely erased when inset by 1000m!")
        
    bounds = safe_zone.bounds
    generated_points = []
    
    print(f"🎲 Generating {total_needed} random control coordinates...")
    attempts = 0
    while len(generated_points) < total_needed:
        attempts += 1
        rand_x = random.uniform(bounds[0], bounds[2])
        rand_y = random.uniform(bounds[1], bounds[3])
        candidate_pt = Point(rand_x, rand_y)
        
        if safe_zone.contains(candidate_pt):
            generated_points.append(candidate_pt)
            
        if attempts > total_needed * 100 and len(generated_points) == 0:
            raise RuntimeError("❌ Loop trapped. Check geometry coordinates framework configurations.")

    points_gdf = gpd.GeoDataFrame(geometry=generated_points, crs=TARGET_CRS)
    points_gdf['LON'] = points_gdf.geometry.x
    points_gdf['LAT'] = points_gdf.geometry.y
    points_gdf['IS_NEW'] = 1  # Track these as newly generated records
    
    buffers_gdf = points_gdf.copy()
    buffers_gdf.geometry = buffers_gdf.geometry.buffer(500)
    
    return buffers_gdf

# ==============================================================================
# MAIN PROCESSING WORKFLOW
# ==============================================================================
def main():
    print("Verifying base operational workspace...")
    print(f"  -> Existing Buffers SHP: {SHP_PATH} [{'FOUND' if os.path.exists(SHP_PATH) else 'MISSING'}]")
    print(f"  -> Swiss Cheese SHP:    {CHEESE_PATH} [{'FOUND' if os.path.exists(CHEESE_PATH) else 'MISSING'}]")
    print(f"  -> Precipitation CSV:   {COMBINED_PRECIP_PATH} [{'FOUND' if os.path.exists(COMBINED_PRECIP_PATH) else 'MISSING'}]")

    if not all(os.path.exists(p) for p in [SHP_PATH, CHEESE_PATH, COMBINED_PRECIP_PATH]):
        raise FileNotFoundError("❌ Error: Required components missing from paths.")

    # 1. LOAD AND PREPARE EXISTING BUFFERS
    print("📂 Processing existing buffers shapefile layer...")
    existing_gdf = gpd.read_file(SHP_PATH).to_crs(TARGET_CRS)
    existing_gdf['LON'] = existing_gdf.geometry.centroid.x
    existing_gdf['LAT'] = existing_gdf.geometry.centroid.y
    existing_gdf['IS_NEW'] = 0

    # Clean column structures
    for col in ['VALID', 'valid']:
        if col in existing_gdf.columns:
            existing_gdf = existing_gdf.rename(columns={col: 'VALID'})
            
    if 'VALID' in existing_gdf.columns:
        existing_gdf['VALID'] = pd.to_datetime(existing_gdf['VALID'], errors='coerce')
        existing_gdf['RND_TIME'] = existing_gdf['VALID'].dt.round('H')
    else:
        raise KeyError("❌ Error: Could not find a 'VALID' column inside the existing buffers.shp attributes table.")

    # 2. GENERATE NEW RANDOM BUFFERS
    new_gdf = generate_inset_sample_buffers(CHEESE_PATH, NUM_RANDOM_SAMPLES)
    new_gdf['VALID'] = pd.NaT
    new_gdf['RND_TIME'] = pd.NaT

    # 3. MERGE BOTH DATASETS INTO ONE PIPELINE MATRIX
    print("🔀 Concatenating existing and new buffer layers into single execution frame...")
    combined_gdf = pd.concat([existing_gdf, new_gdf], ignore_index=True)

    # Clean out any 2026 data across the merged block to prevent bottlenecks
    combined_gdf = combined_gdf[combined_gdf['RND_TIME'].dt.year != 2026].copy()

    # Pre-populate empty data extraction target column schemas
    new_cols = ['P_HR_0', 'P_HR_MIN1', 'P_HR_MIN2', 'P_3HR_CUM', 'P_24HR_TOT', 'P_48HR_TOT']
    for col in new_cols:
        combined_gdf[col] = np.nan
    combined_gdf['SRC_STN'] = ""
    combined_gdf['SRC_TIME'] = ""

    # Load master station registry -- completely unfiltered
    print("📊 Loading ASOS master registry records...")
    df_precip = pd.read_csv(COMBINED_PRECIP_PATH)
    df_precip.columns = df_precip.columns.str.upper()
    print(f"   ↳ Ingested full precipitation log database: {len(df_precip)} records available.")
    
    if 'VALID' in df_precip.columns:
        df_precip['VALID_TIME'] = pd.to_datetime(df_precip['VALID'], errors='coerce')
    else:
        df_precip['VALID_TIME'] = pd.to_datetime(df_precip['WINDOW_END_UTC'], errors='coerce')
        
    df_precip['ROUNDED_TIME'] = df_precip['VALID_TIME'].dt.round('H')
    df_precip = df_precip[df_precip['ROUNDED_TIME'].dt.year != 2026].copy()

    # 4. EXECUTE CLOUD SAMPLING STRATEGY
    print("\nBeginning high-speed cloud-native Zarr extraction...")
    total_records = len(combined_gdf)
    used_csv_indices = set()
    
    for idx, row in combined_gdf.iterrows():
        poly_x = row['LON']
        poly_y = row['LAT']
        is_projected = True if abs(poly_x) > 180 else False
        
        # Determine specific pathway target timeline
        if row['IS_NEW'] == 0:
            # Pathway A: Existing buffer -- STRICTLY use its own pre-calculated rounded timestamp
            target_time = row['RND_TIME']
            if pd.isna(target_time):
                continue
            combined_gdf.at[idx, 'SRC_STN'] = "EXISTING_SHP"
            combined_gdf.at[idx, 'SRC_TIME'] = target_time.strftime('%Y-%m-%d %H:%M')
            print(f"📦 [Record {idx + 1}/{total_records}] Existing Buffer | Time: {target_time.strftime('%Y-%m-%d %H:%M')}")
            
            # Extract once for existing buffers without fallback requirements
            h0 = extract_aorc_precip(target_time, poly_x, poly_y, is_projected)
            h1 = extract_aorc_precip(target_time - pd.Timedelta(hours=1), poly_x, poly_y, is_projected)
            h2 = extract_aorc_precip(target_time - pd.Timedelta(hours=2), poly_x, poly_y, is_projected)
            cum_3hr = sum([x for x in [h0, h1, h2] if not np.isnan(x)]) if not all(np.isnan([h0, h1, h2])) else np.nan
        
        else:
            # Pathway B: Newly created buffer -- Sample based on distance across full database array
            valid_extraction_found = False
            ignored_local_indices = set()
            
            while not valid_extraction_found:
                best_csv_idx = None
                min_dist = float('inf')
                
                for c_idx, c_row in df_precip.iterrows():
                    if c_idx in used_csv_indices or c_idx in ignored_local_indices:
                        continue
                    
                    dist = np.sqrt((poly_x - c_row['X_5070'])**2 + (poly_y - c_row['Y_5070'])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_csv_idx = c_idx
                
                if best_csv_idx is None:
                    print(f"⚠️ [Record {idx + 1}/{total_records}] New Buffer |迅速 Exhausted all nearby station timelines.")
                    break
                
                target_time = df_precip.loc[best_csv_idx, 'ROUNDED_TIME']
                stn_id = str(df_precip.loc[best_csv_idx, 'STATION'])
                
                # Extract directly from Zarr cube
                h0 = extract_aorc_precip(target_time, poly_x, poly_y, is_projected)
                h1 = extract_aorc_precip(target_time - pd.Timedelta(hours=1), poly_x, poly_y, is_projected)
                h2 = extract_aorc_precip(target_time - pd.Timedelta(hours=2), poly_x, poly_y, is_projected)
                
                hourly_slices = [h0, h1, h2]
                if all(np.isnan(x) for x in hourly_slices):
                    cum_3hr = np.nan
                else:
                    cum_3hr = sum([x for x in hourly_slices if not np.isnan(x)])
                
                # Lock in whatever the Zarr value is, even if it is 0.0 or NaN
                used_csv_indices.add(best_csv_idx)
                combined_gdf.at[idx, 'RND_TIME'] = target_time
                combined_gdf.at[idx, 'SRC_STN'] = stn_id
                combined_gdf.at[idx, 'SRC_TIME'] = target_time.strftime('%Y-%m-%d %H:%M')
                valid_extraction_found = True
                
                print(
                    f"🎲 [Record {idx + 1}/{total_records}] New Buffer | Assigned Station: {stn_id} "
                    f"| Extracted Zarr 3HR: {cum_3hr if np.isnan(cum_3hr) else round(cum_3hr, 2)}mm | Dist: {round(min_dist, 1)}m"
                )
            
            if not valid_extraction_found:
                continue

        # Commit results to the target array
        combined_gdf.at[idx, 'P_HR_0'] = h0
        combined_gdf.at[idx, 'P_HR_MIN1'] = h1
        combined_gdf.at[idx, 'P_HR_MIN2'] = h2
        combined_gdf.at[idx, 'P_3HR_CUM'] = cum_3hr
        
        # ==========================================
        # STEP 2: ANTECEDENT ACCUMULATIONS (100% ZARR)
        # ==========================================
        end_of_history = target_time - pd.Timedelta(hours=3)
        
        p_24 = extract_aorc_accumulation_block(end_of_history, 24, poly_x, poly_y, is_projected)
        combined_gdf.at[idx, 'P_24HR_TOT'] = p_24
        
        p_48 = extract_aorc_accumulation_block(end_of_history, 48, poly_x, poly_y, is_projected)
        combined_gdf.at[idx, 'P_48HR_TOT'] = p_48

        # Terminal Output Logging
        print(f"    ↳ Precipitation Slices -> Hourly Core: [{h2}mm, {h1}mm, {h0}mm]")
        print(f"    ↳ Accumulations         -> 3HR Cum: {cum_3hr}mm | 24HR Antecedent: {p_24}mm | 48HR Antecedent: {p_48}mm\n")

    print(f"\n💾 Saving comprehensive spatial data array down to file path: {OUTPUT_PATH}")
    combined_gdf['RND_TIME'] = combined_gdf['RND_TIME'].dt.strftime('%Y-%m-%d %H:%M').fillna("")
    if 'VALID' in combined_gdf.columns:
        combined_gdf['VALID'] = combined_gdf['VALID'].dt.strftime('%Y-%m-%d %H:%M').fillna("")
        
    combined_gdf.to_file(OUTPUT_PATH)
    print("🎉 Pipeline successfully combined! All features executed via strict cloud-native extractions.")

if __name__ == "__main__":
    main()
