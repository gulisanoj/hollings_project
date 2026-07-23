import datetime
import os
import subprocess
import sys
from herbie import Herbie
from osgeo import gdal
import numpy as np

# ==============================================================================
# CONFIGURATION & TARGET TIMES
# ==============================================================================
TARGET_DIR = "/Projects/flood_threat/HRRR/"
os.makedirs(TARGET_DIR, exist_ok=True)

# Your exact 2025 target storm runs
model_runs = [
    (2025, 5, 13, 0),
    (2025, 6, 5, 3),
    (2025, 6, 12, 18),
    (2025, 6, 13, 22),
    (2025, 7, 9, 18)
]

# Tracks consecutive forecast offsets dynamically
forecast_hours = [4, 5, 6]

# ==============================================================================
# BATCH PROCESSING ENGINE
# ==============================================================================
print("==================================================")
print("   HRRR 3-HOUR CUMULATIVE GENERATOR (EPSG:5070)   ")
print("==================================================")

for r_idx, (year, month, day, hour) in enumerate(model_runs, start=1):
    dt = datetime.datetime(year, month, day, hour)
    timestamp_str = dt.strftime("%Y%m%d_%Hz")
    
    print(f"\n🚀 Run [{r_idx}/{len(model_runs)}]: {dt.strftime('%B %d, %Y - %H:00 UTC')}")
    print("--------------------------------------------------")
    
    hourly_tiffs = {}
    run_failed = False
    
    # --------------------------------------------------------------------------
    # STEP 1: Download and Warp Individual HRRR Hours
    # --------------------------------------------------------------------------
    for fxx in forecast_hours:
        H = Herbie(
            dt, 
            model="hrrr", 
            product="sfc", 
            fxx=fxx
        )
        
        try:
            print(f"  ⏳ Fetching HRRR F{fxx:02d} from AWS Archive...")
            grib_file_object = H.download()
            local_grib = str(grib_file_object) if grib_file_object and os.path.exists(str(grib_file_object)) else H.get_localFilePath()
            
            if not os.path.exists(local_grib):
                print(f"    ❌ Skipped: GRIB file missing for F{fxx:02d}")
                run_failed = True
                break
        except Exception as e:
            print(f"    ❌ Herbie error on F{fxx:02d}: {e}")
            run_failed = True
            break
            
        output_filename = f"hrrr_{timestamp_str}_apcp_f{fxx:02d}_5070.tif"
        output_tiff = os.path.join(TARGET_DIR, output_filename)
        hourly_tiffs[fxx] = output_tiff
        
        # Warp curvilinear grids cleanly using the environment's native GDAL
        cmd_warp = [
            "gdalwarp", "-b", "1", "-t_srs", "EPSG:5070", 
            "-dstnodata", "NaN", "-co", "COMPRESS=DEFLATE", "-overwrite",
            local_grib, output_tiff
        ]
        subprocess.run(cmd_warp, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"    ✅ Generated hourly raster: {output_filename}")

    if run_failed:
        print("    ⚠️ Skipping cumulative step for this run due to missing source files.")
        continue

    # --------------------------------------------------------------------------
    # STEP 2: Compute 3-Hour Cumulative Raster via Pure Python NumPy Matrix Math
    # --------------------------------------------------------------------------
    cumulative_filename = f"hrrr_{timestamp_str}_apcp_3hr_cum_5070.tif"
    cumulative_tiff = os.path.join(TARGET_DIR, cumulative_filename)
    
    print(f"    🧮 Summing grids via clean NumPy array matrix...")
    
    try:
        # Dynamically extract and sort keys based on whatever hours were processed
        h_keys = sorted(list(hourly_tiffs.keys()))
        
        # Open the three freshly generated hourly GeoTIFF files via dynamic list handles
        ds1 = gdal.Open(hourly_tiffs[h_keys[0]])
        ds2 = gdal.Open(hourly_tiffs[h_keys[1]])
        ds3 = gdal.Open(hourly_tiffs[h_keys[2]])
        
        # Extract bands as numeric matrices
        arr1 = ds1.GetRasterBand(1).ReadAsArray().astype(np.float32)
        arr2 = ds2.GetRasterBand(1).ReadAsArray().astype(np.float32)
        arr3 = ds3.GetRasterBand(1).ReadAsArray().astype(np.float32)
        
        # Strip out NoData values and DBL_MAX boundary corruption masks
        for arr in [arr1, arr2, arr3]:
            arr[np.isnan(arr)] = 0.0
            arr[arr > 10000.0] = 0.0
            arr[arr < 0.0] = 0.0
            
        # Add values together pixel-by-pixel
        cum_arr = arr1 + arr2 + arr3
        
        # Construct and save the clean output raster file
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            cumulative_tiff, 
            ds1.RasterXSize, 
            ds1.RasterYSize, 
            1, 
            gdal.GDT_Float32, 
            options=["COMPRESS=DEFLATE"]
        )
        out_ds.SetGeoTransform(ds1.GetGeoTransform())
        out_ds.SetProjection(ds1.GetProjection())
        
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(cum_arr)
        out_band.SetNoDataValue(0.0)  # Establish 0.0 as the clear background NoData mask
        
        # Dereference and flush datasets safely to local disk storage
        out_band = None
        out_ds = None
        ds1 = ds2 = ds3 = None
        
        print(f"    🎉 SUCCESS! Created Cumulative Raster:\n    --> {cumulative_filename}")
        
    except Exception as e:
        print(f"    ❌ Array calculation failure on run: {e}")

print("\n==================================================")
print(f"📊 Process complete! Clean 2025 files dropped in: {TARGET_DIR}")
print("==================================================")
