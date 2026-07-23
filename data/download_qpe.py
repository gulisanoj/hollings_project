from datetime import datetime, timedelta
import os
import time
import geopandas as gpd
import pandas as pd
from pyproj import Transformer
import requests

# =========================================================================
# 1. SETUP CONFIGURATIONS & PATHS
# =========================================================================
# Output workspace directory for the downloaded MRMS QPE grids
output_dir = "/Projects/flood_threat/HUC0304/QPE/"
os.makedirs(output_dir, exist_ok=True)

# Path to your localized LSR shapefile 
LSR_SHP_PATH = "/Projects/flood_threat/HUC0304/2025_lsrs.shp"

# AWS Public S3 HTTP Base Endpoint for MRMS CONUS products
S3_BASE = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS"


# =========================================================================
# 2. CORE HELPER FUNCTIONS
# =========================================================================
def extract_lsr_dates(shp_path):
    """Reads the LSR shapefile and returns a sorted list of unique date objects."""
    if not os.path.exists(shp_path):
        print(f"❌ Error: Shapefile not found at {shp_path}")
        return []

    print(f"📖 Reading LSR shapefile from: {shp_path}...")
    gdf = gpd.read_file(shp_path)

    # Standardize column casing
    if "VALID" not in gdf.columns:
        if "valid" in gdf.columns:
            gdf["VALID"] = gdf["valid"]
        else:
            raise KeyError("The shapefile does not contain a 'VALID' column for dates.")

    # Convert to datetime and isolate the exact calendar date (YYYY-MM-DD)
    gdf["VALID_DATE"] = pd.to_datetime(gdf["VALID"]).dt.date
    unique_dates = sorted(gdf["VALID_DATE"].dropna().unique())

    print(f"🎯 Extracted {len(unique_dates)} unique days with LSR events.")
    return unique_dates


def download_mrms_for_date(target_date, output_folder):
    """Generates 3-hour interval sequences for a target day and pulls MRMS QPE files."""
    # Set the time boundary strictly for the target storm day (00:00 to 21:00 UTC)
    start_dt = datetime(target_date.year, target_date.month, target_date.day, 0, 0)
    end_dt = datetime(target_date.year, target_date.month, target_date.day, 21, 0)
    step = timedelta(hours=3)

    current_dt = start_dt
    while current_dt <= end_dt:
        print(f"\nQueueing interval window target: {current_dt.strftime('%Y-%m-%d %H:%M')}")
        
        # Calculate offset timestamps relative to target window T
        t_minus_1h = current_dt - timedelta(hours=1)
        t_minus_2h = current_dt - timedelta(hours=2)
        t_minus_3h = current_dt - timedelta(hours=3)
        
        # Product configuration matching your pipeline sequence matrix
        mrms_requests = [
            ("MultiSensor_QPE_01H_Pass2_00.00", current_dt),
            ("MultiSensor_QPE_01H_Pass2_00.00", t_minus_1h),
            ("MultiSensor_QPE_01H_Pass2_00.00", t_minus_2h),
            ("MultiSensor_QPE_03H_Pass2_00.00", current_dt),
            ("MultiSensor_QPE_24H_Pass2_00.00", t_minus_3h),
            ("MultiSensor_QPE_48H_Pass2_00.00", t_minus_3h)
        ]
        
        for product, dt in mrms_requests:
            date_str = dt.strftime("%Y%m%d")
            time_str = dt.strftime("%H%M%S")
            
            filename = f"MRMS_{product}_{date_str}-{time_str}.grib2.gz"
            url = f"{S3_BASE}/{product}/{date_str}/{filename}"
            local_path = os.path.join(output_folder, filename)
            
            if os.path.exists(local_path):
                print(f"   [Skipped] {filename} already exists.")
                continue
                
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    print(f"   [Downloaded] {filename}")
                else:
                    print(f"   [Missing on S3] {filename} (Status: {response.status_code})")
            except Exception as e:
                print(f"   [Connection Error] Failed to fetch {filename}: {e}")
                
        current_dt += step


# =========================================================================
# 3. RUNTIME EXECUTION
# =========================================================================
if __name__ == "__main__":
    print("==================================================")
    print("      LOWER-48 AUTOMATED MRMS QPE DOWNLOADER      ")
    print("==================================================")

    # 1. Parse LSR shapefile for target storm dates
    lsr_dates = extract_lsr_dates(LSR_SHP_PATH)

    if not lsr_dates:
        print("❌ Stopping execution. No target operational dates extracted.")
    else:
        print("\n[Execution Protocol] Beginning filtered event loop based on LSR records...")
        
        # 2. Loop over each isolated storm event day discovered in the shapefile
        for idx, event_day in enumerate(lsr_dates):
            print(f"\n📅 --- Processing Event Day [{idx + 1}/{len(lsr_dates)}]: {event_day.strftime('%Y-%m-%d')} ---")
            
            # Run the S3 downloading sequence for this specific date
            download_mrms_for_date(event_day, output_dir)
            
            # Subtle delay to prevent overwhelming connection limits
            time.sleep(1.0)
            
        print("\n🚀 Complete processing pipeline finished successfully.")
