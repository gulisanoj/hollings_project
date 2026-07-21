import os
import requests
from datetime import datetime, timedelta

# =========================================================================
# 1. SETUP CONFIGURATIONS & PATHS
# =========================================================================
# Output directory for raw compressed files
output_dir = "/Projects/flood_threat/model_automation/flood_tifs/QPE/"
os.makedirs(output_dir, exist_ok=True)

# AWS Public S3 HTTP Base Endpoint
S3_BASE = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS"

# Define target processing intervals (April 1, 2025 to September 30, 2025 at 3-hour intervals)
start_dt = datetime(2025, 4, 1, 0, 0)
end_dt = datetime(2025, 9, 30, 21, 0)
step = timedelta(hours=3)

# =========================================================================
# 2. DOWNLOAD LOOP
# =========================================================================
current_dt = start_dt
while current_dt <= end_dt:
    print(f"\nQueueing time interval target: {current_dt.strftime('%Y-%m-%d %H:%M')}")
    
    # Calculate offset timestamps relative to target window T
    t_minus_1h = current_dt - timedelta(hours=1)
    t_minus_2h = current_dt - timedelta(hours=2)
    t_minus_3h = current_dt - timedelta(hours=3)
    
    # Define exact product names and target timestamps
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
        
        # Build URL and local file paths
        filename = f"MRMS_{product}_{date_str}-{time_str}.grib2.gz"
        url = f"{S3_BASE}/{product}/{date_str}/{filename}"
        local_path = os.path.join(output_dir, filename)
        
        # Skip download if the file already exists locally (saves time if resumed)
        if os.path.exists(local_path):
            continue
            
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print(f"   [Downloaded] {filename}")
            else:
                print(f"   [Missing on S3] {filename} (Status: {response.status_code})")
        except Exception as e:
            print(f"   [Connection Error] Failed to download {filename}: {e}")
            
    current_dt += step

print("\nAll available raw MRMS files downloaded successfully.")