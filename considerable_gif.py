import os
import glob
import re
from PIL import Image

# =========================================================================
# 1. SETUP PATHS
# =========================================================================
raster_dir = "/Projects/flood_threat/model_automation/flood_tifs"
frame_input_dir = os.path.join(raster_dir, "considerable_2025_frames")
output_dir = os.path.join(raster_dir, "predictions")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

output_gif_path = os.path.join(output_dir, "Model2_Considerable_2025_compiled_timeline.gif")

# =========================================================================
# 2. LOCATE AND SORT PNG SAMPLES CHRONOLOGICALLY
# =========================================================================
print(f"Searching for individual PNG frames in: {frame_input_dir}")
png_files = glob.glob(os.path.join(frame_input_dir, "frame_*.png"))

if not png_files:
    raise FileNotFoundError(f"❌ Error: No matching 'frame_*.png' files found inside {frame_input_dir}")

# Sort files numerically using the frame index (e.g., frame_0001_..., frame_0002_...)
# This guarantees the time-series plays perfectly forward in time.
def extract_frame_number(filepath):
    match = re.search(r'frame_(\d+)_', os.path.basename(filepath))
    return int(match.group(1)) if match else 0

png_files = sorted(png_files, key=extract_frame_number)
print(f"Found {len(png_files)} frames. Commencing GIF compilation pipeline...")

# =========================================================================
# 3. LOAD IMAGES AND COMPILE TIME-SERIES ANIMATION
# =========================================================================
frames = []

for idx, f in enumerate(png_files, 1):
    print(f"   -> Appending frame asset {idx}/{len(png_files)} ({os.path.basename(f)})...", end="\r")
    # Open image layer and convert to clean RGB space safely
    img = Image.open(f).convert("RGB")
    frames.append(img)

print("\n⏳ Writing and encoding multi-frame animated file layer...")

# Save the matrix sequence out cleanly
if frames:
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=400,  # 400 milliseconds per frame (~2.5 frames per second)
        loop=0         # 0 means loop infinitely
    )
    print(f"🎉 Success! High-visibility animated timeline saved to:\n{output_gif_path}")
else:
    print("❌ Process failed: No valid frame data could be collected into RAM.")