import os
import glob
import re
from PIL import Image

# =========================================================================
# CONFIGURATION
# =========================================================================
input_dir = "/Projects/flood_threat/model_automation/flood_tifs/individual_frames/Model1_Any_vs_None"
output_gif_path = os.path.join(input_dir, "optimized_flood_timeline.gif")

target_width = 1000  # Downscales dimensions to save major file space
frame_skip = 1       # Set to 2 to take every second frame if file size is still too large
frame_duration = 500 # Time per frame in milliseconds (500ms = 2 frames per second)

# =========================================================================
# COMPILATION ENGINE
# =========================================================================
print("🔍 Searching for frame assets...")
png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

if not png_files:
    print(f"❌ No PNG files found in {input_dir}. Please check your path.")
    exit()

print(f"📦 Found {len(png_files)} frames. Beginning optimization...")

frames = []
for idx, file_path in enumerate(png_files):
    # Apply frame skipping if enabled
    if idx % frame_skip != 0:
        continue
       
    with Image.open(file_path) as img:
        # Convert to RGBA to preserve composite mapping layers clean layout
        img_rgba = img.convert("RGBA")
       
        # Calculate aspect ratio responsive resize dimensions
        w, h = img_rgba.size
        scale_factor = target_width / float(w)
        target_height = int(float(h) * scale_factor)
       
        # Resize using high-quality resampling
        img_resized = img_rgba.resize((target_width, target_height), Image.Resampling.LANCZOS)
       
        # Convert to Adaptive Palette with restricted colors to force heavy compression
        img_indexed = img_resized.convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
       
        frames.append(img_indexed)
        print(f"-> Processed frame {idx+1}/{len(png_files)}", end="\r")

print(f"\n💾 Saving highly compressed GIF to: {output_gif_path}...")

# Export optimized presentation asset
frames[0].save(
    output_gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=frame_duration,
    loop=0,         # 0 means loop infinitely
    optimize=True   # Enables pillow's internal palette/pixel optimization flags
)

print(f"⚡ Success! Optimized asset generated completely ({os.path.getsize(output_gif_path) / (1024*1024):.2f} MB).")