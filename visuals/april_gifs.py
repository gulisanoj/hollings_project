import os
import glob
import re
import datetime
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.vrt import WarpedVRT
import geopandas as gpd
import contextily as cx
from PIL import Image, ImageDraw, ImageFont
import fiona

# =========================================================================
# 1. PATHS & INITIAL LAYER LOAD
# =========================================================================
output_dir = "/Projects/flood_threat/model_automation/flood_tifs/predictions"
shp_path = "/Projects/flood_threat/model_automation/training_data/buffered_lsr_with_precip.shp"

print("Loading validation shapefile...")
# Load the 500m buffer shapefile and project to Web Mercator to match the basemap
gdf_lsr = gpd.read_file(shp_path, engine="fiona")
gdf_lsr = gdf_lsr.to_crs("EPSG:3857")

# Parse the VALID column into actual Python datetime objects for precise mathematical logic
def parse_valid_time(val):
    val_str = str(val).strip()
    try:
        # Expected format: '202504242230' -> YYYYMMDDHHMM
        return datetime.datetime.strptime(val_str[:12], "%Y%m%d%H%M")
    except Exception:
        return None

# CRITICAL FIX: Explicitly referencing the exact column name 'VALID'
gdf_lsr['VALID_DT'] = gdf_lsr['VALID'].apply(parse_valid_time)
# Drop any rows that don't have a valid timestamp format
gdf_lsr = gdf_lsr.dropna(subset=['VALID_DT']).copy()

# =========================================================================
# 2. DEFINE HUC 0305 BOUNDING BOX & FETCH BASEMAP
# =========================================================================
huc_bounds_4326 = [-83.5, 32.0, -78.0, 36.5] 
print("Fetching Esri World Imagery basemap for HUC 0305 bounds...")
xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", "EPSG:3857", *huc_bounds_4326)

bg_img, bg_ext = cx.bounds2img(xmin, ymin, xmax, ymax, zoom=9, source=cx.providers.Esri.WorldImagery)
bg_pil_template = Image.fromarray(bg_img).convert("RGBA")
target_size = bg_pil_template.size  

# =========================================================================
# 3. HELPER FUNCTION TO TRANSFORM GEOMETRY TO PIXEL COORDINATES
# =========================================================================
def get_pixel_coords(geom, xmin, ymin, xmax, ymax, img_w, img_h):
    """Converts Web Mercator coordinates to pixel XY space on our image canvas"""
    if geom.is_empty:
        return []
    
    x_scale = img_w / (xmax - xmin)
    y_scale = img_h / (ymax - ymin)
    
    coords = []
    geoms = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    
    for g in geoms:
        exterior = list(g.exterior.coords)
        pixel_poly = []
        for x, y in exterior:
            px = int((x - xmin) * x_scale)
            py = int((ymax - y) * y_scale)  # Flip Y for image pixel coordinates
            pixel_poly.append((px, py))
        coords.append(pixel_poly)
    return coords

# =========================================================================
# 4. COMPILE ANIMATION TIME-SERIES WITH POLYGON VISIBILITY WINDOWS
# =========================================================================
for model_name in ["Model1_Any_vs_None", "Model2_Considerable_vs_None"]:
    print(f"\nProcessing animation loop for {model_name}...")
    tif_files = sorted(glob.glob(os.path.join(output_dir, f"*{model_name}.tif")))
    
    frames = []
    for f in tif_files:
        fname = os.path.basename(f)
        # Extract timestamp from filename: '20250424-223000' -> YYYYMMDD-HHMMSS
        ts_match = re.search(r'(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})', fname)
        
        if not ts_match:
            continue
            
        year, month, day, hour, minute = ts_match.groups()
        label_text = f"{year}-{month}-{day} {hour}:{minute}Z"
        
        # Convert current map file frame step to a datetime object
        current_frame_dt = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute))
        
        # Define our temporal search buffers (3 hours back, 3 hours forward)
        start_buffer = current_frame_dt - datetime.timedelta(hours=3)
        end_buffer = current_frame_dt + datetime.timedelta(hours=3)
        
        # Read and crop the prediction raster
        with rasterio.open(f) as src:
            with WarpedVRT(src, crs="EPSG:3857") as vrt:
                window = vrt.window(xmin, ymin, xmax, ymax)
                data = vrt.read(1, window=window, out_shape=(target_size[1], target_size[0]))
        
        # Create base canvas frame
        frame_canvas = bg_pil_template.copy()
        
        # --- Layer 1: Draw Model Predictions (Red) ---
        overlay_rgba = np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8)
        overlay_rgba[data == 1] = [255, 0, 0, 180]  
        overlay_img = Image.fromarray(overlay_rgba)
        frame_canvas.alpha_composite(overlay_img)
        
        # --- Layer 2: Draw Active Shapefile Polygons with 6-Hour Window Context ---
        # Select polygons valid within 3 hours before or 3 hours after this map's timestamp
        active_lsrs = gdf_lsr[
            (gdf_lsr['VALID_DT'] >= start_buffer) & 
            (gdf_lsr['VALID_DT'] <= end_buffer)
        ]
        
        if not active_lsrs.empty:
            vector_overlay = Image.new("RGBA", target_size, (0, 0, 0, 0))
            vector_draw = ImageDraw.Draw(vector_overlay)
            
            for _, row in active_lsrs.iterrows():
                poly_pixel_tracks = get_pixel_coords(row.geometry, xmin, ymin, xmax, ymax, target_size[0], target_size[1])
                
                for poly_nodes in poly_pixel_tracks:
                    if len(poly_nodes) < 3:
                        continue
                    # Translucent yellow fill inside the 500m area
                    vector_draw.polygon(poly_nodes, fill=(255, 235, 0, 130))
                    
                    # METHOD 1 BOLD OUTLINE: width=4 to ensure visibility on a regional scale
                    vector_draw.polygon(poly_nodes, outline=(235, 160, 0, 255), width=4)
            
            frame_canvas.alpha_composite(vector_overlay)
            
        # --- Layer 3: Draw Timestamp Labels and Legends ---
        draw = ImageDraw.Draw(frame_canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except IOError:
            font = ImageFont.load_default()
                
        # Top-left metadata banner
        draw.rectangle([10, 10, 240, 45], fill=(0, 0, 0, 200))
        draw.text((20, 18), label_text, fill=(255, 255, 255), font=font)
        
        # Quick visual legend box in bottom-left corner
        draw.rectangle([10, target_size[1]-70, 235, target_size[1]-10], fill=(0, 0, 0, 200))
        draw.rectangle([20, target_size[1]-60, 40, target_size[1]-45], fill=(255, 0, 0, 255))
        draw.text((50, target_size[1]-62), "Model Predicts Threat", fill=(255, 255, 255), font=font)
        draw.rectangle([20, target_size[1]-35, 40, target_size[1]-20], fill=(255, 235, 0, 255))
        draw.text((50, target_size[1]-37), "Active LSR (+/- 3hr Window)", fill=(255, 255, 255), font=font)
        
        frames.append(frame_canvas)
        print(f"-> Integrated frame tracks: {label_text} | Visible Polygons: {len(active_lsrs)}", end="\r")
        
    if frames:
        gif_path = os.path.join(output_dir, f"{model_name}_HUC0305_rolling_validation.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=400,  # Playback frame rate (400ms per interval)
            loop=0
        )
        print(f"\n[+] Saved Context-Validated GIF: {gif_path}")

print("\nAll ground-truth rolling validation animations completed successfully!")
