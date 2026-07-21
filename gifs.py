import os
import glob
import re
import datetime
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.vrt import WarpedVRT
import geopandas as gpd
import fiona  
import contextily as cx
from PIL import Image, ImageDraw, ImageFont

# =========================================================================
# 1. PATHS & INITIAL LAYER LOAD (WITH EXPLICIT COLUMN ALIGNMENTS)
# =========================================================================
# Updated paths to map directly to your HUC1028 workspace tree
raster_dir = "/Projects/flood_threat/HUC1028"
output_dir = os.path.join(raster_dir, "predictions")
lsr_path = "/Projects/flood_threat/HUC1028/buffers_with_precip.shp"
ffw_path = os.path.join(raster_dir, "ffws.shp")

def parse_valid_time(val):
    val_str = str(val).strip()
    try:
        return datetime.datetime.strptime(val_str[:12], "%Y%m%d%H%M")
    except Exception:
        return None

# --- Process LSR Shapefile ---
print("Loading LSR validation shapefile...")
gdf_lsr = gpd.read_file(lsr_path, engine="fiona")
if gdf_lsr.crs is None:
    gdf_lsr.crs = "EPSG:5070"  
gdf_lsr = gdf_lsr.to_crs("EPSG:3857")

# Unified time tracking using only the VALID field
gdf_lsr['VALID_DT'] = gdf_lsr['VALID'].apply(parse_valid_time)
gdf_lsr = gdf_lsr.dropna(subset=['VALID_DT']).copy()
gdf_lsr['VALID_DAY_STR'] = gdf_lsr['VALID_DT'].dt.strftime("%Y%m%d")

print("Scaling LSR storm report polygons 5x larger...")
gdf_lsr['geometry'] = gdf_lsr.geometry.scale(xfact=5.0, yfact=5.0, origin='centroid')

# --- Process Flash Flood Warning (FFW) Shapefile ---
print("Loading HUC 1028 Flash Flood Warning shapefile...")
gdf_ffw = gpd.read_file(ffw_path, engine="fiona")
if gdf_ffw.crs is None:
    gdf_ffw.crs = "EPSG:4326"  
gdf_ffw = gdf_ffw.to_crs("EPSG:3857")

gdf_ffw['VALID_DT'] = gdf_ffw['ISSUED'].apply(parse_valid_time)
gdf_ffw = gdf_ffw.dropna(subset=['VALID_DT']).copy()
gdf_ffw['VALID_DAY_STR'] = gdf_ffw['VALID_DT'].dt.strftime("%Y%m%d")

# Combine active reporting days from BOTH layers
lsr_days = set(gdf_lsr['VALID_DAY_STR'].unique())
ffw_days = set(gdf_ffw['VALID_DAY_STR'].unique())
active_combined_days = lsr_days.union(ffw_days)
print(f"Identified {len(active_combined_days)} total filtered calendar days to process.")

# =========================================================================
# 2. DEFINE HUC 1028 BOUNDING BOX & FETCH BASEMAP
# =========================================================================
# Boundaries configured for Chariton-Grand basin (North Missouri / South Iowa)
huc_bounds_4326 = [-95.0, 39.0, -91.5, 42.0] 
print("Fetching Esri World Imagery basemap for HUC 1028 bounds...")
xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", "EPSG:3857", *huc_bounds_4326)

# Fetch basemap image and its ACTUAL tile-snapped coordinate extents
bg_img, bg_ext = cx.bounds2img(xmin, ymin, xmax, ymax, zoom=9, source=cx.providers.Esri.WorldImagery)
bg_pil_template = Image.fromarray(bg_img).convert("RGBA")
target_size = bg_pil_template.size  

# CRITICAL FIX: Unpack contextily's true tile-snapped bounds to anchor all layout coordinate logic
cx_xmin, cx_xmax, cx_ymin, cx_ymax = bg_ext

# =========================================================================
# 3. HELPER FUNCTION TO TRANSFORM GEOMETRY TO PIXEL COORDINATES
# =========================================================================
def get_pixel_coords(geom, xmin, ymin, xmax, ymax, img_w, img_h):
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
            py = int((ymax - y) * y_scale)  
            pixel_poly.append((px, py))
        coords.append(pixel_poly)
    return coords

# =========================================================================
# 4. COMPILE ANIMATION TIME-SERIES WITH LARGE MAP LABELS
# =========================================================================
for model_name in ["Model1_Any_vs_None", "Model2_Considerable_vs_None"]:
    print(f"\nProcessing animation loop for {model_name}...")
    tif_files = sorted(glob.glob(os.path.join(output_dir, f"*{model_name}.tif")))
    
    frames = []
    for f in tif_files:
        fname = os.path.basename(f)
        ts_match = re.search(r'(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})', fname)
        
        if not ts_match:
            continue
            
        year, month, day, hour, minute = ts_match.groups()
        file_day_str = f"{year}{month}{day}"
        
        if file_day_str not in active_combined_days:
            continue
            
        label_text = f"{year}-{month}-{day} {hour}:{minute}Z"
        current_frame_dt = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute))
        
        start_buffer = current_frame_dt - datetime.timedelta(hours=3)
        end_buffer = current_frame_dt + datetime.timedelta(hours=3)
        
        # FIX: Read the raster window matching the exact tile-snapped boundaries
        with rasterio.open(f) as src:
            with WarpedVRT(src, crs="EPSG:3857") as vrt:
                window = vrt.window(cx_xmin, cx_ymin, cx_xmax, cx_ymax)
                data = vrt.read(1, window=window, out_shape=(target_size[1], target_size[0]))
        
        frame_canvas = bg_pil_template.copy()
        
        # --- Layer 1: Draw Model Predictions (Red) ---
        overlay_rgba = np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8)
        overlay_rgba[data == 1] = [255, 0, 0, 180]  
        overlay_img = Image.fromarray(overlay_rgba)
        frame_canvas.alpha_composite(overlay_img)
        
        # --- Layer 2: Draw Active LSR Polygons (Yellow / 5x) ---
        active_lsrs = gdf_lsr[
            (gdf_lsr['VALID_DT'] >= start_buffer) & 
            (gdf_lsr['VALID_DT'] <= end_buffer)
        ]
        
        # --- Layer 3: Draw Active Flash Flood Warnings (Neon Green Outlines) ---
        active_ffws = gdf_ffw[
            (gdf_ffw['VALID_DT'] >= start_buffer) & 
            (gdf_ffw['VALID_DT'] <= end_buffer)
        ]
        
        if not active_lsrs.empty or not active_ffws.empty:
            vector_overlay = Image.new("RGBA", target_size, (0, 0, 0, 0))
            vector_draw = ImageDraw.Draw(vector_overlay)
            
            # FIX: Transform vector geometries using the tile-snapped contextily extents
            for _, row in active_lsrs.iterrows():
                poly_pixel_tracks = get_pixel_coords(row.geometry, cx_xmin, cx_ymin, cx_xmax, cx_ymax, target_size[0], target_size[1])
                for poly_nodes in poly_pixel_tracks:
                    if len(poly_nodes) < 3: continue
                    vector_draw.polygon(poly_nodes, fill=(255, 235, 0, 130))
                    vector_draw.polygon(poly_nodes, outline=(235, 160, 0, 255), width=2)
            
            for _, row in active_ffws.iterrows():
                poly_pixel_tracks = get_pixel_coords(row.geometry, cx_xmin, cx_ymin, cx_xmax, cx_ymax, target_size[0], target_size[1])
                for poly_nodes in poly_pixel_tracks:
                    if len(poly_nodes) < 3: continue
                    vector_draw.polygon(poly_nodes, outline=(57, 255, 20, 255), width=3)
            
            frame_canvas.alpha_composite(vector_overlay)
            
        # --- Layer 4: Draw Expanded Timestamp Labels and Legends ---
        draw = ImageDraw.Draw(frame_canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 54)
        except IOError:
            font = ImageFont.load_default()
                
        # Top-left metadata timestamp window banner
        draw.rectangle([10, 10, 720, 115], fill=(0, 0, 0, 220))
        draw.text((30, 25), label_text, fill=(255, 255, 255), font=font)
        
        # Expanded multi-product map legend box dimensions in bottom-left corner
        draw.rectangle([10, target_size[1] - 315, 870, target_size[1] - 10], fill=(0, 0, 0, 220))
        
        # Red: Model Threat
        draw.rectangle([30, target_size[1] - 285, 90, target_size[1] - 235], fill=(255, 0, 0, 255))
        draw.text((120, target_size[1] - 290), "Model Predicts Threat", fill=(255, 255, 255), font=font)
        
        # Yellow: LSR
        draw.rectangle([30, target_size[1] - 205, 90, target_size[1] - 155], fill=(255, 235, 0, 255))
        draw.text((120, target_size[1] - 210), "Active LSR (5x Scale, +/-3hr)", fill=(255, 255, 255), font=font)
        
        # Neon Green: FFW Bounds
        draw.rectangle([30, target_size[1] - 125, 90, target_size[1] - 75], outline=(57, 255, 20, 255), width=6)
        draw.text((120, target_size[1] - 130), "Flash Flood Warning (+/-3hr)", fill=(255, 255, 255), font=font)
        
        frames.append(frame_canvas)
        print(f"-> Frame: {label_text} | LSRs: {len(active_lsrs)} | FFWs: {len(active_ffws)}", end="\r")
        
    if frames:
        gif_path = os.path.join(output_dir, f"{model_name}_HUC1028_validated_with_warnings.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=400,  
            loop=0
        )
        print(f"\n[+] Saved High-Visibility Warning-Validated GIF: {gif_path}")

print("\nMulti-product situational context animation loops completed successfully!")