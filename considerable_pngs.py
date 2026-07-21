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
# 1. PATHS & TARGETING DATA LAYERS
# =========================================================================
raster_dir = "/Projects/flood_threat/model_automation/flood_tifs"
prediction_dir = os.path.join(raster_dir, "predictions")
ffw_path = os.path.join(raster_dir, "flash_flood_warnings/huc0305_ffw.shp")
lsr_path = "/Projects/flood_threat/HUC0305/lsr_final/2025_lsrs.shp"
huc_shp_path = "/Projects/flood_threat/HUC0305/forest_features/intermediate/huc0305_boundary.shp"

# Create a clean subfolder explicitly for individual PNG frame outputs
frame_output_dir = os.path.join(raster_dir, "considerable_2025_frames")
os.makedirs(frame_output_dir, exist_ok=True)

def parse_valid_time(val):
    val_str = str(val).strip()
    try:
        return datetime.datetime.strptime(val_str[:12], "%Y%m%d%H%M")
    except Exception:
        return None

# --- Process HUC 0305 Watershed Boundary ---
print("Loading HUC 0305 watershed boundary...")
gdf_huc = gpd.read_file(huc_shp_path, engine="fiona")
if gdf_huc.crs is None: 
    gdf_huc.crs = "EPSG:4326"  
gdf_huc = gdf_huc.to_crs("EPSG:3857")

# --- Process Your New 2025 LSR Shapefile ---
print("Loading 2025 LSR validation shapefile...")
gdf_lsr = gpd.read_file(lsr_path, engine="fiona")
if gdf_lsr.crs is None:
    gdf_lsr.crs = "EPSG:4326"  
gdf_lsr = gdf_lsr.to_crs("EPSG:3857")

gdf_lsr['VALID_DT'] = gdf_lsr['VALID'].apply(parse_valid_time)
gdf_lsr = gdf_lsr.dropna(subset=['VALID_DT']).copy()
gdf_lsr['VALID_DAY_STR'] = gdf_lsr['VALID_DT'].dt.strftime("%Y%m%d")

# Find the severity column from your LLM process
cls_col = [c for c in gdf_lsr.columns if c.lower() == 'fld_sv_cls']
if not cls_col:
    raise KeyError("Could not find the 'FLD_SV_CLS' attribute column. Please run the classifier first.")
cls_col = cls_col[0]

# CRITICAL FILTER: Keep only MODERATE and SEVERE reports right from the start
print(f"Filtering layer to keep only MODERATE and SEVERE reports from {cls_col}...")
gdf_lsr = gdf_lsr[gdf_lsr[cls_col].astype(str).str.upper().isin(["MODERATE", "SEVERE"])].copy()

print("Scaling remaining 2025 LSR storm report geometries 5x larger for mapping visibility...")
gdf_lsr['geometry'] = gdf_lsr.geometry.scale(xfact=5.0, yfact=5.0, origin='centroid')

# --- Process Flash Flood Warning (FFW) Shapefile ---
print("Loading Flash Flood Warning shapefile...")
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
# 2. DEFINE BOUNDING BOX & FETCH FIXED BASEMAP
# =========================================================================
huc_bounds_4326 = [-83.5, 32.0, -78.0, 36.5] 
print("Fetching Esri World Imagery basemap...")
xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", "EPSG:3857", *huc_bounds_4326)

bg_img, bg_ext = cx.bounds2img(xmin, ymin, xmax, ymax, zoom=9, source=cx.providers.Esri.WorldImagery)
bg_pil_template = Image.fromarray(bg_img).convert("RGBA")
target_size = bg_pil_template.size  

cx_xmin, cx_xmax, cx_ymin, cx_ymax = bg_ext

# =========================================================================
# 3. HELPER FUNCTION TO TRANSFORM GEOMETRY TO PIXEL COORDINATES
# =========================================================================
def get_pixel_coords(geom, xmin, ymin, xmax, ymax, img_w, img_h):
    if geom.is_empty:
        return ("UNKNOWN", [])
    x_scale = img_w / (xmax - xmin)
    y_scale = img_h / (ymax - ymin)
    
    if geom.geom_type in ['Point', 'MultiPoint']:
        points = [geom] if geom.geom_type == 'Point' else list(geom.geoms)
        pixel_pts = []
        for p in points:
            px = int((p.x - xmin) * x_scale)
            py = int((ymax - p.y) * y_scale)
            pixel_pts.append((px, py))
        return ("POINT", pixel_pts)
        
    elif geom.geom_type in ['Polygon', 'MultiPolygon']:
        geoms = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        coords = []
        for g in geoms:
            exterior = list(g.exterior.coords)
            pixel_poly = []
            for x, y in exterior:
                px = int((x - xmin) * x_scale)
                py = int((ymax - y) * y_scale)  
                pixel_poly.append((px, py))
            coords.append(pixel_poly)
        return ("POLYGON", coords)
        
    return ("UNKNOWN", [])

# =========================================================================
# 4. COMPILE AND EXPORT INDIVIDUAL STATIC IMAGE ASSET FRAMES
# =========================================================================
model_name = "Model2_Considerable_vs_None"
print(f"\nExporting individual image frames strictly for {model_name}...")
tif_files = sorted(glob.glob(os.path.join(prediction_dir, f"*{model_name}.tif")))

frame_counter = 0
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
    
    # --- Layer 2, 3 & 4: Vector Elements Overlay ---
    active_lsrs = gdf_lsr[(gdf_lsr['VALID_DT'] >= start_buffer) & (gdf_lsr['VALID_DT'] <= end_buffer)]
    active_ffws = gdf_ffw[(gdf_ffw['VALID_DT'] >= start_buffer) & (gdf_ffw['VALID_DT'] <= end_buffer)]
    
    if not active_lsrs.empty or not active_ffws.empty or not gdf_huc.empty:
        vector_overlay = Image.new("RGBA", target_size, (0, 0, 0, 0))
        vector_draw = ImageDraw.Draw(vector_overlay)
        
        # Draw filtered LSR layers (Only Moderate/Severe)
        for _, row in active_lsrs.iterrows():
            geom_type, tracks = get_pixel_coords(row.geometry, cx_xmin, cx_ymin, cx_xmax, cx_ymax, target_size[0], target_size[1])
            
            if geom_type == "POLYGON":
                for poly_nodes in tracks:
                    if len(poly_nodes) < 3: continue
                    vector_draw.polygon(poly_nodes, fill=(255, 235, 0, 130))
                    vector_draw.polygon(poly_nodes, outline=(235, 160, 0, 255), width=2)
            elif geom_type == "POINT":
                for px, py in tracks:
                    r = 14
                    vector_draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 235, 0, 230), outline=(235, 160, 0, 255), width=3)
        
        # Draw Flash Flood Warning layers
        for _, row in active_ffws.iterrows():
            geom_type, tracks = get_pixel_coords(row.geometry, cx_xmin, cx_ymin, cx_xmax, cx_ymax, target_size[0], target_size[1])
            
            if geom_type == "POLYGON":
                for poly_nodes in tracks:
                    if len(poly_nodes) < 3: continue
                    vector_draw.polygon(poly_nodes, outline=(57, 255, 20, 255), width=3)
            elif geom_type == "POINT":
                for px, py in tracks:
                    r = 14
                    vector_draw.ellipse([px - r, py - r, px + r, py + r], outline=(57, 255, 20, 255), width=4)
                    
        # Draw HUC 0305 Boundary (Solid Blue Outline)
        for _, row in gdf_huc.iterrows():
            geom_type, tracks = get_pixel_coords(row.geometry, cx_xmin, cx_ymin, cx_xmax, cx_ymax, target_size[0], target_size[1])
            if geom_type == "POLYGON":
                for poly_nodes in tracks:
                    if len(poly_nodes) < 3: continue
                    vector_draw.polygon(poly_nodes, outline=(0, 100, 255, 255), width=4)
            elif geom_type == "POINT":
                for px, py in tracks:
                    r = 8
                    vector_draw.ellipse([px - r, py - r, px + r, py + r], outline=(0, 100, 255, 255), width=4)

        frame_canvas.alpha_composite(vector_overlay)
        
    # --- Layer 5: Legend UI overlays ---
    draw = ImageDraw.Draw(frame_canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 54)
    except IOError:
        font = ImageFont.load_default()
            
    draw.rectangle([10, 10, 720, 115], fill=(0, 0, 0, 220))
    draw.text((30, 25), label_text, fill=(255, 255, 255), font=font)
    
    draw.rectangle([10, target_size[1] - 315, 870, target_size[1] - 10], fill=(0, 0, 0, 220))
    
    draw.rectangle([30, target_size[1] - 285, 90, target_size[1] - 235], fill=(255, 0, 0, 255))
    draw.text((120, target_size[1] - 290), "Considerable Model Threat", fill=(255, 255, 255), font=font)
    
    draw.rectangle([30, target_size[1] - 205, 90, target_size[1] - 155], fill=(255, 235, 0, 255))
    draw.text((120, target_size[1] - 210), "Active Mod/Sev LSR (5x, +/-3hr)", fill=(255, 255, 255), font=font)
    
    draw.rectangle([30, target_size[1] - 125, 90, target_size[1] - 75], outline=(57, 255, 20, 255), width=6)
    draw.text((120, target_size[1] - 130), "Flash Flood Warning (+/-3hr)", fill=(255, 255, 255), font=font)
    
    frame_counter += 1
    output_png_name = f"frame_{frame_counter:04d}_{year}{month}{day}-{hour}{minute}.png"
    output_png_path = os.path.join(frame_output_dir, output_png_name)
    
    frame_canvas.save(output_png_path, "PNG")
    print(f"-> Saved individual frame: {output_png_name}", end="\r")

print(f"\n\n🎉 Done! All separate high-vis PNG maps saved smoothly to:\n{frame_output_dir}")