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
# 1. PATHS & INITIAL LAYER LOAD
# =========================================================================
raster_dir = "/Projects/flood_threat/HUC1028"
prediction_dir = os.path.join(raster_dir, "predictions")
lsr_path = "/Projects/flood_threat/HUC1028/2026_ff_lsrs.shp"
ffw_path = os.path.join(raster_dir, "ffws.shp")
huc_shp_path = os.path.join(raster_dir, "huc1028_boundary.shp")

frame_output_root = os.path.join(raster_dir, "individual_frames")
os.makedirs(frame_output_root, exist_ok=True)

def parse_valid_time(val):
    val_str = str(val).strip()
    try:
        return datetime.datetime.strptime(val_str[:12], "%Y%m%d%H%M")
    except Exception:
        return None

# --- Process HUC 1028 Watershed Boundary ---
print("Loading HUC 1028 watershed boundary...")
if os.path.exists(huc_shp_path):
    gdf_huc = gpd.read_file(huc_shp_path, engine="fiona")
    if gdf_huc.crs is None: gdf_huc.crs = "EPSG:4326"  
    gdf_huc = gdf_huc.to_crs("EPSG:3857")
else:
    print(f"[WARNING] Boundary file not found at {huc_shp_path}, creating empty proxy...")
    gdf_huc = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

# --- Process LSR Shapefile ---
print("Loading LSR validation shapefile...")
gdf_lsr = gpd.read_file(lsr_path, engine="fiona")
if gdf_lsr.crs is None: gdf_lsr.crs = "EPSG:5070"  
gdf_lsr = gdf_lsr.to_crs("EPSG:3857")
gdf_lsr['VALID_DT'] = gdf_lsr['VALID'].apply(parse_valid_time)
gdf_lsr = gdf_lsr.dropna(subset=['VALID_DT']).copy()
gdf_lsr['VALID_DAY_STR'] = gdf_lsr['VALID_DT'].dt.strftime("%Y%m%d")

# Note: scale only affects Polygons/LineStrings. Points will remain Points.
gdf_lsr['geometry'] = gdf_lsr.geometry.scale(xfact=5.0, yfact=5.0, origin='centroid')

# --- Process Flash Flood Warning (FFW) Shapefile ---
print("Loading HUC 1028 Flash Flood Warning shapefile...")
gdf_ffw = gpd.read_file(ffw_path, engine="fiona")
if gdf_ffw.crs is None: gdf_ffw.crs = "EPSG:4326"  
gdf_ffw = gdf_ffw.to_crs("EPSG:3857")
gdf_ffw['VALID_DT'] = gdf_ffw['ISSUED'].apply(parse_valid_time)
gdf_ffw = gdf_ffw.dropna(subset=['VALID_DT']).copy()
gdf_ffw['VALID_DAY_STR'] = gdf_ffw['VALID_DT'].dt.strftime("%Y%m%d")

# Target exclusively May 19th and June 5th 
target_run_days = {"20260519", "20260605"}
print(f"Pipeline running filters strictly for active target dates: {target_run_days}")

# =========================================================================
# 2. DEFINE HUC 1028 BOUNDING BOX & FETCH BASEMAP
# =========================================================================
huc_bounds_4326 = [-95.0, 39.0, -91.5, 42.0] 
print("Fetching Esri World Imagery basemap for HUC 1028 bounds...")
xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", "EPSG:3857", *huc_bounds_4326)
bg_img, bg_ext = cx.bounds2img(xmin, ymin, xmax, ymax, zoom=9, source=cx.providers.Esri.WorldImagery)
bg_pil_template = Image.fromarray(bg_img).convert("RGBA")
target_size = bg_pil_template.size  

cx_xmin, cx_xmax, cx_ymin, cx_ymax = bg_ext

# =========================================================================
# 3. HELPER FUNCTION TO TRANSFORM GEOMETRY TO PIXEL COORDINATES (FIXED)
# =========================================================================
def get_pixel_coords(geom, xmin, ymin, xmax, ymax, img_w, img_h):
    if geom.is_empty: return []
    x_scale, y_scale = img_w / (xmax - xmin), img_h / (ymax - ymin)
    coords = []
    
    if geom.geom_type in ['Polygon', 'MultiPolygon']:
        geoms = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
        for g in geoms:
            exterior = list(g.exterior.coords)
            pixel_poly = [(int((x - xmin) * x_scale), int((ymax - y) * y_scale)) for x, y in exterior]
            coords.append(pixel_poly)
    elif geom.geom_type == 'Point':
        coords.append((int((geom.x - xmin) * x_scale), int((ymax - geom.y) * y_scale)))
    elif geom.geom_type == 'MultiPoint':
        for p in geom.geoms:
            coords.append((int((p.x - xmin) * x_scale), int((ymax - p.y) * y_scale)))
            
    return coords

# =========================================================================
# 4. RUN EXPORT TIMELINE WITH COMPACT CORNER LEGEND
# =========================================================================
for model_name in ["Model1_Any_vs_None", "Model2_Considerable_vs_None"]:
    model_folder = os.path.join(frame_output_root, model_name)
    os.makedirs(model_folder, exist_ok=True)
    
    # Establish dynamic model headers for drawing
    if model_name == "Model1_Any_vs_None":
        model_title = "Model 1: Any Impact vs. None"
        model_legend_text = "Any Threat Predicted"
    else:
        model_title = "Model 2: Considerable vs. None"
        model_legend_text = "Considerable Threat Predicted"
        
    print(f"\nExporting static frames to folder: {model_folder}")
    tif_files = sorted(glob.glob(os.path.join(prediction_dir, f"*{model_name}.tif")))
    
    frame_counter = 0
    for f in tif_files:
        fname = os.path.basename(f)
        ts_match = re.search(r'(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})', fname)
        if not ts_match: continue
            
        year, month, day, hour, minute = ts_match.groups()
        file_day_str = f"{year}{month}{day}"
        
        # Enforce strict day bounds execution filter
        if file_day_str not in target_run_days: continue
            
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
        
        # --- Layer 2: Draw Active LSR, FFW, and HUC Polygons ---
        active_lsrs = gdf_lsr[(gdf_lsr['VALID_DT'] >= start_buffer) & (gdf_lsr['VALID_DT'] <= end_buffer)]
        active_ffws = gdf_ffw[(gdf_ffw['VALID_DT'] >= start_buffer) & (gdf_ffw['VALID_DT'] <= end_buffer)]
        
        if not active_lsrs.empty or not active_ffws.empty or not gdf_huc.empty:
            vector_overlay = Image.new("RGBA", target_size, (0, 0, 0, 0))
            vector_draw = ImageDraw.Draw(vector_overlay)
            
            # Draw LSR validations split out by fld_sv_cls profiles (handling both Points and Polygons)
            for _, row in active_lsrs.iterrows():
                geom_type = row.geometry.geom_type
                pixel_tracks = get_pixel_coords(row.geometry, cx_xmin, cx_ymin, cx_xmax, cx_ymax, target_size[0], target_size[1])
                
                # Severe class verification matching logic
                sev_class = str(row['fld_sv_cls']).strip().upper() if 'fld_sv_cls' in row else 'NUISANCE'
                if sev_class in ['MODERATE', 'SEVERE']:
                    fill_color = (255, 128, 0, 130)    # Translucent Orange
                    solid_color = (255, 100, 0, 255)   # Solid Dark Orange
                else:
                    fill_color = (255, 235, 0, 130)    # Translucent Yellow
                    solid_color = (235, 160, 0, 255)   # Solid Dark Yellow
                    
                for track in pixel_tracks:
                    if geom_type in ['Point', 'MultiPoint']:
                        # Draw as circle point marker
                        r = 15  # Radius of point marker
                        vector_draw.ellipse([track[0]-r, track[1]-r, track[0]+r, track[1]+r], fill=solid_color, outline=(0, 0, 0, 255), width=2)
                    elif geom_type in ['Polygon', 'MultiPolygon']:
                        # Draw as polygon region
                        if len(track) < 3: continue
                        vector_draw.polygon(track, fill=fill_color)
                        vector_draw.polygon(track, outline=solid_color, width=2)
            
            # Draw FFW active polygons
            for _, row in active_ffws.iterrows():
                poly_pixel_tracks = get_pixel_coords(row.geometry, cx_xmin, cx_ymin, cx_xmax, cx_ymax, target_size[0], target_size[1])
                for poly_nodes in poly_pixel_tracks:
                    if len(poly_nodes) < 3: continue
                    vector_draw.polygon(poly_nodes, outline=(57, 255, 20, 255), width=3)
            
            # Draw HUC basin outline
            for _, row in gdf_huc.iterrows():
                poly_pixel_tracks = get_pixel_coords(row.geometry, cx_xmin, cx_ymin, cx_xmax, cx_ymax, target_size[0], target_size[1])
                for poly_nodes in poly_pixel_tracks:
                    if len(poly_nodes) < 3: continue
                    vector_draw.polygon(poly_nodes, outline=(0, 100, 255, 255), width=4)
            
            frame_canvas.alpha_composite(vector_overlay)
            
        # --- Layer 3: Draw Timestamp and New Downscaled Corner Legend ---
        draw = ImageDraw.Draw(frame_canvas)
        try: 
            font_ts = ImageFont.truetype("DejaVuSans.ttf", 48)
            font_leg = ImageFont.truetype("DejaVuSans.ttf", 32)
        except IOError: 
            font_ts = ImageFont.load_default()
            font_leg = ImageFont.load_default()
                
        # Top-left metadata timestamp and model banner (Sized taller to support 2-line structure)
        draw.rectangle([10, 10, 780, 145], fill=(0, 0, 0, 220))
        draw.text((25, 20), label_text, fill=(255, 255, 255), font=font_ts)
        draw.text((25, 85), model_title, fill=(255, 200, 0, 255), font=font_leg)
        
        # RESPONSIVE CORNER LEGEND PARAMETERS (Expands to host new multi-tiered LSR keys)
        leg_width, leg_height = 620, 295
        leg_xmin = target_size[0] - leg_width - 15
        leg_ymin = target_size[1] - leg_height - 15
        leg_xmax = target_size[0] - 15
        leg_ymax = target_size[1] - 15
        
        draw.rectangle([leg_xmin, leg_ymin, leg_xmax, leg_ymax], fill=(0, 0, 0, 220))
        
        # Row height alignments within the new compact box
        row_y_starts = [leg_ymin + 20, leg_ymin + 75, leg_ymin + 130, leg_ymin + 185, leg_ymin + 240]
        box_w, box_h = 40, 30
        x_box = leg_xmin + 20
        x_text = leg_xmin + 80
        
        # Row 1: Model Threat Map (Updated dynamically to print the active model name)
        draw.rectangle([x_box, row_y_starts[0], x_box + box_w, row_y_starts[0] + box_h], fill=(255, 0, 0, 255))
        draw.text((x_text, row_y_starts[0] - 2), model_legend_text, fill=(255, 255, 255), font=font_leg)
        
        # Row 2: Nuisance LSRs
        draw.rectangle([x_box, row_y_starts[1], x_box + box_w, row_y_starts[1] + box_h], fill=(255, 235, 0, 255))
        draw.text((x_text, row_y_starts[1] - 2), "Nuisance LSR (+/-3hr)", fill=(255, 255, 255), font=font_leg)
        
        # Row 3: Mod or Severe LSRs
        draw.rectangle([x_box, row_y_starts[2], x_box + box_w, row_y_starts[2] + box_h], fill=(255, 128, 0, 255))
        draw.text((x_text, row_y_starts[2] - 2), "Mod / Severe LSR (+/-3hr)", fill=(255, 255, 255), font=font_leg)
        
        # Row 4: FFW Polygon Boundaries
        draw.rectangle([x_box, row_y_starts[3], x_box + box_w, row_y_starts[3] + box_h], outline=(57, 255, 20, 255), width=4)
        draw.text((x_text, row_y_starts[3] - 2), "Flash Flood Warning (+/-3hr)", fill=(255, 255, 255), font=font_leg)

        # Row 5: HUC Basin Bounds
        draw.rectangle([x_box, row_y_starts[4], x_box + box_w, row_y_starts[4] + box_h], outline=(0, 100, 255, 255), width=4)
        draw.text((x_text, row_y_starts[4] - 2), "HUC 1028 Basin Boundary", fill=(255, 255, 255), font=font_leg)
        
        # Save Frame Asset
        frame_counter += 1
        output_png_name = f"frame_{frame_counter:04d}_{year}{month}{day}-{hour}{minute}.png"
        output_png_path = os.path.join(model_folder, output_png_name)
        
        frame_canvas.save(output_png_path, "PNG")
        print(f"-> Saved individual frame: {output_png_name}", end="\r")

print("\n\nAll static image directory assets updated with a clean, low-profile corner legend layout!")
