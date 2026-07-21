import os
import glob
import re
import datetime
import time
import numpy as np
import pandas as pd  
import rasterio
from rasterio.warp import transform_bounds
from rasterio.vrt import WarpedVRT
import geopandas as gpd
import contextily as cx
from PIL import Image, ImageDraw, ImageFont

# =========================================================================
# 1. PATHS & INITIAL LAYER LOAD
# =========================================================================
raster_dir = "/Projects/flood_threat/model_automation/flood_tifs"
prediction_dir = os.path.join(raster_dir, "predictions")
lsr_path = "/Projects/flood_threat/model_automation/training_data/buffered_lsr_with_precip.shp"
ffw_path = os.path.join(raster_dir, "flash_flood_warnings/huc0305_ffw.shp")
huc_shp_path = "/Projects/flood_threat/HUC0305/forest_features/intermediate/huc0305_boundary.shp"

frame_output_root = os.path.join(raster_dir, "final_individual_frames")
os.makedirs(frame_output_root, exist_ok=True)

def parse_valid_time(val):
    val_str = str(val).strip()
    try:
        return datetime.datetime.strptime(val_str[:12], "%Y%m%d%H%M")
    except Exception:
        return None

# --- Process HUC 0305 Watershed Boundary ---
print("Loading HUC 0305 watershed boundary...")
gdf_huc = gpd.read_file(huc_shp_path, engine="fiona")
if gdf_huc.crs is None: gdf_huc.crs = "EPSG:4326"  
gdf_huc = gdf_huc.to_crs("EPSG:3857")

# --- Process LSR Shapefile ---
print("Loading LSR validation shapefile...")
gdf_lsr = gpd.read_file(lsr_path, engine="fiona")
if gdf_lsr.crs is None: gdf_lsr.crs = "EPSG:5070"  
gdf_lsr = gdf_lsr.to_crs("EPSG:3857")
gdf_lsr['VALID_DT'] = gdf_lsr['VALID'].apply(parse_valid_time)
gdf_lsr = gdf_lsr.dropna(subset=['VALID_DT']).copy()
gdf_lsr['VALID_DAY_STR'] = gdf_lsr['VALID_DT'].dt.strftime("%Y%m%d")
gdf_lsr['geometry'] = gdf_lsr.geometry.scale(xfact=5.0, yfact=5.0, origin='centroid')

# --- Process Flash Flood Warning (FFW) Shapefile ---
print("Loading Flash Flood Warning shapefile...")
gdf_ffw = gpd.read_file(ffw_path, engine="fiona")
if gdf_ffw.crs is None: gdf_ffw.crs = "EPSG:4326"  
gdf_ffw = gdf_ffw.to_crs("EPSG:3857")
gdf_ffw['VALID_DT'] = gdf_ffw['ISSUED'].apply(parse_valid_time)
gdf_ffw = gdf_ffw.dropna(subset=['VALID_DT']).copy()
gdf_ffw['VALID_DAY_STR'] = gdf_ffw['VALID_DT'].dt.strftime("%Y%m%d")

# Explicit timeline strings filter targeting system bounds
target_timestamps = {
    "20250513-0600", "20250614-0300", "20250822-2100", 
    "20250823-1800", "20250816-2100", "20250805-1500", 
    "20250702-0000", "20250714-2100"
}
print(f"Targeting execution windows strictly matching: {target_timestamps}")

# =========================================================================
# 2. DEFINE HUC 0305 BOUNDING BOX & FETCH BASEMAP
# =========================================================================
huc_bounds_4326 = [-83.5, 32.0, -78.0, 36.5] 
print("📡 Fetching Esri World Imagery basemap for HUC 0305 bounds...")
xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", "EPSG:3857", *huc_bounds_4326)
bg_img, bg_ext = cx.bounds2img(xmin, ymin, xmax, ymax, zoom=9, source=cx.providers.Esri.WorldImagery)
bg_pil_template = Image.fromarray(bg_img).convert("RGBA")
target_size = bg_pil_template.size  

cx_xmin, cx_xmax, cx_ymin, cx_ymax = bg_ext

# =========================================================================
# 3. HELPER FUNCTION TO TRANSFORM GEOMETRY TO PIXEL COORDINATES
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
# 4. RUN EXPORT TIMELINE WITH SEVERITY-CLASSIFIED OVERLAYS
# =========================================================================
for model_name in ["Model1_Any_vs_None", "Model2_Considerable_vs_None"]:
    model_folder = os.path.join(frame_output_root, model_name)
    os.makedirs(model_folder, exist_ok=True)
    
    if model_name == "Model1_Any_vs_None":
        model_title_str = "Model 1: Any Impact vs. None"
        model_legend_str = "Any Threat Predicted"
    else:
        model_title_str = "Model 2: Considerable vs. None"
        model_legend_str = "Considerable Threat Predicted"
        
    print(f"\nExporting static frames to folder: {model_folder}")
    tif_files = sorted(glob.glob(os.path.join(prediction_dir, f"*{model_name}.tif")))
    
    frame_counter = 0
    for f in tif_files:
        fname = os.path.basename(f)
        ts_match = re.search(r'(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})', fname)
        if not ts_match: continue
            
        year, month, day, hour, minute = ts_match.groups()
        file_timestamp_str = f"{year}{month}{day}-{hour}{minute}"
        
        # Enforce strict date targeting gate filter
        if file_timestamp_str not in target_timestamps: continue
            
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
            
            # Draw LSR elements split out cleanly by fld_sv_cls profiles
            for _, row in active_lsrs.iterrows():
                geom_type = row.geometry.geom_type
                pixel_tracks = get_pixel_coords(row.geometry, cx_xmin, cx_ymin, cx_xmax, cx_ymax, target_size[0], target_size[1])
                
                # Severity check string matching sequence
                sev_class = str(row['fld_sv_cls']).strip().upper() if 'fld_sv_cls' in row else 'NUISANCE'
                if sev_class in ['MODERATE', 'SEVERE', 'CONSIDERABLE']:
                    fill_color = (255, 128, 0, 255)    # Opaque Solid Orange
                    solid_color = (255, 100, 0, 255)   # Solid Dark Orange
                else:
                    fill_color = (255, 235, 0, 255)    # Opaque Solid Yellow
                    solid_color = (235, 160, 0, 255)   # Solid Dark Yellow
                    
                for track in pixel_tracks:
                    if geom_type in ['Point', 'MultiPoint']:
                        r = 20  
                        vector_draw.ellipse([track[0]-r, track[1]-r, track[0]+r, track[1]+r], fill=solid_color, outline=(0, 0, 0, 255), width=3)
                    else:
                        if len(track) < 3: continue
                        vector_draw.polygon(track, fill=fill_color)
                        vector_draw.polygon(track, outline=solid_color, width=2)
            
            # Render FFW elements
            for _, row in active_ffws.iterrows():
                poly_pixel_tracks = get_pixel_coords(row.geometry, cx_xmin, cx_ymin, cx_xmax, cx_ymax, target_size[0], target_size[1])
                for poly_nodes in poly_pixel_tracks:
                    if len(poly_nodes) < 3: continue
                    vector_draw.polygon(poly_nodes, outline=(57, 255, 20, 255), width=4)
            
            # Render HUC 0305 Watershed outline
            for _, row in gdf_huc.iterrows():
                poly_pixel_tracks = get_pixel_coords(row.geometry, cx_xmin, cx_ymin, cx_xmax, cx_ymax, target_size[0], target_size[1])
                for poly_nodes in poly_pixel_tracks:
                    if len(poly_nodes) < 3: continue
                    vector_draw.polygon(poly_nodes, outline=(0, 100, 255, 255), width=5)
            
            frame_canvas.alpha_composite(vector_overlay)
            
        # --- Layer 3: Draw Expanded Labels, Legends, and Graphics ---
        draw = ImageDraw.Draw(frame_canvas)
        
        header_font_size = int(target_size[1] * 0.035)  
        legend_font_size = int(target_size[1] * 0.024)
        
        # FIXED: Harmonized variable fallback configurations to securely capture name definitions
        try: 
            font_title = ImageFont.truetype("DejaVuSans.ttf", header_font_size)
            font_legend = ImageFont.truetype("DejaVuSans.ttf", legend_font_size)
        except IOError: 
            font_title = ImageFont.load_default()
            font_legend = ImageFont.load_default()
                
        # Dynamic width box generation for top-left metadata block to completely prevent clipping text string bounds
        time_text = f"Time: {label_text}"
        system_text = f"System: {model_title_str}"
        
        bbox_time = draw.textbbox((0, 0), time_text, font=font_title)
        bbox_sys = draw.textbbox((0, 0), system_text, font=font_title)
        max_header_w = max(bbox_time[2] - bbox_time[0], bbox_sys[2] - bbox_sys[0])
        
        draw.rectangle([10, 10, 10 + max_header_w + 50, 175], fill=(0, 0, 0, 220))
        draw.text((30, 20), time_text, fill=(255, 255, 255), font=font_title)
        draw.text((30, 95), system_text, fill=(255, 190, 0, 255), font=font_title)
        
        # Sized up legend layout width panel dynamically using bbox calculation to keep text securely bounded
        longest_legend_string = "Considerable/Severe LSR (+/-3hr)"
        bbox_leg = draw.textbbox((0, 0), longest_legend_string, font=font_legend)
        max_legend_w = bbox_leg[2] - bbox_leg[0]
        
        leg_box_w = 140 + max_legend_w
        leg_box_h = int(legend_font_size * 8.2)
        
        box_x = 5
        box_y = target_size[1] - leg_box_h - 5
        
        draw.rectangle([box_x, box_y, box_x + leg_box_w, target_size[1] - 5], fill=(0, 0, 0, 240), outline=(50, 50, 50, 255), width=3)
        
        # Item 1: Model Threat
        y_offset = box_y + int(legend_font_size * 0.5)
        draw.rectangle([box_x + 20, y_offset, box_x + 70, y_offset + int(legend_font_size * 0.7)], fill=(255, 0, 0, 255))
        draw.text((box_x + 100, y_offset - 4), model_legend_str, fill=(255, 255, 255), font=font_legend)
        
        # Item 2: HUC Boundary
        y_offset += int(legend_font_size * 1.5)
        draw.rectangle([box_x + 20, y_offset + int(legend_font_size * 0.3), box_x + 70, y_offset + int(legend_font_size * 0.5)], fill=(0, 100, 255, 255))
        draw.text((box_x + 100, y_offset - 4), "HUC Watershed Boundary", fill=(255, 255, 255), font=font_legend)
        
        # Item 3: Nuisance LSRs
        y_offset += int(legend_font_size * 1.5)
        dot_center_x, dot_center_y = box_x + 45, y_offset + int(legend_font_size * 0.4)
        rad = int(legend_font_size * 0.35)
        draw.ellipse([dot_center_x-rad, dot_center_y-rad, dot_center_x+rad, dot_center_y+rad], fill=(255, 235, 0, 255), outline=(0, 0, 0, 255), width=2)
        draw.text((box_x + 100, y_offset - 4), "Nuisance LSR (+/-3hr)", fill=(255, 255, 255), font=font_legend)
        
        # Item 4: Moderate/Severe LSRs
        y_offset += int(legend_font_size * 1.5)
        dot_center_x, dot_center_y = box_x + 45, y_offset + int(legend_font_size * 0.4)
        draw.ellipse([dot_center_x-rad, dot_center_y-rad, dot_center_x+rad, dot_center_y+rad], fill=(255, 128, 0, 255), outline=(0, 0, 0, 255), width=2)
        draw.text((box_x + 100, y_offset - 4), "Considerable/Severe LSR (+/-3hr)", fill=(255, 255, 255), font=font_legend)
        
        # Item 5: Flash Flood Warning
        y_offset += int(legend_font_size * 1.5)
        draw.rectangle([box_x + 20, y_offset, box_x + 70, y_offset + int(legend_font_size * 0.7)], outline=(57, 255, 20, 255), width=4)
        draw.text((box_x + 100, y_offset - 4), "Flash Flood Warning (+/-3hr)", fill=(255, 255, 255), font=font_legend)
        
        # Save and Overwrite
        frame_counter += 1
        output_png_name = f"frame_{frame_counter:04d}_{year}{month}{day}-{hour}{minute}.png"
        output_png_path = os.path.join(model_folder, output_png_name)
        
        frame_canvas.save(output_png_path, "PNG")
        print(f"-> Saved individual frame: {output_png_name}", end="\r")

print("\n\nAll static image directory assets updated with a clean background legend layout!")