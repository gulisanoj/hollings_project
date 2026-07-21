import os
import geopandas as gpd
import planetary_computer as pc
from pystac_client import Client
import stackstac
import rioxarray

def download_and_merge_dems(shapefile_path, output_raster_path, target_crs=5070):
    """
    Finds 30m Copernicus DEM tiles intersecting the shapefile using Planetary Computer,
    reprojects and merges them virtually into EPSG:5070, and saves the final DEM to disk.
    """
    if not os.path.exists(shapefile_path):
        print(f"Error: Shapefile not found at {shapefile_path}")
        return

    # 1. Load the shapefile
    print(f"Reading shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    
    # We need the bounding box in EPSG:4326 for the STAC API catalog search
    if gdf.crs != "EPSG:4326":
        gdf_4326 = gdf.to_crs("EPSG:4326")
    else:
        gdf_4326 = gdf
    api_bbox = list(gdf_4326.total_bounds)
    print(f"Shapefile bounding box (WGS84) for API search: {api_bbox}")

    # We *also* need the bounding box in EPSG:5070 to define the bounds of the stackstac cube
    if gdf.crs != f"EPSG:{target_crs}":
        gdf_target = gdf.to_crs(f"EPSG:{target_crs}")
    else:
        gdf_target = gdf
    stack_bounds = [float(x) for x in list(gdf_target.total_bounds)]
    print(f"Shapefile bounding box (EPSG:{target_crs}) for raster bounds: {stack_bounds}")

    # 2. Search Planetary Computer STAC API for Copernicus DEM 30m
    print("Searching Planetary Computer STAC API...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=api_bbox
    )
    
    items = search.item_collection()
    num_items = len(items)
    print(f"Found {num_items} intersecting DEM tile(s).")

    if num_items == 0:
        print("No intersecting DEM tiles found. Exiting.")
        return

    # 3. Lazily stack and reproject the items using stackstac
    print(f"Virtually mosaic-ing, reprojecting to EPSG:{target_crs}, and downloading...")
    
    # Setting epsg=5070 forces stackstac to handle reprojection on-the-fly
    da = stackstac.stack(items, assets=["data"], bounds=stack_bounds, epsg=target_crs)

    # Clean up dimensions natively in Xarray
    da = da.squeeze("band", drop=True)
    
    # Flatten the time/tile dimension by taking the first valid (non-NaN) pixel
    merged_da = da.ffill(dim="time").isel(time=-1)

    # 4. Save the final merged raster directly to disk
    output_dir = os.path.dirname(output_raster_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Writing final EPSG:{target_crs} merged DEM to: {output_raster_path}")
    
    # Set proper nodata value for Copernicus DEM
    merged_da = merged_da.rio.write_nodata(-32767, inplace=True)
    
    # This triggers the streaming download & saving process
    merged_da.rio.to_raster(output_raster_path)
    print("Success! Merged 5070 DEM created.")

if __name__ == "__main__":
    SHAPEFILE = "/Projects/flood_threat/HUC1028/huc1028_boundary.shp"
    OUTPUT_DEM = "/Projects/flood_threat/HUC1028/huc1028_dem_30m.tif"
    
    download_and_merge_dems(SHAPEFILE, OUTPUT_DEM)