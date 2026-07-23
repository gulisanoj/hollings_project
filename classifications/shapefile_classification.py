import os
import geopandas as gpd  # Replaced pandas with geopandas for spatial support
import ollama

# ----------------------------------------------------
# 1. Setup File and Directory Paths
# ----------------------------------------------------
base_dir = "/Projects/flood_threat/HUC0304"
input_file = os.path.join(base_dir, "2025_lsrs.shp")           # Updated to .shp
output_file = os.path.join(base_dir, "2025_lsrs_classed.shp")   # Updated to .shp
model_name = 'gemma4:e4b'

# Original Severity Prompt
PROMPT_SINGLE = """You are an emergency manager working with FEMA and the National Weather Service, tasked to analyze the text remarks of flood-related Local Storm Reports and determine the absolute maximum severity class of the described flooding. 

[CRITICAL EVALUATION ORDER]
You must evaluate the remark starting with Rule 1 (SEVERE). If any criteria in Rule 1 are met, classify as SEVERE immediately and ignore all other minor impacts described. If Rule 1 is not met, check Rule 2, then Rule 3.

[Instructions & Classes]

1. Classify as "SEVERE" if the remark describes ANY of the following:
- Widespread ground-floor or structural inundation across an entire neighborhood, community, or town.
- Large infrastructure flooded (e.g., schools, water plants, commercial districts).
- Closures of Interstates or US Highways.
- Buildings or structures swept away, collapsed, or severely structurally destroyed.
- Dam, levee, or flood mitigation system failures.
- Mandatory evacuations or community-wide displacements.
- Any rescue operation (singular or plural), including swift water rescues, boat extractions, or aerial rescues.
*(CRITICAL EXCLUSION: Do NOT classify localized or multiple basement floodings as SEVERE unless accompanied by rescue operations or structural collapse).*

2. Classify as "MODERATE" if the remark does not meet SEVERE criteria, but describes ANY of the following:
- Floodwaters entering a single building/home, or several/multiple homes (including basement flooding, crawlspace flooding, or localized indoor flooding).
- Vehicles submerged, stalled, trapped, or swept away in water.
- Structural damage to local roads or bridges (e.g., washed-out roads, collapsed culverts).
- ACTUAL CLOSURES of State Highways (must explicitly state the highway is closed, blocked, impassable, or barricaded).
- Rockslides, mudslides, or debris flows blocking transit.
*(CRITICAL EXCLUSION: Do NOT classify remarks describing standing water, minor flooding, or "water on road" on State Highways as MODERATE unless the road is explicitly reported as closed or impassable).*

3. Classify as "NUISANCE" if the remark ONLY describes minor, non-damaging impacts:
- Rivers, creeks, or streams overflowing their banks into open areas or floodplains.
- Minor street or low-lying road flooding without structural damage to the road or bridges.
- Water on State Highways, Interstates, or US Highways that does NOT result in a closure (e.g., "water over road," "ponding on highway," "slow traffic due to water").
- Crop, farmland, yard, or parking lot flooding.
- Closures of minor local roads, city streets, avenues, or boulevards (NOT state highways, US highways, or interstates).

[Rule]
You must respond with exactly one of these three words: "NUISANCE", "MODERATE", or "SEVERE". Do not include any introduction, explanation, punctuation, thoughts, or surrounding text.

Your classification:"""

def classify_remark(remark):
    """Sends the remark to the local gemma4:e4b model for severity classification."""
    if gpd.pd.isna(remark) or str(remark).strip() == "":
        return "INDETERMINATE"
    
    full_prompt = f"{PROMPT_SINGLE}\n\nRemark text to analyze:\n\"{remark}\""
    
    try:
        response = ollama.generate(
            model=model_name,
            prompt=full_prompt,
            options={
                'temperature': 0.0  # Forces completely deterministic output
            }
        )
        
        # Clean response and strip trailing artifacts or newlines
        result = response['response'].strip().upper()
        
        # Guardrail check to cleanly isolate the keyword out of any conversational trailing text
        for valid_class in ["NUISANCE", "MODERATE", "SEVERE"]:
            if valid_class in result:
                return valid_class
                
        return "INDETERMINATE"
        
    except Exception as e:
        print(f"\n❌ Error communicating with Ollama: {e}")
        return "ERROR"

def main():
    if not os.path.exists(input_file):
        print(f"❌ Error: Input Shapefile not found at {input_file}")
        return

    # Ensure output directory folder structure exists baseline
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"⏳ Loading the ENTIRE spatial dataset cleanly from {os.path.basename(input_file)}...")
    
    # Read the spatial layer using geopandas
    gdf = gpd.read_file(input_file)
    
    remark_col = 'REMARK' 
    if remark_col not in gdf.columns:
        print(f"❌ Error: The input Shapefile must contain a column named '{remark_col}'.")
        print(f"    Available columns: {list(gdf.columns)}")
        return

    total_rows = len(gdf)
    print(f"🌲 Starting full dataset classification pipeline ({total_rows} total rows)...")
    
    # Pre-allocate classification column if it doesn't already exist
    # NOTE: Shapefile column tracking requires string pre-allocation
    if 'fld_sv_cls' not in gdf.columns:
        gdf['fld_sv_cls'] = "INDETERMINATE"
    
    # Run pipeline sequentially across all rows
    for idx, (index, row) in enumerate(gdf.iterrows(), 1):
        # Skip row processing if a clean prediction was already cached
        if gdf.at[index, 'fld_sv_cls'] in ["NUISANCE", "MODERATE", "SEVERE"]:
            continue
            
        remark_text = row[remark_col]
        print(f"    -> Classifying row {idx}/{total_rows} (Index: {index})...", end="\r")
        
        result = classify_remark(remark_text)
        gdf.at[index, 'fld_sv_cls'] = result
        
        # SAFEGUARD: Progress checkpoint save to protect processing time
        if idx % 100 == 0:
            gdf.to_file(output_file)

    print(f"\n⏳ Writing final full spatial shapefile to {os.path.basename(output_file)}...")
    gdf.to_file(output_file)
    print("🎉 Comprehensive vector dataset severity classification complete cleanly!")

if __name__ == "__main__":
    main()
