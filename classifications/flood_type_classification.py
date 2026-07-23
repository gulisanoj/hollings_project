import os
import csv
import pandas as pd
import ollama

# ----------------------------------------------------
# 1. Setup File and Directory Paths
# ----------------------------------------------------
base_dir = "/Projects/flood_threat/HUC0305/lsr_final"
input_file = os.path.join(base_dir, "fho_validation_list.csv")
output_file = input_file
model_name = 'gemma4:e4b'

# Streamlined Prompt with strict Infrastructure Default for Built Environments
PROMPT_SINGLE = """You are an emergency manager analyzing text remarks of flood-related Local Storm Reports to determine the primary cause of flooding.

[CRITICAL INSTRUCTIONS]
1. Classify as "FLUVIAL" if the remark explicitly mentions flooding originating from natural water bodies like streams, rivers, creeks, bayous, or branches overflowing their banks (even if roads are flooded as a result). Also include flooding over low water crossings or bridges here.
2. Classify as "INFRASTRUCTURE" if the remark describes flooding affecting human-built infrastructure (roads, highways, streets, intersections, county roads, farm-to-market roads) WITHOUT mentioning a natural river or creek. This includes clogged storm drains, catch basins, poor urban drainage design, street drainage backups, or roads rendered impassable by surface runoff. Also include non-swift water rescues here.
3. Classify as "INDETERMINATE" ONLY if there is absolutely no mention of any human-built infrastructure (no roads, streets, or buildings) AND no mention of natural water bodies (e.g., "floodwaters in corn field", "standing water in pasture").

[RULE]
You must respond with exactly one word: "FLUVIAL", "INFRASTRUCTURE", or "INDETERMINATE". Do not include any introduction, explanation, punctuation, markdown formatting, or surrounding text.

Your classification:"""

def classify_remark(remark, current_row_num):
    """Sends the remark to the local model and aggressively sanitizes the output."""
    if pd.isna(remark) or str(remark).strip() == "":
        return "INDETERMINATE"
    
    remark_str = str(remark).strip()
    remark_upper = remark_str.upper()
    
    # 1. Python-level override for BRIDGES (Forced FLUVIAL)
    # Placed at the top so it takes absolute precedence over road keyword matching rules
    if "BRIDGE" in remark_upper:
        if current_row_num <= 5:
            print(f"\n🌉 [OVERRIDE Row {current_row_num}] Bridge mention detected -> Forced FLUVIAL")
        return "FLUVIAL"
    
    # 2. Python-level override for Low Water Crossings (Forced FLUVIAL)
    if "LOW WATER" in remark_upper and any(k in remark_upper for k in ["CROSSING", "XING"]):
        return "FLUVIAL"
    
    # 3. Python-level override for non-swift water rescues (Forced INFRASTRUCTURE)
    if "RESCUE" in remark_upper and not any(k in remark_upper for k in ["SWIFT", "CREEK", "RIVER", "STREAM"]):
        return "INFRASTRUCTURE"

    # 4. Python-level override: Catch road/infrastructure mentions that lack natural water context
    # This prevents reports like "CR 250 at FM 485 impassable" from defaulting to INDETERMINATE
    road_keywords = ["CR ", "FM ", "ROAD", "RD ", "HWY", "HIGHWAY", "STREET", "ST ", "AVE", "INTERSECTION", "IMPASSABLE", "BARRICADE"]
    fluvial_keywords = ["CREEK", "RIVER", "STREAM", "BAYOU", "BRANCH", "BROOK", "RUN"]
    
    if any(rk in remark_upper for rk in road_keywords) and not any(fk in remark_upper for fk in fluvial_keywords):
        if current_row_num <= 5:
            print(f"\n⚡ [OVERRIDE Row {current_row_num}] Built infrastructure detected without natural water body context -> Forced INFRASTRUCTURE")
        return "INFRASTRUCTURE"

    full_prompt = f"{PROMPT_SINGLE}\n\nRemark text to analyze:\n\"{remark_str}\""
    
    try:
        response = ollama.generate(
            model=model_name,
            prompt=full_prompt,
            options={
                'temperature': 0.0  # Forces completely deterministic output
            }
        )
        
        # Capture raw response
        raw_response = response['response']
        
        # Aggressive cleaning: strip spaces, newlines, markdown asterisks, quotes, and punctuation
        cleaned_result = raw_response.strip().replace("*", "").replace('"', "").replace("'", "").upper()
        
        # Debug Print: Let's see exactly what the model is thinking for the first few rows
        if current_row_num <= 5:
            print(f"\n🔍 [DEBUG Row {current_row_num}] Raw LLM Output: '{raw_response.strip()}' -> Parsed As: '{cleaned_result}'")

        # Guardrail check using flexible string inclusion
        if "FLUVIAL" in cleaned_result:
            return "FLUVIAL"
        elif "INFRASTRUCTURE" in cleaned_result:
            return "INFRASTRUCTURE"
        elif "INDETERMINATE" in cleaned_result:
            return "INDETERMINATE"
            
        return "INDETERMINATE"
        
    except Exception as e:
        print(f"\n❌ Error communicating with Ollama: {e}")
        return "ERROR"

def main():
    if not os.path.exists(input_file):
        print(f"❌ Error: Input CSV file not found at {input_file}")
        return

    print(f"⏳ Loading LSR data safely from {os.path.basename(input_file)}...")
    
    df = pd.read_csv(
        input_file, 
        on_bad_lines='warn', 
        engine='python', 
        encoding='utf-8'
    )
    
    remark_col = 'REMARK' 
    if remark_col not in df.columns:
        print(f"❌ Error: The input CSV must contain a column named '{remark_col}'.")
        print(f"   Available columns: {list(df.columns)}")
        return

    total_rows = len(df)
    print(f"🌲 Starting flood type classification on the ENTIRE dataset ({total_rows} rows)...")
    
    df['flood_type'] = "INDETERMINATE"
    
    for idx, (index, row) in enumerate(df.iterrows(), 1):
        remark_text = row[remark_col]
        
        if idx > 5:
            print(f"   -> Classifying row {idx}/{total_rows} (Index: {index})...", end="\r")
        
        result = classify_remark(remark_text, idx)
        df.at[index, 'flood_type'] = result
        
    print(f"\n⏳ Overwriting original CSV data at {os.path.basename(output_file)}...")
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8')
    print("🎉 Flood type classification complete. Bridge override processing applied successfully!")

if __name__ == "__main__":
    main()
