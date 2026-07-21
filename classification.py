import os
import pandas as pd
import ollama

# File paths and configuration defined by the planning document
input_file = "/Projects/flood_threat/HUC0305/june17_lsrs.csv"
output_file = "/Projects/flood_threat/HUC0305/june17_lsrs.csv"
model_name = "gemma4:e4b"

# Master prompt placed globally outside the functions
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

The target remark to evaluate is: "{remark_text}"

Your classification:"""

def classify_remark(remark):
    """Sends the remark to the local gemma4:e4b model for severity classification."""
    if pd.isna(remark) or str(remark).strip() == "" or str(remark).lower() == "nan":
        return "UNKNOWN"
    
    full_prompt = PROMPT_SINGLE.format(remark_text=str(remark).strip())
    
    try:
        response = ollama.generate(
            model=model_name,
            prompt=full_prompt,
            options={
                'temperature': 0.0  # Forces deterministic output
            }
        )
        # Clean response and strip any trailing artifacts or newlines
        result = response['response'].strip().upper()
        
        # Guardrail check in case the LLM appends trailing text or punctuation
        for valid_class in ["NUISANCE", "MODERATE", "SEVERE"]:
            if valid_class in result:
                return valid_class
                
        return "UNKNOWN"
        
    except Exception as e:
        print(f"\nError communicating with Ollama: {e}")
        return "ERROR"

def main():
    if not os.path.exists(input_file):
        print(f"Error: Input CSV file not found at {input_file}")
        return

    print(f"Loading data from {input_file}...")
    # Read tabular data natively using pandas
    df = pd.read_csv(input_file)
    
    remark_col = 'REMARK' 
    if remark_col not in df.columns:
        print(f"Error: The input CSV must contain an attribute field named '{remark_col}'.")
        print(f"Available fields: {list(df.columns)}")
        return

    total_rows = len(df)
    print(f"Starting classification on the ENTIRE dataset ({total_rows} rows) using {model_name}...")
    
    classifications = []
    
    # Run pipeline sequentially across all features in the tabular layer
    for idx, (index, row) in enumerate(df.iterrows(), 1):
        remark_text = row[remark_col]
        print(f"Classifying record {idx}/{total_rows} (Index: {index})...", end="\r")
        
        result = classify_remark(remark_text)
        classifications.append(result)
        
    # Inject the new string array directly as an attribute field
    df['fld_sv_cls'] = classifications
    
    print(f"\nWriting updated data layers back to {output_file}...")
    
    # Ensure directory exists before dumping output data
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write directly back to CSV format
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print(" PROCESSING COMPLETE!")
    print("--- SEVERITY SPREAD SUMMARY (fld_sv_cls) ---")
    print(df['fld_sv_cls'].value_counts())
    print("="*50 + "\n")

if __name__ == "__main__":
    main()