import os
import pandas as pd
import ollama

# File paths configured for your original CSV file structure
base_dir = "/Projects/flood_threat/HUC0305/forest_features"
input_file = os.path.join(base_dir, "buffered_lsr_with_500_new_samples.csv")
output_file = os.path.join(base_dir, "buffered_lsr_with_500_new_samples.csv")
model_name = 'gemma4:e4b'

# Your original flood type cause prompt
PROMPT_TYPE = """You are an emergency manager working with FEMA and the National Weather Service, tasked to analyze the text remarks of flood-related Local Storm Reports and determine the primary underlying cause/type of the described flooding.

[Instructions & Classes]

1. Classify as "FLUVIAL" if the remark describes flooding originating from natural water bodies:
- Text explicitly mentions streams, rivers, creeks, or bayous overflowing their banks.
- Text mentions river flooding, stream flooding, or main-stem water body rises inundating areas.

2. Classify as "INFRASTRUCTURE" if the remark describes flooding caused by human-built systems or drainage failure:
- Text explicitly mentions clogged storm drains, catch basins, or grates blocked by debris.
- Text mentions poor drainage design, street drainage backups, culvert blockages, or neighborhood storm sewer failures.

3. Classify as "INDETERMINATE" if:
- The text does not contain enough context or details to clearly distinguish between natural water body overflow (Fluvial) and drainage system failures (Infrastructure).
- The report only mentions vague impacts like "water on road" or "flooding reported" without pointing to a specific cause.

[Rule]
You must respond with exactly one of these three words: "FLUVIAL", "INFRASTRUCTURE", or "INDETERMINATE". Do not include any introduction, explanation, punctuation, thoughts, or surrounding text.

The target remark to evaluate is: "{remark_text}"

Your classification:"""

def classify_remark_type(remark):
    """Sends the remark to the local gemma4:e4b model for flood type classification."""
    if pd.isna(remark) or str(remark).strip() == "" or str(remark).lower() == 'nan':
        return "INDETERMINATE"
   
    full_prompt = PROMPT_TYPE.format(remark_text=str(remark).strip())
   
    try:
        response = ollama.generate(
            model=model_name,
            prompt=full_prompt,
            options={
                'temperature': 0.0,  # Forces deterministic output
                'num_predict': 5     # Limits generation tokens to prevent hanging
            }
        )
       
        result = response['response'].strip().upper()
       
        # Guardrail check
        for valid_class in ["FLUVIAL", "INFRASTRUCTURE", "INDETERMINATE"]:
            if valid_class in result:
                return valid_class
               
        return "INDETERMINATE"
       
    except Exception as e:
        print(f"\nError communicating with Ollama: {e}")
        return "ERROR"

def main():
    if not os.path.exists(input_file):
        print(f"Error: Input CSV file not found at {input_file}")
        return

    print(f"Loading CSV data from {input_file}...")
    df = pd.read_csv(input_file)
   
    remark_col = 'REMARK'
    if remark_col not in df.columns:
        print(f"Error: The input CSV must contain a field named '{remark_col}'.")
        return

    total_rows = len(df)
    print(f"Starting flood type classification on the ENTIRE dataset ({total_rows} rows) using {model_name}...")
   
    classifications = []
   
    for idx, (index, row) in enumerate(df.iterrows(), 1):
        remark_text = row[remark_col]
        print(f"Classifying record {idx}/{total_rows}...", end="\r")
       
        result = classify_remark_type(remark_text)
        classifications.append(result)
       
    # Assign strictly to the requested TYPE column field
    df['TYPE'] = classifications
   
    print(f"\nWriting updated data back to original CSV at {output_file}...")
    df.to_csv(output_file, index=False)
   
    print("\n" + "="*50)
    print(" FLOOD TYPE PROCESSING COMPLETE!")
    print("--- CAUSE TYPE SPREAD (TYPE) ---")
    print(df['TYPE'].value_counts())
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
