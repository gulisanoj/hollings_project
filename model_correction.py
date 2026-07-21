import os
import joblib
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Paths
# Updated paths to maintain structural alignment with your automation directory
shp_path = "/Projects/flood_threat/model_automation/training_data/buffered_lsr_with_precip.shp"
output_2025_csv = "/Projects/flood_threat/model_automation/training_data/2025_lsrs.csv"

# Target path for your production inference engine
TARGET_MODEL_PATH = "/Projects/flood_threat/py_scripts/model_automation/training_data/flood_rf_model.joblib"

print("Loading shapefile...")
gdf = gpd.read_file(shp_path, engine="fiona")

# Feature engineering interaction terms & acceleration rates
gdf["rain_x_imp"] = gdf["P_3HR_CUM"] * gdf["imp_med"]
gdf["precip_acceleration"] = gdf["P_HR_0"] - gdf["P_HR_MIN1"]

print("Extracting and combining 2025 test datasets...")

# Standard 2025 records (VALID starts with '2025')
is_standard_2025 = gdf['VALID'].astype(str).str.strip().str.startswith('2025')
gdf_standard_2025 = gdf[is_standard_2025].copy()

# Robust space-insensitive regex matching for empty VALID cells ending in /25
valid_str = gdf['VALID'].astype(str).str.strip()
is_valid_null = gdf['VALID'].isna() | (valid_str == "") | (valid_str == "None") | (valid_str == "nan")
ends_with_25 = gdf['RND_TIME'].astype(str).str.contains(r'25\s*$', regex=True)

gdf_null_2025 = gdf[is_valid_null & ends_with_25].copy()

# Sample exactly up to 100 rows from the null/25 subset
n_rows_to_add = min(100, len(gdf_null_2025))
gdf_null_2025_sample = gdf_null_2025.head(n_rows_to_add)

print(f"-> Standard 2025 records found: {len(gdf_standard_2025)}")
print(f"-> Matching blank-VALID rows ending in '25' found: {len(gdf_null_2025)}")
print(f"   (Adding {n_rows_to_add} of them to the test batch)")

# Combine both subsets into the final 2025 test pool
gdf_2025 = pd.concat([gdf_standard_2025, gdf_null_2025_sample], axis=0)

# Historical training data drops anything that went into our 2025 test pool
gdf_historical = gdf.drop(gdf_2025.index).copy()

# Save the combined 2025 testing data to a clean CSV
print(f"Saving combined 2025 test records to {output_2025_csv}...")
df_2025_out = gdf_2025.drop(columns=['geometry'], errors='ignore')
df_2025_out.to_csv(output_2025_csv, index=False)

# Define Features
feature_cols = [
    "P_HR_0",
    "P_HR_MIN1",
    "P_HR_MIN2",
    "P_3HR_CUM",
    "P_24HR_TOT",
    "P_48HR_TOT",
    "hand_med",
    "imp_med",
    "tcc_med",
    "twi_max",
    "slp_min", 
    "rain_x_imp",
    "precip_acceleration"
]

# Drop missing values for training and evaluation datasets
df_hist_clean = gdf_historical.dropna(subset=feature_cols + ["IMPACTS"]).copy()
df_2025_clean = gdf_2025.dropna(subset=feature_cols + ["IMPACTS"]).copy()


# 2. Dual Evaluation Reporting Engine
def run_rf_and_evaluate_2025(X_hist, y_hist, X_2025, y_2025, model_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X_hist, y_hist, test_size=0.2, random_state=42, stratify=y_hist
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        oob_score=True,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    
    # --- EVALUATION 1: Standard Historical Test Split ---
    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n" + "=" * 65)
    print(f" MODEL: {model_name}")
    print("=" * 65)
    print(f"Reliability (OOB Score)       : {rf.oob_score_:.2%}")
    print(f"Historical Test Split Accuracy : {report['accuracy']:.2%}")
    print(f"Historical Combined F1-Score   : {report['macro avg']['f1-score']:.2%}")
    print("-" * 65)

    print(f"{'HISTORICAL CLASS':<16} | {'PRECISION':<10} | {'RECALL (CATCHING)':<18} | {'F1-SCORE':<10}")
    print("-" * 65)
    labels = sorted(list(y_hist.unique()))
    for label in labels:
        metrics = report[str(label)]
        print(f"{label:<16} | {metrics['precision']:<10.2%} | {metrics['recall']:<18.2%} | {metrics['f1-score']:<10.2%}")
        
    # --- EVALUATION 2: Unseen 2025 Test Dataset (Includes the added rows) ---
    print("\n" + "." * 65)
    print(" RUNNING MODEL ON UNSEEN 2025 DATASET (Combined)")
    print("." * 65)
    
    if X_2025.empty:
        print("No valid 2025 records available matching this model's target criteria.")
    else:
        y_pred_2025 = rf.predict(X_2025)
        report_2025 = classification_report(y_2025, y_pred_2025, output_dict=True)
        
        print(f"2025 Dataset Total Accuracy   : {report_2025['accuracy']:.2%}")
        print(f"2025 Combined F1-Score        : {report_2025['macro avg']['f1-score']:.2%}")
        print("-" * 65)
        print(f"{'2025 RISK CLASS':<16} | {'PRECISION':<10} | {'RECALL (CATCHING)':<18} | {'F1-SCORE':<10}")
        print("-" * 65)
        for label in labels:
            if str(label) in report_2025:
                metrics_2025 = report_2025[str(label)]
                print(f"{label:<16} | {metrics_2025['precision']:<10.2%} | {metrics_2025['recall']:<18.2%} | {metrics_2025['f1-score']:<10.2%}")
            else:
                print(f"{label:<16} | {'N/A':<10} | {'N/A':<18} | {'N/A':<10}")
                
        print("\n2025 CONFUSION MATRIX")
        cm_2025 = confusion_matrix(y_2025, y_pred_2025, labels=labels)
        header_line = f"{'ACTUAL 2025':<16} | " + " | ".join([f"PRED {l}" for l in labels])
        print(header_line)
        print("-" * len(header_line))
        for i, label in enumerate(labels):
            row_values = " | ".join([f"{val:<10}" for val in cm_2025[i]])
            print(f"{label:<16} | {row_values}")
        print("-" * len(header_line))

    return rf


# -------------------------------------------------------------------------
# RUN SIMPLIFIED DUAL MODELS
# -------------------------------------------------------------------------

m1_map = {
    "CONSIDERABLE": "ANY_IMPACT",
    "MINIMAL": "ANY_IMPACT",
    "NO_IMPACTS": "NO_IMPACTS",
}

# --- MODEL 1: Any Impact vs. No Impacts (Binary) ---
X1_hist = df_hist_clean[feature_cols]
y1_hist = df_hist_clean["IMPACTS"].map(m1_map)

X1_2025 = df_2025_clean[feature_cols]
y1_2025 = df_2025_clean["IMPACTS"].map(m1_map)

# Captured the model instance from Model 1 execution
model_1_object = run_rf_and_evaluate_2025(X1_hist, y1_hist, X1_2025, y1_2025, "Any Impact vs. No Impacts (Binary)")


# --- MODEL 2: Considerable vs. No Impacts (Binary, ignoring Minimal) ---
df_hist_m2 = df_hist_clean[df_hist_clean["IMPACTS"] != "MINIMAL"]
X2_hist = df_hist_m2[feature_cols]
y2_hist = df_hist_m2["IMPACTS"]

df_2025_m2 = df_2025_clean[df_2025_clean["IMPACTS"] != "MINIMAL"]
X2_2025 = df_2025_m2[feature_cols]
y2_2025 = df_2025_m2["IMPACTS"]

run_rf_and_evaluate_2025(X2_hist, y2_hist, X2_2025, y2_2025, "Considerable vs. No Impacts (Binary)")


# -------------------------------------------------------------------------
# PRODUCTION DIRECTORY INTERCEPT EXPORT
# -------------------------------------------------------------------------
print("\n" + "=" * 65)
print(" EXPORTING MODEL 1 FOR INFERENCE SYSTEM")
print("=" * 65)

# Bypasses environment directory locks by building out the missing workspace trees
parent_dir = os.path.dirname(TARGET_MODEL_PATH)
os.makedirs(parent_dir, exist_ok=True)

print(f"Writing production file out to: {TARGET_MODEL_PATH}")
joblib.dump(model_1_object, TARGET_MODEL_PATH)
print("✨ Complete. Model 1 is successfully built and saved to disk.")