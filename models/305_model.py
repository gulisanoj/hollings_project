import os
import joblib
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# =========================================================================
# 1. SETUP CONFIGURATIONS & PATHS
# =========================================================================
shp_path = "/Projects/flood_threat/model_automation/training_data/buffered_lsr_with_precip.shp"
output_2025_csv = "/Projects/flood_threat/model_automation/training_data/2025_lsrs.csv"
model_dir = "/Projects/flood_threat/model_automation/trained_models/"

os.makedirs(model_dir, exist_ok=True)

print("Loading shapefile...")
gdf = gpd.read_file(shp_path, engine="fiona")

# Feature engineering interaction terms & acceleration rates
gdf["rain_x_imp"] = gdf["P_3HR_CUM"] * gdf["imp_med"]
gdf["precip_acceleration"] = gdf["P_HR_0"] - gdf["P_HR_MIN1"]

print("Extracting and combining 2025 datasets...")

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
print(f"   (Adding {n_rows_to_add} of them to the master pool)")

# Combine both subsets into the final 2025 pool
gdf_2025 = pd.concat([gdf_standard_2025, gdf_null_2025_sample], axis=0)

# Save the combined 2025 records to a clean CSV for separate tracking
print(f"Saving combined 2025 records to {output_2025_csv}...")
df_2025_out = gdf_2025.drop(columns=['geometry'], errors='ignore')
df_2025_out.to_csv(output_2025_csv, index=False)

# Keep all data (Historical + 2025) intact for the combined master workflow
gdf_master_pool = gdf.copy() 

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

# Drop missing values across the feature/target grid
df_master_clean = gdf_master_pool.dropna(subset=feature_cols + ["IMPACTS"]).copy()


# =========================================================================
# 2. VISUALIZATION ENGINE (WITH WRAPPED TITLE & SPACED X-TICKS)
# =========================================================================
def generate_rf_visualizations(rf, y_test, y_pred, feature_cols, model_name, output_dir):
    """
    Generates and saves visual evaluations for a trained Random Forest model.
    """
    # Create an explicit directory path for this model's plots
    clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plot_dir = os.path.join(output_dir, "plots", clean_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # --- PLOT 1: Feature Importance Bar Chart ---
    importances = rf.feature_importances_
    df_imp = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.barplot(x="Importance", y="Feature", data=df_imp, palette="viridis", ax=ax)
    
    # Wrap title strings cleanly so they fit within a narrow 6-inch frame width
    wrapped_title = f"Feature Importance -\n{model_name}"
    ax.set_title(wrapped_title, fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Gini Importance Score")
    ax.set_ylabel("Features")
    
    # Space out x-axis ticks so they clear visually at regular intervals of 0.05
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    
    plt.tight_layout()
    feat_imp_path = os.path.join(plot_dir, "feature_importance.png")
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"📊 Feature Importance plot saved to: {feat_imp_path}")
    
    # --- PLOT 2: Confusion Matrix Heatmap ---
    labels = sorted(list(y_test.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels, cbar=False,
                annot_kws={"size": 12, "weight": "bold"})
    plt.title(f"Confusion Matrix Heatmap - {model_name}", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Target", fontsize=10)
    plt.ylabel("Actual Target", fontsize=10)
    plt.tight_layout()
    
    cm_path = os.path.join(plot_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"📊 Confusion Matrix heatmap saved to: {cm_path}")
    
    # --- PLOT 3: Visualize a Single Representative Tree Architecture ---
    plt.figure(figsize=(18, 8)) 
    single_tree = rf.estimators_[0]  # Extracts the first decision tree from the forest
    
    plot_tree(
        single_tree,
        max_depth=2,  # Level 2 cuts
        feature_names=feature_cols,
        class_names=[str(c) for c in labels],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f"Sub-Tree Architecture (Max Depth = 2) - {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    tree_path = os.path.join(plot_dir, "decision_tree_sample.png")
    plt.savefig(tree_path, dpi=300)
    plt.close()
    print(f"📊 Sample Decision Tree structure saved to: {tree_path}")


# =========================================================================
# 3. TRAINING & EVALUATION ENGINE
# =========================================================================
def run_rf_and_evaluate(X_data, y_hist, model_name):
    # Splits the unified data matrix (All Historical + 2025 rows combined)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_hist, test_size=0.2, random_state=42, stratify=y_hist
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        oob_score=True,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    
    # --- STANDARDIZED EVALUATION HOLD_OUT ---
    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n" + "=" * 65)
    print(f" MODEL: {model_name}")
    print("=" * 65)
    print(f"Reliability (OOB Score)       : {rf.oob_score_:.2%}")
    print(f"Blended Test Split Accuracy   : {report['accuracy']:.2%}")
    print(f"Blended Macro F1-Score        : {report['macro avg']['f1-score']:.2%}")
    print("-" * 65)

    print(f"{'TEST SPLIT RISK CLASS':<16} | {'PRECISION':<10} | {'RECALL (CATCHING)':<18} | {'F1-SCORE':<10}")
    print("-" * 65)
    labels = sorted(list(y_hist.unique()))
    for label in labels:
        metrics = report[str(label)]
        print(f"{label:<16} | {metrics['precision']:<10.2%} | {metrics['recall']:<18.2%} | {metrics['f1-score']:<10.2%}")
        
    print("\nBLENDED TEST SPLIT CONFUSION MATRIX")
    cm_test = confusion_matrix(y_test, y_pred, labels=labels)
    header_line = f"{'ACTUAL TARGET':<16} | " + " | ".join([f"PRED {l}" for l in labels])
    print(header_line)
    print("-" * len(header_line))
    for i, label in enumerate(labels):
        row_values = " | ".join([f"{val:<10}" for val in cm_test[i]])
        print(f"{label:<16} | {row_values}")
    print("-" * len(header_line))

    # --- TRIGGER ENGINE VISUALIZATIONS ---
    generate_rf_visualizations(
        rf=rf, 
        y_test=y_test, 
        y_pred=y_pred, 
        feature_cols=feature_cols, 
        model_name=model_name,
        output_dir=model_dir
    )

    return rf


# =========================================================================
# 4. RUN PIPELINE AND SERIALIZE MODELS
# =========================================================================

m1_map = {
    "CONSIDERABLE": "ANY_IMPACT",
    "MINIMAL": "ANY_IMPACT",
    "NO_IMPACTS": "NO_IMPACTS",
}

# --- MODEL 1: Any Impact vs. No Impacts (Binary) ---
X1_master = df_master_clean[feature_cols]
y1_master = df_master_clean["IMPACTS"].map(m1_map)

# Train Model 1
rf_model_1 = run_rf_and_evaluate(X1_master, y1_master, "Any Impact vs. No Impacts (Binary)")

# Serialize and dump Model 1 to disk
model_1_filename = os.path.join(model_dir, "rf_any_impact_v1.pkl")
joblib.dump(rf_model_1, model_1_filename)
print(f"💾 Model 1 saved successfully to: {model_1_filename}\n")


# --- MODEL 2: Considerable vs. No Impacts (Binary, ignoring Minimal) ---
df_master_m2 = df_master_clean[df_master_clean["IMPACTS"] != "MINIMAL"]
X2_master = df_master_m2[feature_cols]
y2_master = df_master_m2["IMPACTS"]

# Train Model 2
rf_model_2 = run_rf_and_evaluate(X2_master, y2_master, "Considerable vs. No Impacts (Binary)")

# Serialize and dump Model 2 to disk
model_2_filename = os.path.join(model_dir, "rf_considerable_v1.pkl")
joblib.dump(rf_model_2, model_2_filename)
print(f"💾 Model 2 saved successfully to: {model_2_filename}\n")

print("🚀 All pipeline training runs completed and models successfully saved to disk.")
