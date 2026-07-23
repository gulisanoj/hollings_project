import os
import geopandas as gpd
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# =========================================================================
# 1. PATHS & INITIAL LOADING
# =========================================================================
shp_path = "/Projects/flood_threat/MERIT/HUC1029/final_buffers.shp"
output_all_data_csv = "/Projects/flood_threat/MERIT/HUC1029/corrected_data.csv"
model_dir = "/Projects/flood_threat/MERIT/HUC1029/"

print("Loading shapefile...")
gdf = gpd.read_file(shp_path, engine="fiona")
print(f"-> Total raw rows loaded: {len(gdf)}")

# Clean strings for robust matching, forcing UPPERCASE alignment
gdf["fld_sv_cls_clean"] = gdf["FLD_SV_CLS"].astype(str).str.strip().str.upper()

# Map severities using exact upper-case string configurations
conditions = [
    (gdf["fld_sv_cls_clean"] == "SEVERE") | (gdf["fld_sv_cls_clean"] == "MODERATE"),
    (gdf["fld_sv_cls_clean"] == "NUISANCE"),
    (gdf["fld_sv_cls_clean"] == "NO_IMPACTS") | (gdf["fld_sv_cls_clean"] == "") | (gdf["fld_sv_cls_clean"] == "NONE") | (gdf["FLD_SV_CLS"].isna())
]
choices = ["CONSIDERABLE", "MINIMAL", "NO_IMPACTS"]
gdf["IMPACTS"] = np.select(conditions, choices, default="NO_IMPACTS")

# Feature engineering 
gdf["precip_acceleration"] = gdf["P_HR_0"] - gdf["P_HR_MIN1"]

# Define CLEAN final ML features (No target leakage)
feature_cols = [
    "P_HR_0",
    "P_HR_MIN1",
    "P_HR_MIN2",
    "P_24HR_TOT",
    "P_48HR_TOT",
    "med_hand",
    "mea_imper",
    "med_tcc",
    "max_twi",
    "min_slope",
    "precip_acceleration",
]

# Drop rows with missing values globally across features
gdf_clean = gdf.dropna(subset=feature_cols + ["IMPACTS"]).copy()
print(f"-> Total valid records available after dropping NULLs: {len(gdf_clean)}")

# Export clean data to CSV
gdf_clean[feature_cols + ["IMPACTS"]].to_csv(output_all_data_csv, index=False)
print(f"-> Cleaned dataset exported to {output_all_data_csv}\n")


# =========================================================================
# 2. VISUALIZATION ENGINE
# =========================================================================
def generate_rf_visualizations(rf, feature_cols, labels, model_name, output_dir):
    """
    Generates and saves visual evaluations (Feature Importance & Decision Tree)
    using the optimized dashboard parameters.
    """
    # Create an explicit directory path for this model's plots
    clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").lower()
    plot_dir = os.path.join(output_dir, "plots", clean_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Set plotting theme context
    sns.set_theme(style="white")
    
    # --- PLOT 1: Feature Importance Bar Chart ---
    importances = rf.feature_importances_
    df_imp = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.barplot(x="Importance", y="Feature", data=df_imp, palette="viridis", ax=ax)
    
    # Wrap title string cleanly within the narrow frame layout
    wrapped_title = f"Feature Importance -\n{model_name}"
    ax.set_title(wrapped_title, fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Gini Importance Score")
    ax.set_ylabel("Features")
    
    # Prevent x-axis tick packing using a major spacing locator
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    
    plt.tight_layout()
    feat_imp_path = os.path.join(plot_dir, "feature_importance.png")
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"📊 Feature Importance plot saved to: {feat_imp_path}")
    
    # --- PLOT 2: Visualize a Single Representative Tree Architecture ---
    plt.figure(figsize=(18, 8)) 
    single_tree = rf.estimators_[0]  # Extracts the first decision tree tree from the forest
    
    plot_tree(
        single_tree,
        max_depth=2,  # Hard locked to level 2 logic breaks
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
# 3. PIPELINE: Model 1: Any Impact vs. No Impacts (All Pooled Data)
# =========================================================================
print("=================================================================")
print(" PIPELINE: Model 1: Any Impact vs. No Impacts (All Pooled Data)")
print("=================================================================")

# Create target string labels matching report formatting
y_m1 = np.where(gdf_clean["IMPACTS"] == "NO_IMPACTS", "NO_IMPACTS", "ANY_IMPACT")
X_m1 = gdf_clean[feature_cols]

# 80/20 Train/Test Split
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_m1, y_m1, test_size=0.2, random_state=42, stratify=y_m1
)

print(f"Initial training set split size: {len(X_train1)} rows")
print("Class distribution before SMOTE balancing:")
counts1 = pd.Series(y_train1).value_counts()
print(f"  - NO_IMPACTS: {counts1.get('NO_IMPACTS', 0)}")
print(f"  - ANY_IMPACT: {counts1.get('ANY_IMPACT', 0)}")

print("Balancing training classes using SMOTE...")
smote1 = SMOTE(random_state=42)
X_train1_res, y_train1_res = smote1.fit_resample(X_train1, y_train1)
print(f"-> Resampled balanced training size: {len(X_train1_res)} rows\n")

# RF with OOB tracking enabled
rf_m1 = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42, n_jobs=-1)
rf_m1.fit(X_train1_res, y_train1_res)

print(f"-> Evaluating model performance on remaining 20% test split ({len(X_test1)} rows)")
y_pred1 = rf_m1.predict(X_test1)

print(f"Reliability (OOB Score)       : {rf_m1.oob_score_ * 100:.2f}%")
print(f"Test Split Total Accuracy      : {accuracy_score(y_test1, y_pred1) * 100:.2f}%")
print(f"Combined Macro F1-Score        : {f1_score(y_test1, y_pred1, average='macro') * 100:.2f}%")

print("-" * 65)
print(f"{'RISK CLASS':<17} | {'PRECISION':<9} | {'RECALL (CATCHING)':<19} | {'F1-SCORE':<9}")
print("-" * 65)
rep1 = classification_report(y_test1, y_pred1, output_dict=True)
labels_m1 = ["ANY_IMPACT", "NO_IMPACTS"]
for cls in labels_m1:
    print(f"{cls:<17} | {rep1[cls]['precision']*100:.2f}%  | {rep1[cls]['recall']*100:.2f}%            | {rep1[cls]['f1-score']*100:.2f}%")

print("\nCONFUSION MATRIX")
print(f"{'ACTUAL CLASS':<17} | {'PRED ANY_IMPACT':<15} | {'PRED NO_IMPACTS':<15}")
print("-" * 52)
cm1 = confusion_matrix(y_test1, y_pred1, labels=labels_m1)
print(f"{'ANY_IMPACT':<17} | {cm1[0][0]:<15} | {cm1[0][1]:<15}")
print(f"{'NO_IMPACTS':<17} | {cm1[1][0]:<15} | {cm1[1][1]:<15}")
print("-" * 52)

print("\nFEATURE IMPORTANCE RANKINGS")
print(f"{'FEATURE':<26} | {'IMPORTANCE SCORE'}")
print("-" * 45)
fi1 = pd.Series(rf_m1.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, val in fi1.items():
    print(f"{feat:<26} | {val:.4f}")
print("-" * 45)

# Trigger Model 1 Plots
generate_rf_visualizations(
    rf=rf_m1,
    feature_cols=feature_cols,
    labels=labels_m1,
    model_name="Any Impact vs. No Impacts (Binary)",
    output_dir=model_dir
)


# =========================================================================
# 4. PIPELINE: Model 2: Considerable vs. No Impacts (All Pooled Data)
# =========================================================================
print("\n=================================================================")
print(" PIPELINE: Model 2: Considerable vs. No Impacts (All Pooled Data)")
print("=================================================================")

# Isolate Severe/Moderate vs No Impacts (Drop Minimal)
gdf_m2 = gdf_clean[gdf_clean["IMPACTS"] != "MINIMAL"].copy()
y_m2 = np.where(gdf_m2["IMPACTS"] == "NO_IMPACTS", "NO_IMPACTS", "CONSIDERABLE_IMPACT")
X_m2 = gdf_m2[feature_cols]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_m2, y_m2, test_size=0.2, random_state=42, stratify=y_m2
)

print(f"Initial training set split size: {len(X_train2)} rows")
print("Class distribution before SMOTE balancing:")
counts2 = pd.Series(y_train2).value_counts()
print(f"  - NO_IMPACTS: {counts2.get('NO_IMPACTS', 0)}")
print(f"  - CONSIDERABLE_IMPACT: {counts2.get('CONSIDERABLE_IMPACT', 0)}")

print("Balancing training classes using SMOTE...")
smote2 = SMOTE(random_state=42)
X_train2_res, y_train2_res = smote2.fit_resample(X_train2, y_train2)
print(f"-> Resampled balanced training size: {len(X_train2_res)} rows\n")

rf_m2 = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42, n_jobs=-1)
rf_m2.fit(X_train2_res, y_train2_res)

print(f"-> Evaluating model performance on remaining 20% test split ({len(X_test2)} rows)")
y_pred2 = rf_m2.predict(X_test2)

print(f"Reliability (OOB Score)       : {rf_m2.oob_score_ * 100:.2f}%")
print(f"Test Split Total Accuracy      : {accuracy_score(y_test2, y_pred2) * 100:.2f}%")
print(f"Combined Macro F1-Score        : {f1_score(y_test2, y_pred2, average='macro') * 100:.2f}%")

print("-" * 65)
print(f"{'RISK CLASS':<20} | {'PRECISION':<9} | {'RECALL (CATCHING)':<19} | {'F1-SCORE':<9}")
print("-" * 65)
rep2 = classification_report(y_test2, y_pred2, output_dict=True)
labels_m2 = ["CONSIDERABLE_IMPACT", "NO_IMPACTS"]
for cls in labels_m2:
    print(f"{cls:<20} | {rep2[cls]['precision']*100:.2f}%  | {rep2[cls]['recall']*100:.2f}%            | {rep2[cls]['f1-score']*100:.2f}%")

print("\nCONFUSION MATRIX")
print(f"{'ACTUAL CLASS':<20} | {'PRED CONSIDERABLE':<22} | {'PRED NO_IMPACTS':<15}")
print("-" * 65)
cm2 = confusion_matrix(y_test2, y_pred2, labels=labels_m2)
print(f"{'CONSIDERABLE_IMPACT':<20} | {cm2[0][0]:<22} | {cm2[0][1]:<15}")
print(f"{'NO_IMPACTS':<20} | {cm2[1][0]:<22} | {cm2[1][1]:<15}")
print("-" * 65)

print("\nFEATURE IMPORTANCE RANKINGS")
print(f"{'FEATURE':<26} | {'IMPORTANCE SCORE'}")
print("-" * 45)
fi2 = pd.Series(rf_m2.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, val in fi2.items():
    print(f"{feat:<26} | {val:.4f}")
print("-" * 45)

# Trigger Model 2 Plots (Updated model_name parameter here)
generate_rf_visualizations(
    rf=rf_m2,
    feature_cols=feature_cols,
    labels=labels_m2,
    model_name="Considerable vs. No Impacts (Binary)",
    output_dir=model_dir
)


# =========================================================================
# 5. EXPORTING MODELS FOR SPATIAL INFERENCE & STACKING
# =========================================================================
print("\n=================================================================")
print(" EXPORTING TRAINED MODELS FOR INFERENCE MAPPING")
print("=================================================================")

# Package model with explicit feature sequence order to prevent mismatch on reload
model_1_payload = {"model": rf_m1, "features": feature_cols}
model_2_payload = {"model": rf_m2, "features": feature_cols}

m1_filename = model_dir + "rf_model_1_any_impact.joblib"
m2_filename = model_dir + "rf_model_2_severe_impact.joblib"

joblib.dump(model_1_payload, m1_filename)
joblib.dump(model_2_payload, m2_filename)

print(f"-> Successfully saved Model 1 payload to: {m1_filename}")
print(f"-> Successfully saved Model 2 payload to: {m2_filename}")
print("\nAll tasks completed successfully!")
