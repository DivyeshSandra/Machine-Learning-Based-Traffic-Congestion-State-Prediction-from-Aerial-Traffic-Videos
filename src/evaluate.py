"""
evaluate.py
Computes comprehensive evaluation metrics, classification reports,
confusion matrices, ROC curves, and ablation studies.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier


CLASS_NAMES = ["clump", "neutral", "unclump"]


def evaluate_models(trained_models: dict, data_dict: dict, output_dir: str = "results") -> pd.DataFrame:
    """
    Evaluates all trained models on the held-out test split and writes logs & plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    y_test = data_dict["y_test"]
    X_test = data_dict["X_test"]
    X_test_scaled = data_dict["X_test_scaled"]
    
    metrics_records = []
    reports_text = []
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, entry) in enumerate(trained_models.items()):
        model = entry["model"]
        is_scaled = entry["is_scaled"]
        X_eval = X_test_scaled if is_scaled else X_test
        
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval) if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        metrics_records.append({
            "Model": name,
            "CV_Macro_F1": entry.get("cv_f1", 0.0),
            "Test_Accuracy": acc,
            "Test_Precision_Macro": prec_macro,
            "Test_Recall_Macro": rec_macro,
            "Test_F1_Macro": f1_macro,
            "Test_F1_Weighted": f1_weighted
        })
        
        # Classification report text
        clf_rep = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
        reports_text.append(f"================ {name} ================\n{clf_rep}\n")
        
        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            ax=axes[idx],
            cbar=False,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES
        )
        axes[idx].set_title(f"{name}\n(Acc: {acc*100:.1f}%, F1: {f1_macro:.3f})", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted State", fontsize=10)
        axes[idx].set_ylabel("True State", fontsize=10)
        
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Saved confusion matrices -> {cm_path}")
    
    # Save text report
    rep_path = os.path.join(output_dir, "classification_reports.txt")
    with open(rep_path, "w") as f:
        f.writelines(reports_text)
    print(f"[Evaluate] Saved classification reports -> {rep_path}")
    
    # Save metrics summary table
    summary_df = pd.DataFrame(metrics_records)
    summary_csv = os.path.join(output_dir, "metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Evaluate] Saved metrics summary -> {summary_csv}")
    
    # Plot ROC curves for XGBoost
    plot_multiclass_roc(trained_models["XGBoost"]["model"], X_test, y_test, output_dir)
    
    return summary_df


def plot_multiclass_roc(model, X_test, y_test, output_dir: str):
    """Generates One-vs-Rest ROC curve for multi-class classification."""
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = model.predict_proba(X_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(8, 6))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, color in zip(range(3), colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f"Class: {CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})"
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Multi-Class ROC Curves (XGBoost Classifier)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Saved ROC curves -> {roc_path}")


def run_ablation_study(df: pd.DataFrame, output_dir: str = "results", seed: int = 42) -> pd.DataFrame:
    """
    Performs ablation study comparing:
    1. Kinematics Only: [vehicle_count, density, mean_speed, speed_std, speed_min, speed_max, velocity_variance]
    2. Kinematics + Graph Proximity: (1) + [mean_inter_vehicle_dist, min_inter_vehicle_dist, mean_graph_degree, max_graph_degree, interaction_intensity]
    3. Full Spatio-Temporal: (2) + [delta_count, delta_speed, delta_density, delta_inter_dist, rolling_speed_mean, rolling_count_mean, acceleration_proxy]
    """
    feature_sets = {
        "A: Kinematics Only": [
            "vehicle_count", "density", "mean_speed", "speed_std",
            "speed_min", "speed_max", "velocity_variance"
        ],
        "B: Kinematics + Graph Interaction": [
            "vehicle_count", "density", "mean_speed", "speed_std",
            "speed_min", "speed_max", "velocity_variance",
            "mean_inter_vehicle_dist", "min_inter_vehicle_dist",
            "mean_graph_degree", "max_graph_degree", "interaction_intensity"
        ],
        "C: Full Spatio-Temporal Model": [
            "vehicle_count", "density", "mean_speed", "speed_std",
            "speed_min", "speed_max", "velocity_variance",
            "mean_inter_vehicle_dist", "min_inter_vehicle_dist",
            "mean_graph_degree", "max_graph_degree", "interaction_intensity",
            "delta_count", "delta_speed", "delta_density", "delta_inter_dist",
            "rolling_speed_mean", "rolling_count_mean", "acceleration_proxy"
        ]
    }
    
    y = df["target"].values
    ablation_records = []
    
    for name, feats in feature_sets.items():
        X = df[feats].values
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        
        acc = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, average="macro", zero_division=0)
        rec = recall_score(y_te, preds, average="macro", zero_division=0)
        f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        
        ablation_records.append({
            "Feature_Configuration": name,
            "Num_Features": len(feats),
            "Accuracy": acc,
            "Macro_Precision": prec,
            "Macro_Recall": rec,
            "Macro_F1": f1
        })
        
    abl_df = pd.DataFrame(ablation_records)
    abl_path = os.path.join(output_dir, "ablation_study.csv")
    abl_df.to_csv(abl_path, index=False)
    print(f"[Evaluate] Saved ablation study -> {abl_path}")
    return abl_df
