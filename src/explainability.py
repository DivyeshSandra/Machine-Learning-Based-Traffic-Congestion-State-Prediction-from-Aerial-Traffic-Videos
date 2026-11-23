"""
explainability.py
Interprets model decisions using SHAP (SHapley Additive exPlanations)
and feature importance rankings.
"""

import os
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_shap_analysis(model, X_sample: pd.DataFrame, feature_names: list, output_dir: str = "results"):
    """
    Computes TreeSHAP explanations for the tree-based model (e.g. XGBoost or Random Forest)
    and generates global feature importance and multi-class summary plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*50)
    print("RUNNING SHAP EXPLAINABILITY ANALYSIS")
    print("="*50)
    
    # Subsample for efficient SHAP calculation
    sample_size = min(300, len(X_sample))
    X_sub = X_sample.sample(n=sample_size, random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)
    
    # 1. Feature Importance Bar Plot (MDI / Tree importance)
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        sorted_feats = [feature_names[i] for i in indices]
        sorted_scores = importances[indices]
        
        plt.barh(range(len(sorted_feats)), sorted_scores[::-1], align="center", color="#2980b9")
        plt.yticks(range(len(sorted_feats)), sorted_feats[::-1], fontsize=9)
        plt.xlabel("Gini / Gain Importance", fontsize=11)
        plt.title("Tree Global Feature Importance", fontsize=13, fontweight="bold")
        plt.tight_layout()
        
        feat_imp_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(feat_imp_path, dpi=300)
        plt.close()
        print(f"[SHAP] Saved global feature importance -> {feat_imp_path}")
    
    # 2. SHAP Summary Plot
    # For multi-class, shap_values is a list of arrays (one per class) or a 3D array (samples, features, classes)
    try:
        plt.figure(figsize=(11, 7))
        # If list of classes
        if isinstance(shap_values, list):
            # Compute mean absolute SHAP across all classes
            mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            shap.summary_plot(
                shap_values,
                X_sub,
                feature_names=feature_names,
                class_names=["clump", "neutral", "unclump"],
                show=False
            )
        elif len(shap_values.shape) == 3:
            # 3D array (N, d, C)
            shap.summary_plot(
                [shap_values[:, :, i] for i in range(shap_values.shape[2])],
                X_sub,
                feature_names=feature_names,
                class_names=["clump", "neutral", "unclump"],
                show=False
            )
        else:
            shap.summary_plot(shap_values, X_sub, feature_names=feature_names, show=False)
            
        plt.title("SHAP Multi-Class Feature Impact Summary", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        shap_path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(shap_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Saved SHAP summary plot -> {shap_path}")
    except Exception as e:
        print(f"[SHAP] Notice during summary plot generation: {e}")
        # Fallback to bar plot of SHAP importances
        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            vals = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            vals = np.abs(shap_values).mean(axis=0)
        indices = np.argsort(vals)[::-1]
        plt.barh(range(len(feature_names)), vals[indices][::-1], color="#8e44ad")
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices][::-1], fontsize=9)
        plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=11)
        plt.title("SHAP Global Feature Attribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        shap_path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(shap_path, dpi=300)
        plt.close()
        print(f"[SHAP] Saved fallback SHAP attribution plot -> {shap_path}")
