"""
main.py
End-to-End Execution Pipeline for Traffic Congestion State Prediction
CSD361 - Introduction to Machine Learning
Authors: Divyesh Sandra, A. Pavan Sri Hari, Anish Reddy, M Thatvik
"""

import sys
import os
import time

# Ensure src is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_raw_trajectories
from src.feature_extraction import build_spatio_temporal_dataset
from src.train import prepare_datasets, train_models
from src.evaluate import evaluate_models, run_ablation_study
from src.cross_intersection import run_cross_intersection_generalization
from src.explainability import run_shap_analysis


def main():
    start_time = time.time()
    print("="*70)
    print(" CSD361 - Traffic Congestion State Prediction Pipeline ")
    print(" Authors: Divyesh Sandra, A. Pavan Sri Hari, Anish Reddy, M Thatvik ")
    print(" Base Paper: TSANet (Pattern Recognition 2024) | EyeonTraffic Dataset ")
    print("="*70)
    
    # Step 1: Trajectory Data Ingestion / Generation
    print("\n[Step 1/6] Loading Trajectory Data...")
    raw_df = load_raw_trajectories("data/eot_raw_trajectories.csv")
    
    # Step 2: Spatio-Temporal Feature Extraction
    print("\n[Step 2/6] Extracting Spatio-Temporal & Interaction Graph Features...")
    feat_df = build_spatio_temporal_dataset(raw_df, "data/eot_features.csv")
    
    # Step 3: Train/Test Partition & Scaling
    print("\n[Step 3/6] Preparing Train/Test Splits & Scaling...")
    data_dict = prepare_datasets(feat_df, test_size=0.20, seed=42)
    
    # Step 4: Model Training & Cross-Validation
    print("\n[Step 4/6] Training Logistic Regression, Random Forest, & XGBoost...")
    trained_models = train_models(data_dict, cv_folds=5, seed=42)
    
    # Step 5: Test Evaluation, Reports, & Confusion Matrices
    print("\n[Step 5/6] Evaluating Held-Out Performance...")
    metrics_df = evaluate_models(trained_models, data_dict, output_dir="results")
    
    # Step 6: Advanced Research Experiments
    print("\n[Step 6/6] Conducting Ablation Study & Generalization Tests...")
    ablation_df = run_ablation_study(feat_df, output_dir="results", seed=42)
    gen_df = run_cross_intersection_generalization(feat_df, output_dir="results", seed=42)
    
    # SHAP Explainability
    best_model = trained_models["XGBoost"]["model"]
    run_shap_analysis(
        best_model,
        data_dict["X_test"],
        data_dict["feature_names"],
        output_dir="results"
    )
    
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS ")
    print("="*70)
    print("\nFinal Test Metrics Summary:")
    print(metrics_df.to_string(index=False))
    print("\nAblation Study Summary:")
    print(ablation_df.to_string(index=False))
    print("\nCross-Intersection Generalization Summary:")
    print(gen_df.to_string(index=False))
    print("\nAll trained models saved to 'models/'")
    print("All experimental figures and tables saved to 'results/'")


if __name__ == "__main__":
    main()
