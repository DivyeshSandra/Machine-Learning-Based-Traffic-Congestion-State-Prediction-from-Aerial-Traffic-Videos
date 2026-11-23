"""
predict.py
Inference module for Traffic Congestion State Prediction.
Loads trained models (XGBoost, Random Forest, Logistic Regression) and performs
predictions on new traffic observation feature vectors or batch CSV files.
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from src.train import FEATURE_COLUMNS

CLASS_NAMES = ["clump", "neutral", "unclump"]


def load_model(model_name: str = "xgboost"):
    """Loads a pre-trained model and associated scaler."""
    name_map = {
        "xgboost": ("models/xgboost.joblib", False),
        "random_forest": ("models/random_forest.joblib", False),
        "logistic_regression": ("models/logistic_regression.joblib", True)
    }
    if model_name not in name_map:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(name_map.keys())}")
        
    path, is_scaled = name_map[model_name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trained model not found at {path}. Please run main.py first.")
        
    model = joblib.load(path)
    scaler = None
    if is_scaled:
        scaler = joblib.load("models/scaler.joblib")
    return model, scaler, is_scaled


def predict_single_sample(features: dict, model_name: str = "xgboost") -> dict:
    """
    Performs inference on a single dictionary of traffic features.
    
    Returns:
        dict: Predicted state, class probabilities, and status summary.
    """
    model, scaler, is_scaled = load_model(model_name)
    df = pd.DataFrame([features])
    
    # Fill missing features with default 0.0
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[FEATURE_COLUMNS]
    if is_scaled and scaler:
        X = scaler.transform(X)
        
    pred_idx = int(model.predict(X)[0])
    predicted_state = CLASS_NAMES[pred_idx]
    
    probs = {}
    if hasattr(model, "predict_proba"):
        prob_vec = model.predict_proba(X)[0]
        probs = {cls: float(prob_vec[i]) for i, cls in enumerate(CLASS_NAMES)}
        
    return {
        "model_used": model_name,
        "predicted_state": predicted_state,
        "probabilities": probs,
        "confidence": float(np.max(list(probs.values()))) if probs else 1.0
    }


def predict_batch_csv(csv_path: str, model_name: str = "xgboost", output_path: str = "results/predictions.csv") -> pd.DataFrame:
    """Performs inference on a batch CSV file."""
    model, scaler, is_scaled = load_model(model_name)
    df = pd.read_csv(csv_path)
    
    X = df[FEATURE_COLUMNS]
    if is_scaled and scaler:
        X = scaler.transform(X)
        
    preds = model.predict(X)
    df["predicted_target"] = preds
    df["predicted_state"] = [CLASS_NAMES[p] for p in preds]
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        for i, cls in enumerate(CLASS_NAMES):
            df[f"prob_{cls}"] = probs[:, i]
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Predict] Saved batch predictions to {output_path}")
    return df


PRESET_SAMPLES = {
    "clump": {
        "description": "High density queue forming, low speeds, decelerating",
        "features": {
            "vehicle_count": 68,
            "density": 0.68,
            "mean_speed": 1.45,
            "speed_std": 0.72,
            "speed_min": 0.20,
            "speed_max": 2.80,
            "velocity_variance": 0.55,
            "mean_inter_vehicle_dist": 3.12,
            "min_inter_vehicle_dist": 1.10,
            "mean_graph_degree": 8.4,
            "max_graph_degree": 14,
            "interaction_intensity": 0.24,
            "delta_count": 6.0,
            "delta_speed": -0.85,
            "delta_density": 0.06,
            "delta_inter_dist": -0.42,
            "rolling_speed_mean": 1.60,
            "rolling_count_mean": 65.0,
            "acceleration_proxy": -0.85
        }
    },
    "neutral": {
        "description": "Stable steady-state flow, constant speed, normal headway",
        "features": {
            "vehicle_count": 36,
            "density": 0.36,
            "mean_speed": 9.54,
            "speed_std": 1.78,
            "speed_min": 5.22,
            "speed_max": 12.54,
            "velocity_variance": 93.37,
            "mean_inter_vehicle_dist": 4.98,
            "min_inter_vehicle_dist": 1.47,
            "mean_graph_degree": 2.72,
            "max_graph_degree": 5.0,
            "interaction_intensity": 0.21,
            "delta_count": -1.0,
            "delta_speed": 2.32,
            "delta_density": -0.01,
            "delta_inter_dist": -0.04,
            "rolling_speed_mean": 8.26,
            "rolling_count_mean": 37.7,
            "acceleration_proxy": 2.32
        }
    },
    "unclump": {
        "description": "Bottleneck discharge, accelerating away, expanding gaps",
        "features": {
            "vehicle_count": 35,
            "density": 0.35,
            "mean_speed": 8.15,
            "speed_std": 1.44,
            "speed_min": 4.52,
            "speed_max": 12.63,
            "velocity_variance": 66.52,
            "mean_inter_vehicle_dist": 5.63,
            "min_inter_vehicle_dist": 0.87,
            "mean_graph_degree": 2.0,
            "max_graph_degree": 4.0,
            "interaction_intensity": 0.20,
            "delta_count": 2.0,
            "delta_speed": 2.17,
            "delta_density": 0.02,
            "delta_inter_dist": -1.16,
            "rolling_speed_mean": 6.84,
            "rolling_count_mean": 35.14,
            "acceleration_proxy": 2.17
        }
    }
}


def print_prediction_result(name: str, desc: str, res: dict):
    print("\n" + "-"*60)
    print(f"Scenario: {name.upper()} ({desc})")
    print(f"Model: {res['model_used']}")
    print(f"Predicted State : {res['predicted_state'].upper()}")
    print(f"Confidence      : {res['confidence']*100:.2f}%")
    print("Class Probabilities:")
    for cls_name, prob in res["probabilities"].items():
        bar = "█" * int(prob * 25)
        print(f"  - {cls_name:<8}: {prob*100:6.2f}% | {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Traffic Congestion State")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "random_forest", "logistic_regression"])
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV feature file")
    parser.add_argument("--preset", type=str, default="all", choices=["clump", "neutral", "unclump", "all"],
                        help="Pre-configured traffic condition scenario to evaluate")
    args = parser.parse_args()
    
    if args.input:
        predict_batch_csv(args.input, args.model)
    else:
        print("="*60)
        print(f" TRAFFIC CONGESTION INFERENCE ENGINE [{args.model.upper()}]")
        print("="*60)
        
        targets = [args.preset] if args.preset != "all" else ["clump", "neutral", "unclump"]
        for target in targets:
            preset = PRESET_SAMPLES[target]
            res = predict_single_sample(preset["features"], model_name=args.model)
            print_prediction_result(target, preset["description"], res)
