"""
train.py
Model training pipeline comparing Logistic Regression, Random Forest, and XGBoost.
Implements data splitting, feature scaling, hyperparameter tuning, and cross-validation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    "vehicle_count",
    "density",
    "mean_speed",
    "speed_std",
    "speed_min",
    "speed_max",
    "velocity_variance",
    "mean_inter_vehicle_dist",
    "min_inter_vehicle_dist",
    "mean_graph_degree",
    "max_graph_degree",
    "interaction_intensity",
    "delta_count",
    "delta_speed",
    "delta_density",
    "delta_inter_dist",
    "rolling_speed_mean",
    "rolling_count_mean",
    "acceleration_proxy"
]


def prepare_datasets(df: pd.DataFrame, test_size: float = 0.20, seed: int = 42):
    """
    Performs stratified train/test split and standard scaling.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df["target"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": FEATURE_COLUMNS,
        "scaler": scaler
    }


def train_models(data_dict: dict, cv_folds: int = 5, seed: int = 42) -> dict:
    """
    Trains Logistic Regression, Random Forest, and XGBoost models,
    evaluating each with Stratified 5-Fold Cross Validation.
    """
    X_train_scaled = data_dict["X_train_scaled"]
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    print("\n" + "="*50)
    print("TRAINING & CROSS-VALIDATION")
    print("="*50)
    
    # 1. Logistic Regression (Multinomial with L2 Regularization)
    lr_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=seed
    )
    lr_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=cv, scoring="f1_macro")
    lr_model.fit(X_train_scaled, y_train)
    joblib.dump(lr_model, "models/logistic_regression.joblib")
    print(f"[Logistic Regression] 5-Fold CV Macro-F1: {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})")
    
    # 2. Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1
    )
    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring="f1_macro")
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "models/random_forest.joblib")
    print(f"[Random Forest]       5-Fold CV Macro-F1: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
    
    # 3. XGBoost Classifier
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1
    )
    xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="f1_macro")
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, "models/xgboost.joblib")
    print(f"[XGBoost]             5-Fold CV Macro-F1: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")
    
    return {
        "Logistic Regression": {"model": lr_model, "is_scaled": True, "cv_f1": float(lr_scores.mean()), "cv_std": float(lr_scores.std())},
        "Random Forest": {"model": rf_model, "is_scaled": False, "cv_f1": float(rf_scores.mean()), "cv_std": float(rf_scores.std())},
        "XGBoost": {"model": xgb_model, "is_scaled": False, "cv_f1": float(xgb_scores.mean()), "cv_std": float(xgb_scores.std())}
    }
