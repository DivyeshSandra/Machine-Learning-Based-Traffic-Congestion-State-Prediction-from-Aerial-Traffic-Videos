"""
cross_intersection.py
Evaluates cross-intersection generalization.
Trains ML models on Paldi & Nehru Bridge intersections,
and tests directly on the unseen APMC Market intersection.
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.train import FEATURE_COLUMNS


def run_cross_intersection_generalization(
    df: pd.DataFrame,
    train_intersections: list = ["Paldi", "Nehru_Bridge"],
    test_intersection: str = "APMC_Market",
    output_dir: str = "results",
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulates real-world transfer: tests models on an intersection never seen during training.
    """
    print("\n" + "="*50)
    print("CROSS-INTERSECTION GENERALIZATION TEST")
    print(f"Training on: {train_intersections}")
    print(f"Testing on:  {test_intersection} (Unseen Domain)")
    print("="*50)
    
    train_mask = df["intersection"].isin(train_intersections)
    test_mask = df["intersection"] == test_intersection
    
    train_data = df[train_mask]
    test_data = df[test_mask]
    
    X_train = train_data[FEATURE_COLUMNS].copy()
    y_train = train_data["target"].values
    
    X_test = test_data[FEATURE_COLUMNS].copy()
    y_test = test_data["target"].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=seed), True),
        "Random Forest": (RandomForestClassifier(n_estimators=150, max_depth=12, random_state=seed, n_jobs=-1), False),
        "XGBoost": (XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=seed, n_jobs=-1), False)
    }
    
    results = []
    
    for name, (clf, is_scaled) in models.items():
        X_tr = X_train_scaled if is_scaled else X_train
        X_te = X_test_scaled if is_scaled else X_test
        
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        
        results.append({
            "Model": name,
            "Train_Domains": "+".join(train_intersections),
            "Test_Domain": test_intersection,
            "Generalization_Accuracy": acc,
            "Macro_Precision": prec,
            "Macro_Recall": rec,
            "Macro_F1": f1
        })
        print(f"[{name:19s}] APMC Unseen Test -> Acc: {acc*100:.2f}%, Macro-F1: {f1:.4f}")
        
    res_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    res_path = os.path.join(output_dir, "cross_intersection_results.csv")
    res_df.to_csv(res_path, index=False)
    print(f"[Generalization] Saved cross-intersection metrics -> {res_path}")
    return res_df
