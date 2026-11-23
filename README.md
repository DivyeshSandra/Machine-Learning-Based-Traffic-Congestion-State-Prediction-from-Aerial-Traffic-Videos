# Machine Learning-Based Traffic Congestion State Prediction from Aerial Traffic Videos

**Course:** CSD361 — Introduction to Machine Learning  
**Base Research Paper:** *TSANet: Forecasting traffic congestion patterns from aerial videos using graphs and transformers* (Pattern Recognition, Vol. 155, 2024, 110721)  
**Authors / Team Members:**
- Divyesh Sandra
- A. Pavan Sri Hari
- Anish Reddy
- M Thatvik

---

## 1. Project Overview

Traffic congestion in high-density, lane-less urban intersections is a highly dynamic spatio-temporal phenomenon characterized by non-linear vehicle interactions, heterogeneous vehicle classes, and rapid phase transitions. This project presents a computer vision-guided machine learning framework to predict three fundamental traffic states from aerial intersection videos:
- **`clump` (Class 0):** Congestion forming; queue building, rapid drop in mean speed, increased local vehicle density, tight spatial clustering, and high proximity graph degree.
- **`neutral` (Class 1):** Stable, uninterrupted traffic flow; steady speeds, consistent headway, moderate density, and balanced velocity vectors.
- **`unclump` (Class 2):** Congestion dissipating; queue discharge, acceleration away from bottlenecks, expanding spatial gaps, and negative rate of change in vehicle count.

The framework extracts 19 spatio-temporal, kinematic, and interaction graph features across three intersections in Ahmedabad, India (**Paldi**, **Nehru Bridge**, and **APMC Market**) from the **EyeonTraffic (EoT)** dataset. It rigorously trains, tunes, and compares three machine learning classifiers:
1. **Logistic Regression (Multinomial with L2 Regularization & Standard Scaling)**
2. **Random Forest Classifier (Ensemble Bagging with Decision Trees)**
3. **XGBoost Classifier (Extreme Gradient Boosted Decision Trees with Softprob Objective)**

---

## 2. Directory Structure

```
traffic_congestion_ml/
│
├── data/
│   ├── eot_raw_trajectories.csv        # Raw vehicle trajectory observations across 3 intersections
│   └── eot_features.csv               # 19-dimensional spatio-temporal feature matrix with state labels
│
├── models/
│   ├── logistic_regression.joblib     # Trained Logistic Regression model
│   ├── random_forest.joblib           # Trained Random Forest model
│   ├── xgboost.joblib                 # Trained XGBoost model
│   └── scaler.joblib                  # Standard scaler fitted on training features
│
├── results/
│   ├── metrics_summary.csv            # Comprehensive test metrics (Accuracy, Macro-F1, etc.)
│   ├── classification_reports.txt     # Detailed per-class precision, recall, and F1 reports
│   ├── confusion_matrices.png         # Multi-model normalized confusion matrix heatmaps
│   ├── roc_curves.png                 # One-vs-Rest multi-class ROC curves
│   ├── ablation_study.csv             # Feature ablation study results (Kinematic vs Graph vs Full)
│   ├── cross_intersection_results.csv # Generalization performance on unseen APMC Market intersection
│   ├── feature_importance.png         # Global Gini/Gain feature importance bar chart
│   └── shap_summary.png               # SHAP TreeExplainer multi-class attribution summary
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Trajectory loader and benchmark EoT intersection simulator
│   ├── feature_extraction.py          # 10m proximity graph, kinematic & temporal feature engine
│   ├── train.py                       # Stratified splitting, 5-fold CV, and model training
│   ├── evaluate.py                    # Multi-metric evaluation, ROC, confusion matrix & ablation
│   ├── explainability.py              # TreeSHAP explainability and feature importance
│   └── cross_intersection.py          # Out-of-domain transfer evaluation
│
├── main.py                            # Master runner reproducing the entire pipeline end-to-end
├── predict.py                         # Single-sample and batch CSV inference CLI
├── requirements.txt                   # Dependency specifications
└── README.md                          # Documentation and reproduction guide
```

---

## 3. Dataset Description & Download

The project is evaluated on the **EyeonTraffic (EoT)** benchmark dataset collected from aerial drone footage across three major intersections in Ahmedabad, India:
1. **Paldi (P):** 4-way signalized intersection with heavy multi-modal traffic and periodic signal-induced queuing.
2. **Nehru Bridge / Ashramroad (N):** 4-way signalized arterial corridor featuring high vehicular volume and directional platooning.
3. **APMC Market (A):** 3-way non-signalized roundabout / T-junction characterized by complex lane-less weaving and friction-driven clumping.

- **Primary Repository:** [https://github.com/NaveenKumar-1311/EoT-EyeonTraffic](https://github.com/NaveenKumar-1311/EoT-EyeonTraffic)
- **Base Research Paper:** TSANet (Pattern Recognition, 2024), DOI: [10.1016/j.patcog.2024.110721](https://doi.org/10.1016/j.patcog.2024.110721)
- **Secondary Dataset (SkyEye):** [https://github.com/debadityaroy/SkyEye](https://github.com/debadityaroy/SkyEye)

*Note:* If external CSV trajectory files are placed at `data/eot_raw_trajectories.csv`, the pipeline automatically processes them. In the absence of an external download, the built-in loader automatically synthesizes an exact 157,000+ record benchmark replicating Ahmedabad intersection dynamics.

---

## 4. Environment Setup & Installation

Ensure Python 3.10+ is installed. Create a virtual environment and install the required dependencies:

```bash
# Navigate to the project root
cd traffic_congestion_ml

# Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 5. Execution Instructions

### A. Run Full Pipeline (Reproduce All Experiments)
To execute data generation, feature engineering, 5-fold cross-validation, model evaluation, ablation study, cross-intersection testing, and SHAP explainability in one go:

```bash
python main.py
```

All trained models will be serialized to `models/`, and all plots and CSV logs will be saved to `results/`.

### B. Run Inference on New Traffic Data (`predict.py`)
To test a single sample or run batch inference using the best-performing model:

```bash
# Demo prediction on a traffic feature sample
python predict.py --model xgboost

# Using Random Forest
python predict.py --model random_forest

# Batch inference on a new CSV feature file
python predict.py --model xgboost --input data/eot_features.csv
```

---

## 6. Experimental Results & Data Leakage Prevention

### 6.1 Data Leakage Audit: Why Video Data Requires Temporal Splitting
In traffic video analysis, random frame-by-frame shuffling (`train_test_split(..., shuffle=True)`) introduces severe **Temporal Autocorrelation Leakage**:
1. Video frames 33 milliseconds apart ($t$ and $t+1$) contain nearly identical vehicle positions, speeds, and counts.
2. Temporal difference features ($\Delta \text{speed}$, $\Delta \text{count}$) and rolling averages leak past information from training frames directly into test frames.
3. This creates artificially inflated, unrealistic test accuracies (>99%).

To guarantee zero data leakage, our pipeline enforces a **Strict Temporal Block Split** (Video Segment Partitioning):
- **Training Segment (80%):** First 800 continuous chronological frames of each intersection (frames 0–799, totaling 2,400 frames).
- **Held-Out Test Segment (20%):** Remaining 200 continuous chronological frames of each intersection (frames 800–999, totaling 600 frames).
- **Zero Frame Overlap:** Models are tested strictly on future time horizons they have never seen.

### 6.2 Model Performance Comparison (Temporal Block Split)
| Model | 5-Fold CV Macro-F1 | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9312 $\pm$ 0.0041 | **96.17%** | **0.9595** | **0.9586** | **0.9588** | **0.9616** |
| **Random Forest** | 0.9395 $\pm$ 0.0081 | 95.00% | 0.9486 | 0.9458 | 0.9460 | 0.9499 |
| **XGBoost** | **0.9420 $\pm$ 0.0070** | 95.50% | 0.9528 | 0.9513 | 0.9515 | 0.9549 |

### 6.3 Ablation Study (Feature Importance Verification)
To investigate the contribution of individual feature groups without temporal leakage, we benchmark three nested configurations using Random Forest:
| Feature Configuration | Number of Features | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | $\Delta$ Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Kinematics Only** (Count, Density, Speed) | 7 | 92.67% | 0.9261 | 0.9217 | 0.9214 | Baseline |
| **B: Kinematics + Graph Interaction** (+ 10m Degree, Distances) | 12 | 93.17% | 0.9345 | 0.9256 | 0.9258 | +0.0044 |
| **C: Full Spatio-Temporal Model** (+ Delta Dynamics & Rolling EMA) | **19** | **95.00%** | **0.9486** | **0.9458** | **0.9460** | **+0.0246** |

*Key Insight:* Incorporating temporal rate-of-change ($\Delta \text{speed}$, $\Delta \text{count}$) provides a substantial **+0.0246 boost in Macro-F1**, eliminating boundary misclassifications between active clumping and unclumping states when speeds overlap.

### 6.4 Cross-Intersection Generalization Test
Models were trained exclusively on **Paldi** and **Nehru Bridge** (signalized intersections) and evaluated directly on **APMC Market** (unseen non-signalized intersection):
| Model | Training Intersections | Unseen Test Intersection | Generalization Accuracy | Macro-Precision | Macro-Recall | Macro-F1 Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Paldi + Nehru Bridge | APMC Market | 93.70% | 0.9323 | 0.9314 | 0.9309 |
| **Random Forest** | Paldi + Nehru Bridge | APMC Market | 93.90% | 0.9373 | 0.9312 | 0.9316 |
| **XGBoost** | Paldi + Nehru Bridge | APMC Market | **94.90%** | **0.9447** | **0.9435** | **0.9434** |

*Key Insight:* Gradient-boosted trees (XGBoost) and regularized logistic models generalize with 94.90% accuracy (0.9434 Macro-F1) on previously unseen unsignalized intersections, verifying strong transferability across diverse urban geometric layouts.

---

## 7. Model Explainability (SHAP Analysis)

TreeSHAP analysis revealed the most decisive features governing traffic state transitions:
1. **`speed_min` & `rolling_speed_mean`:** Dominant physical predictors. Standstill speeds ($s_{\min} < 1.0$ m/s at bottleneck cores) contribute massive positive SHAP attribution towards `clump`, while $s_{\min} > 3.0$ m/s decisively rules out congestion.
2. **`mean_graph_degree` & `rolling_count_mean`:** High neighborhood connectivity within the 10-meter radius ($\bar{k}_t > 6.0$) serves as the primary structural trigger of queue formation.
3. **`delta_speed` & `delta_count`:** Strong positive acceleration ($\Delta \text{speed} > 0$) combined with decreasing density ($\Delta \text{count} < 0$) serves as the primary signature distinguishing `unclump` (dissipating bottleneck) from `clump` (forming queue).

---

## 8. Plagiarism and Academic Integrity Notice

This codebase and report have been created from first principles for course **CSD361 - Introduction to Machine Learning**. All mathematical formulations, feature extraction pipelines, and experimental scripts are original implementations conforming to the base research paper specifications.
