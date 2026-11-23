"""
feature_extraction.py
Extracts spatio-temporal, kinematic, and graph proximity features from aerial vehicle trajectories.
Constructs the structured ML dataset aligned with the TSANet / EyeonTraffic formulation.
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def extract_frame_features(frame_df: pd.DataFrame, proximity_threshold: float = 10.0) -> dict:
    """
    Computes spatial, kinematic, and interaction graph features for a single video frame.
    
    Parameters:
        frame_df: DataFrame containing vehicle detections in one frame.
        proximity_threshold: Distance in meters defining an edge in the traffic proximity graph (TSANet standard = 10m).
    """
    n_vehicles = len(frame_df)
    if n_vehicles == 0:
        return {}
    
    speeds = frame_df["speed"].values
    coords = frame_df[["x", "y"]].values
    vx = frame_df["vx"].values
    vy = frame_df["vy"].values
    
    # 1. Kinematic & Density Features
    mean_speed = float(np.mean(speeds))
    speed_std = float(np.std(speeds)) if n_vehicles > 1 else 0.0
    speed_min = float(np.min(speeds))
    speed_max = float(np.max(speeds))
    
    # Area considered is standard 100m x 100m intersection canvas (10,000 m^2)
    density = float(n_vehicles / 100.0)  # vehicles per 100m^2
    
    # Directional conflict / velocity variance
    vx_std = float(np.std(vx)) if n_vehicles > 1 else 0.0
    vy_std = float(np.std(vy)) if n_vehicles > 1 else 0.0
    velocity_variance = float(vx_std**2 + vy_std**2)
    
    # 2. Graph & Proximity Interaction Features
    if n_vehicles > 1:
        dist_matrix = cdist(coords, coords)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Nearest neighbor distances
        nearest_dists = np.min(dist_matrix, axis=1)
        mean_inter_vehicle_dist = float(np.mean(nearest_dists))
        min_inter_vehicle_dist = float(np.min(nearest_dists))
        
        # Proximity Graph Adjacency (TSANet 10-meter neighborhood)
        adj_10m = (dist_matrix <= proximity_threshold).astype(int)
        degrees = np.sum(adj_10m, axis=1)
        mean_graph_degree = float(np.mean(degrees))
        max_graph_degree = float(np.max(degrees))
        
        # Congestion Intensity Index (Inverse distance interaction sum)
        clipping_eps = 1.0
        interaction_intensity = float(np.mean(1.0 / (nearest_dists + clipping_eps)))
    else:
        mean_inter_vehicle_dist = 50.0
        min_inter_vehicle_dist = 50.0
        mean_graph_degree = 0.0
        max_graph_degree = 0.0
        interaction_intensity = 0.0
        
    return {
        "vehicle_count": n_vehicles,
        "density": density,
        "mean_speed": mean_speed,
        "speed_std": speed_std,
        "speed_min": speed_min,
        "speed_max": speed_max,
        "velocity_variance": velocity_variance,
        "mean_inter_vehicle_dist": mean_inter_vehicle_dist,
        "min_inter_vehicle_dist": min_inter_vehicle_dist,
        "mean_graph_degree": mean_graph_degree,
        "max_graph_degree": max_graph_degree,
        "interaction_intensity": interaction_intensity
    }


def build_spatio_temporal_dataset(
    raw_df: pd.DataFrame,
    output_path: str = "data/eot_features.csv"
) -> pd.DataFrame:
    """
    Iterates over intersections and frames to extract spatial features, then derives
    temporal dynamics (delta differences and rolling averages).
    """
    records = []
    
    for (intersection, frame_id), group in raw_df.groupby(["intersection", "frame_id"]):
        feat = extract_frame_features(group)
        if not feat:
            continue
        feat["intersection"] = intersection
        feat["frame_id"] = frame_id
        # Ground truth state is majority or representative label for that frame
        feat["traffic_state"] = group["ground_truth_state"].iloc[0]
        records.append(feat)
        
    feat_df = pd.DataFrame(records)
    feat_df.sort_values(by=["intersection", "frame_id"], inplace=True)
    
    # 3. Temporal Dynamics & Differencing (per intersection)
    grouped = feat_df.groupby("intersection")
    
    # First-order rate of change (Delta features)
    feat_df["delta_count"] = grouped["vehicle_count"].diff().fillna(0.0)
    feat_df["delta_speed"] = grouped["mean_speed"].diff().fillna(0.0)
    feat_df["delta_density"] = grouped["density"].diff().fillna(0.0)
    feat_df["delta_inter_dist"] = grouped["mean_inter_vehicle_dist"].diff().fillna(0.0)
    
    # Rolling Temporal Moving Averages (EMA 3-frame window)
    feat_df["rolling_speed_mean"] = grouped["mean_speed"].transform(lambda x: x.ewm(span=3).mean())
    feat_df["rolling_count_mean"] = grouped["vehicle_count"].transform(lambda x: x.ewm(span=3).mean())
    feat_df["acceleration_proxy"] = feat_df["delta_speed"]
    
    # Map state labels to standard integers: clump=0, neutral=1, unclump=2
    label_map = {"clump": 0, "neutral": 1, "unclump": 2}
    feat_df["target"] = feat_df["traffic_state"].map(label_map)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feat_df.to_csv(output_path, index=False)
    print(f"[FeatureExtraction] Built feature matrix {feat_df.shape} -> {output_path}")
    print(f"[FeatureExtraction] Class distribution:\n{feat_df['traffic_state'].value_counts()}")
    return feat_df
