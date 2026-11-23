"""
data_loader.py
Dataset ingestion and synthetic trajectory generator for the EyeonTraffic (EoT) benchmark.
Represents aerial video vehicle trajectories across three Ahmedabad intersections:
1. Paldi (4-way signalized)
2. Nehru Bridge (4-way signalized arterial)
3. APMC Market (3-way non-signalized)
"""

import os
import numpy as np
import pandas as pd


def generate_eot_benchmark_dataset(
    output_path: str = "data/eot_raw_trajectories.csv",
    seed: int = 42,
    total_frames_per_intersection: int = 1000
) -> pd.DataFrame:
    """
    Generates realistic trajectory data aligned with the EyeonTraffic (EoT) dataset
    and TSANet paper specifications for lane-less traffic environments.
    """
    np.random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    intersections = [
        {"name": "Paldi", "type": "signalized_4way", "cycle_length": 60, "base_count": 45},
        {"name": "Nehru_Bridge", "type": "signalized_4way", "cycle_length": 70, "base_count": 50},
        {"name": "APMC_Market", "type": "unsignalized_3way", "cycle_length": 45, "base_count": 40}
    ]
    
    vehicle_types = ["two_wheeler", "auto_rickshaw", "car", "bus_truck"]
    vehicle_type_probs = [0.50, 0.25, 0.20, 0.05]
    
    all_records = []
    global_vehicle_id = 1000
    
    for inter in intersections:
        name = inter["name"]
        cycle = inter["cycle_length"]
        base_cnt = inter["base_count"]
        
        # State cycling simulating realistic urban intersection flow
        # In signalized: Red cycle -> Clump (forming), Green onset -> Unclump (clearing), Mid green -> Neutral
        # In unsignalized: Bottleneck blockage -> Clump, Discharge -> Unclump, Clear -> Neutral
        
        for frame in range(total_frames_per_intersection):
            phase = (frame % cycle) / cycle
            
            if phase < 0.38:
                # Congestion forming / Queuing (Clump)
                state = "clump"
                active_count = int(np.clip(base_cnt * (1.2 + 0.6 * np.sin(phase * np.pi / 0.38)), 30, 85))
                mean_speed = np.random.uniform(0.8, 3.2)  # m/s (low speed)
                speed_std = 0.8
                inter_vehicle_dist_mean = np.random.uniform(2.5, 4.5)  # tight packing
            elif phase < 0.68:
                # Dissipation / Discharge (Unclump)
                state = "unclump"
                active_count = int(np.clip(base_cnt * (1.4 - 0.7 * (phase - 0.38) / 0.30), 20, 60))
                mean_speed = np.random.uniform(5.5, 11.5)  # accelerating
                speed_std = 2.2
                inter_vehicle_dist_mean = np.random.uniform(5.5, 9.5)
            else:
                # Normal steady flow (Neutral)
                state = "neutral"
                active_count = int(np.clip(base_cnt * (0.8 + 0.2 * np.sin(frame * 0.05)), 18, 45))
                mean_speed = np.random.uniform(7.0, 13.0)  # free flow speed
                speed_std = 1.4
                inter_vehicle_dist_mean = np.random.uniform(8.5, 15.0)
                
            # Generate vehicles for this frame
            for _ in range(active_count):
                global_vehicle_id += 1
                v_type = np.random.choice(vehicle_types, p=vehicle_type_probs)
                
                # Spatial clustering based on state
                if state == "clump":
                    # Concentrated near center junction (50, 50)
                    center_x, center_y = 50.0, 50.0
                    cluster_spread = 14.0
                    x = np.clip(np.random.normal(center_x, cluster_spread), 5.0, 95.0)
                    y = np.clip(np.random.normal(center_y, cluster_spread), 5.0, 95.0)
                    heading = np.random.uniform(0, 2 * np.pi)
                    inst_speed = max(0.2, np.random.normal(mean_speed, speed_std))
                elif state == "unclump":
                    # Dispersing away from center
                    dir_angle = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2]) + np.random.normal(0, 0.25)
                    radial_dist = np.random.uniform(15.0, 48.0)
                    x = np.clip(50.0 + radial_dist * np.cos(dir_angle), 5.0, 95.0)
                    y = np.clip(50.0 + radial_dist * np.sin(dir_angle), 5.0, 95.0)
                    heading = dir_angle
                    inst_speed = max(1.5, np.random.normal(mean_speed, speed_std))
                else:
                    # Uniform steady passage through corridors
                    corridor = np.random.choice(["EW", "NS"])
                    if corridor == "EW":
                        x = np.random.uniform(5.0, 95.0)
                        y = np.random.normal(50.0, 7.0)
                        heading = 0.0 if np.random.rand() > 0.5 else np.pi
                    else:
                        x = np.random.normal(50.0, 7.0)
                        y = np.random.uniform(5.0, 95.0)
                        heading = np.pi/2 if np.random.rand() > 0.5 else 3*np.pi/2
                    inst_speed = max(3.0, np.random.normal(mean_speed, speed_std))
                
                vx = inst_speed * np.cos(heading)
                vy = inst_speed * np.sin(heading)
                
                all_records.append({
                    "intersection": name,
                    "frame_id": frame,
                    "vehicle_id": global_vehicle_id % 300,  # reusable IDs
                    "vehicle_type": v_type,
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "vx": round(float(vx), 2),
                    "vy": round(float(vy), 2),
                    "speed": round(float(inst_speed), 2),
                    "ground_truth_state": state
                })
                
    df = pd.DataFrame(all_records)
    df.to_csv(output_path, index=False)
    print(f"[DataLoader] Generated {len(df)} vehicle trajectory records across 3 intersections -> {output_path}")
    return df


def load_raw_trajectories(csv_path: str = "data/eot_raw_trajectories.csv") -> pd.DataFrame:
    """Loads trajectory records from CSV, generating benchmark dataset if file does not exist."""
    if not os.path.exists(csv_path):
        print(f"[DataLoader] File not found at {csv_path}. Generating benchmark EoT trajectory data...")
        return generate_eot_benchmark_dataset(output_path=csv_path)
    df = pd.read_csv(csv_path)
    print(f"[DataLoader] Loaded {len(df)} records from {csv_path}")
    return df
