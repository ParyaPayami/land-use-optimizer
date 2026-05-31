import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def generate_manhattan_grid(num_parcels=5000):
    """
    Generates a dense grid of parcels shaped roughly like Manhattan.
    Manhattan is long (N-S) and narrow (E-W).
    Approx Bounds: Lat 40.70 to 40.88, Lon -74.02 to -73.93
    """
    print("Generating Manhattan Parcel Grid...")
    
    # Grid Logic
    lats = np.linspace(40.70, 40.88, 150) # North-South
    lons = np.linspace(-74.02, -73.93, 40)   # East-West
    
    parcels = []
    pid = 0
    
    for lat in lats:
        for lon in lons:
            # Simple shape filter for "Island" look (tilted rectangle)
            # Map outline approximation
            # Check if point is roughly inside the diagonal island shape
            # y = mx + b check or just simple noise bounding
            
            # Simple bounding box for demo speed but with some shape
            slope = (40.88 - 40.70) / (-73.93 - (-74.02))
            
            # Add some randomness to grid to look organic
            lat_jit = lat + np.random.normal(0, 0.0005)
            lon_jit = lon + np.random.normal(0, 0.0005)
            
            parcels.append({
                'parcel_id': pid,
                'lat': lat_jit, 
                'lon': lon_jit,
                'block_id': int(pid / 20)
            })
            pid += 1
            
    return pd.DataFrame(parcels)

def assign_current_land_use(df):
    """
    Assigns semi-realistic 'Current' land uses based on Lat/Lon zones.
    """
    print("Mapping Current Land Uses...")
    uses = []
    
    for _, row in df.iterrows():
        lat = row['lat']
        
        # Financial District (Downtown) -> Commercial/High Density
        if lat < 40.72:
            base_probs = [0.1, 0.6, 0.0, 0.2, 0.1, 0.0] # Res, Com, Ind, Mix, Pub, Open
            
        # SoHo/Tribeca/Village -> Mixed/Res
        elif 40.72 <= lat < 40.74:
            base_probs = [0.4, 0.2, 0.1, 0.2, 0.05, 0.05]
            
        # Midtown -> HEAVY Commercial
        elif 40.74 <= lat < 40.77:
            base_probs = [0.1, 0.7, 0.05, 0.1, 0.05, 0.0]
            
        # Upper West/East Side -> Residential
        elif 40.77 <= lat < 40.82:
            base_probs = [0.7, 0.1, 0.0, 0.05, 0.1, 0.05] # Central Park gap is hard to model purely by lat/lon without complex polys
            
        # Harlem/Uptown -> Residential/Mixed
        else:
            base_probs = [0.6, 0.2, 0.05, 0.1, 0.05, 0.0]
            
        use_code = np.random.choice(range(6), p=base_probs)
        uses.append(use_code)
        
    df['current_use_code'] = uses
    df['current_use_label'] = df['current_use_code'].map({
        0: 'Residential', 1: 'Commercial', 2: 'Industrial', 
        3: 'Mixed-Use', 4: 'Public', 5: 'Open Space'
    })
    return df

def optimize_land_use(df):
    """
    Simulates the PIMALUOS Agent Optimization.
    Logic:
    1. Increase Mixed-Use in Commercial zones (vitality).
    2. Add Open Space in dense Residential areas (equity).
    3. Buffer Industrial zones.
    """
    print("Running PIMALUOS Agent Optimization...")
    proposed_uses = []
    roi_lifts = []
    
    for _, row in df.iterrows():
        current = row['current_use_code']
        lat = row['lat']
        
        proposed = current # Default: Maintain (Action 1)
        roi = np.random.normal(2.5, 1.0) # Base ROI
        
        # Rule 1: Midtown (Lat 40.75) - Convert old Commercial to Mixed (Res+Com)
        if current == 1 and 40.74 < lat < 40.76 and np.random.random() > 0.7:
            proposed = 3 # To Mixed
            roi += 15.0 # High value add
            
        # Rule 2: Upper sides - Convert some Res to Open Space if dense
        if current == 0 and np.random.random() > 0.95:
            proposed = 5 # To Park (Environmental Agent win)
            roi -= 5.0 # Economic loss, but Social gain
            
        # Rule 3: SoHo - Convert Ind to Mixed (Loft conversions)
        if current == 2 and 40.72 < lat < 40.73:
            proposed = 3 # To Mixed
            roi += 25.0 # Massive ROI
            
        proposed_uses.append(proposed)
        roi_lifts.append(roi)
        
    df['proposed_use_code'] = proposed_uses
    df['proposed_use_label'] = df['proposed_use_code'].map({
        0: 'Residential', 1: 'Commercial', 2: 'Industrial', 
        3: 'Mixed-Use', 4: 'Public', 5: 'Open Space'
    })
    df['roi_lift'] = roi_lifts
    
    # Calculate Action Status
    status = []
    for c, p in zip(df['current_use_code'], df['proposed_use_code']):
        if c == p: status.append("Maintain")
        elif p == 3: status.append("Upzone (Mix)")
        elif p == 5: status.append("Preserve (Green)")
        else: status.append("Re-Zone")
    df['action_label'] = status
    
    return df

def main():
    # 1. Generate Space
    df = generate_manhattan_grid(num_parcels=6000)
    
    # 2. Assign Current State
    df = assign_current_land_use(df)
    
    # 3. Optimze
    df = optimize_land_use(df)
    
    # 4. Save
    output_dir = "results/full_scale_simulation"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "manhattan_landuse.csv")
    df.to_csv(output_path, index=False)
    print(f"Simulation Complete. Generated {len(df)} parcels.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
