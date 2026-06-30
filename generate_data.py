"""
generate_data.py
-----------------
Generates a synthetic agricultural supply chain shipment dataset.

Real shipment-level spoilage data is not publicly available at scale, so this
script simulates realistic data using domain-informed rules (e.g. longer
storage and higher humidity increase spoilage), plus random noise so the
dataset isn't perfectly predictable. This mirrors how storage conditions and
crop loss relate in real agricultural supply chains.

Run:
    python generate_data.py

Output:
    data/shipments.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000

crop_types = ["Tomato", "Banana", "Potato", "Onion", "Mango", "Wheat", "Rice"]
transport_modes = ["Truck", "Rail", "Refrigerated Truck", "Open Cart"]
regions = ["North", "South", "East", "West", "Central"]

crop_base_loss = {
    "Tomato": 6, "Banana": 7, "Mango": 6.5, "Potato": 3,
    "Onion": 2.5, "Wheat": 1, "Rice": 1.2
}

df = pd.DataFrame({
    "shipment_id": range(1, N + 1),
    "crop_type": np.random.choice(crop_types, N),
    "storage_days": np.random.randint(0, 15, N),
    "temperature": np.round(np.random.uniform(10, 40, N), 1),
    "humidity": np.round(np.random.uniform(30, 95, N), 1),
    "distance_km": np.random.randint(5, 1500, N),
    "transport_mode": np.random.choice(transport_modes, N, p=[0.45, 0.15, 0.25, 0.15]),
    "delay_hours": np.round(np.abs(np.random.normal(5, 6, N)), 1),
    "region": np.random.choice(regions, N),
})

base = df["crop_type"].map(crop_base_loss)
storage_effect = df["storage_days"] * 1.3
humidity_effect = np.where(df["humidity"] > 80, (df["humidity"] - 80) * 0.9, 0)
temp_effect = np.where(df["temperature"] > 30, (df["temperature"] - 30) * 0.5, 0)
distance_effect = df["distance_km"] * 0.004
delay_effect = df["delay_hours"] * 0.6
transport_effect = df["transport_mode"].map({
    "Refrigerated Truck": -4, "Truck": 0, "Rail": -1, "Open Cart": 5
})
noise = np.random.normal(0, 2.5, N)

loss = (base + storage_effect + humidity_effect + temp_effect +
        distance_effect + delay_effect + transport_effect + noise)
df["loss_percent"] = np.clip(loss, 0, 95).round(2)

missing_idx = np.random.choice(df.index, size=int(0.02 * N), replace=False)
df.loc[missing_idx, "humidity"] = np.nan

dup_rows = df.sample(10, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

df.to_csv("data/shipments.csv", index=False)
print(f"Saved {len(df)} rows to data/shipments.csv")
