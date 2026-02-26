import os
import json
from PIL import Image
import numpy as np

def check_map_colors(json_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    mesh_info = data.get("mesh_info", {})
    norm_path = mesh_info.get("normal_map")
    rough_path = mesh_info.get("roughness_map")
    
    print(f"\nAsset: {data.get('asset_id')}")
    
    for label, path in [("Normal Map", norm_path), ("Roughness Map", rough_path)]:
        if not path or not os.path.exists(path):
            print(f"{label}: Not found or null ({path})")
            continue
            
        try:
            img = Image.open(path).convert("RGBA") # Use RGBA to check alpha
            arr = np.array(img)
            avg_color = arr.mean(axis=(0, 1))
            print(f"{label}: {os.path.basename(path)}")
            print(f"  - Avg Color (RGBA): {avg_color}")
            
            if label == "Normal Map":
                # Standard: R~128, G~128, B~255
                # Swizzled: R~small, G~128+, B~128+, A~128+
                if avg_color[2] > 200 and avg_color[0] > 100:
                    print("  - Status: LOOKS LIKE Standard Purple Normal Map")
                elif avg_color[1] > 120 and avg_color[2] > 120 and avg_color[0] < 100:
                    print("  - Status: LOOKS LIKE Unity Swizzled Normal Map (Teal/Cyan)")
                else:
                    print("  - Status: LOOKS LIKE Something else (possibly misidentified)")
        except Exception as e:
            print(f"{label}: Error reading {path} - {e}")

if __name__ == "__main__":
    check_map_colors("output/iso_analysis/SM_fireex_analysis.json")
