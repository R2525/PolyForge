import os
import pickle
import glob
import json

def generate_report():
    report = {
        "objaverse": {"total": 0, "dino_analyzed": 0, "qwen_tagged": 0, "remaining": []},
        "sketchfab": {"total": 0, "dino_analyzed": 0, "qwen_tagged": 0, "remaining": []}
    }

    # 1. Check Objaverse
    obj_images = glob.glob("objaverse_mass_data/**/thumbnail.jpg", recursive=True)
    report["objaverse"]["total"] = len(obj_images)
    
    # DINO check
    dino_obj_file = "sketchfab_tools/objaverse_mass_embeddings.pkl"
    if os.path.exists(dino_obj_file):
        with open(dino_obj_file, "rb") as f:
            dino_obj = pickle.load(f)
            report["objaverse"]["dino_analyzed"] = len(dino_obj)

    # Qwen check
    qwen_obj_files = glob.glob("output/semantic_tags/objaverse/*.json")
    report["objaverse"]["qwen_tagged"] = len(qwen_obj_files)

    # 2. Check Sketchfab
    sf_images = glob.glob("sketchfab_data/*.jpg")
    report["sketchfab"]["total"] = len(sf_images)

    # DINO check
    dino_sf_file = "sketchfab_tools/sketchfab_embeddings.pkl"
    if os.path.exists(dino_sf_file):
        with open(dino_sf_file, "rb") as f:
            dino_sf = pickle.load(f)
            report["sketchfab"]["dino_analyzed"] = len(dino_sf)
    
    # Qwen check
    qwen_sf_files = glob.glob("output/semantic_tags/sketchfab/*.json")
    report["sketchfab"]["qwen_tagged"] = len(qwen_sf_files)

    # Save summary
    with open("analysis_status_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print("\n=== Analysis Progress Report ===")
    print(f"Objaverse: {report['objaverse']['qwen_tagged']}/{report['objaverse']['total']} Tagged, {report['objaverse']['dino_analyzed']} Embedded")
    print(f"Sketchfab: {report['sketchfab']['qwen_tagged']}/{report['sketchfab']['total']} Tagged, {report['sketchfab']['dino_analyzed']} Embedded")
    print("================================")

if __name__ == "__main__":
    generate_report()
