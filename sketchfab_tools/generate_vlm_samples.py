import os
import glob
from PIL import Image

def generate_samples(input_dir, output_dir, count=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images = glob.glob(os.path.join(input_dir, "**", "thumbnail.jpg"), recursive=True)
    samples = images[:count]
    
    saved_paths = []
    for i, img_path in enumerate(samples):
        img = Image.open(img_path).convert("RGB")
        # Resize to approx half (960x540)
        img_resized = img.resize((960, 540), Image.Resampling.LANCZOS)
        
        asset_id = os.path.basename(os.path.dirname(img_path))
        out_path = os.path.abspath(os.path.join(output_dir, f"sample_{i}_{asset_id}_960.jpg"))
        img_resized.save(out_path)
        saved_paths.append(out_path)
        print(f"[*] Saved sample: {out_path}")
    
    return saved_paths

if __name__ == "__main__":
    input_dir = "objaverse_mass_data"
    # Artifact directory for embedding in walkthrough
    output_dir = r"C:\Users\yulee\.gemini\antigravity\brain\d551d9b8-30e0-4f57-889a-cfdf871dc7ad\samples"
    generate_samples(input_dir, output_dir)
