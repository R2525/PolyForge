import argparse
import glob
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm

# classification 폴더를 모듈 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "classification")))

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from qwen_vl_utils import process_vision_info
from classification_config import CATEGORY_PROMPTS, GENRE_TAXONOMY, STYLE_TAXONOMY, SURFACE_CONDITION_TAXONOMY

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

def load_vlm():
    print(f"[*] Loading Custom VLM (Qwen3-VL): {MODEL_ID} on {DEVICE}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    processor = Qwen3VLProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor

def build_prompt():
    genre_list = ", ".join(GENRE_TAXONOMY.keys())
    style_list = ", ".join(STYLE_TAXONOMY.keys())
    surface_list = ", ".join(SURFACE_CONDITION_TAXONOMY.keys())
    categories = ", ".join(CATEGORY_PROMPTS.keys())

    prompt = f"""
    Analyze the provided image of a 3D model.
    Extract the following semantic features in a structured JSON format.
    
    1. CATEGORY: The most suitable category from: {categories}.
    2. TOP_GENRES: Up to 3 most relevant genres from: {genre_list}.
       Provide a "name" and "score" (0.0 to 1.0) for each.
    3. STYLE: The main visual style from: {style_list}.
    4. TOP_STYLES: Up to 4 most relevant styles from: {style_list}.
       Provide a "name" and "score" (0.0 to 1.0) for each.
    5. SURFACE_CONDITIONS: Relevant material/surface traits from: {surface_list}.
    6. TAGS: 3 to 5 simple, descriptive keywords for the object.
    
    IMPORTANT:
    - Only include items with a score > 0.4.
    - Output ONLY valid JSON.
    
    Example Output:
    {{
        "category": "Food & Drink",
        "top_genres": [
            {{"name": "Post-Apocalyptic", "score": 0.75}},
            {{"name": "Historical", "score": 0.65}}
        ],
        "style": "Low-Poly",
        "top_styles": [
            {{"name": "Low-Poly", "score": 0.9}},
            {{"name": "Retro / PS1", "score": 0.6}}
        ],
        "surface_conditions": ["Rusty", "Clean"],
        "tags": ["chicken", "roasted", "food"]
    }}
    """
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Mass Semantic Tagger using Qwen VLM")
    parser.add_argument("--input_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_dir", required=True, help="Directory to save JSON tags")
    parser.add_argument("--filter", default="*.jpg", help="File filter (e.g., thumbnail.jpg or *.jpg)")
    parser.add_argument("--metadata_dir", help="Directory containing metadata JSONs for remote fetching")
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tags")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Find inputs
    all_inputs = [] # List of (id, path_or_url, is_remote)
    
    if args.metadata_dir:
        # Remote mode: Iterate through metadata JSONs
        meta_files = glob.glob(os.path.join(args.metadata_dir, "*.json"))
        print(f"[*] Found {len(meta_files)} metadata files in {args.metadata_dir}")
        for mf in meta_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    asset_id = data.get("uid")
                    # Sketchfab style: thumbnails.images[0].url
                    thumb_url = None
                    if "thumbnails" in data and "images" in data["thumbnails"]:
                        thumb_url = data["thumbnails"]["images"][0]["url"]
                    
                    if asset_id and thumb_url:
                        all_inputs.append((asset_id, thumb_url, True))
            except:
                continue
    else:
        # Local mode: Glob images
        search_pattern = os.path.join(args.input_dir, args.filter)
        if args.recursive:
            search_pattern = os.path.join(args.input_dir, "**", args.filter)
        
        img_files = glob.glob(search_pattern, recursive=args.recursive)
        print(f"[*] Found {len(img_files)} images in {args.input_dir}")
        for img_path in img_files:
            if "objaverse_mass_data" in img_path:
                asset_id = os.path.basename(os.path.dirname(img_path))
            else:
                asset_id = os.path.splitext(os.path.basename(img_path))[0].split('_')[0]
            all_inputs.append((asset_id, img_path, False))

    if not all_inputs:
        print("[!] No inputs found.")
        return

    # Load VLM
    model, processor = load_vlm()
    prompt = build_prompt()

    for asset_id, source, is_remote in tqdm(all_inputs, desc="Tagging"):
        out_file = os.path.join(args.output_dir, f"{asset_id}_tags.json")
        
        if os.path.exists(out_file) and not args.overwrite:
            continue

        try:
            if is_remote:
                # Fetch from URL
                resp = requests.get(source, timeout=10)
                image = Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                # Load from local
                image = Image.open(source).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            # --- Optimization: Half Resolution (approx 960x540 = 518,400 pixels) ---
            # Pass min_pixels and max_pixels directly to processor
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                min_pixels=518400,
                max_pixels=518400
            ).to(DEVICE)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            # JSON Parsing
            json_str = output_text.strip()
            if "```" in json_str:
                parts = json_str.split("```")
                for p in parts:
                    if "{" in p and "}" in p:
                        json_str = p
                        if json_str.startswith("json"): json_str = json_str[4:]
                        break
            
            tag_data = json.loads(json_str)
            
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(tag_data, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"\n[!] Error processing {img_path}: {e}")
            # pass

    print(f"\n[*] Finished! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
