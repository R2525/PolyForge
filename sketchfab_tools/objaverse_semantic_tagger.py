import os
import sys
import torch
import json
from PIL import Image

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

def run_sample_tagging():
    sample_file = "sample_paths.txt"
    if not os.path.exists(sample_file):
        print("[!] sample_paths.txt not found.")
        return

    with open(sample_file, "r", encoding="utf-16") as f: # PowerShell Out-File default
        paths = [line.strip() for line in f.readlines() if line.strip()]

    model, processor = load_vlm()
    prompt = build_prompt()
    output_dir = "output/semantic_tags"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    for path in paths:
        print(f"[*] Tagging: {path}")
        try:
            image = Image.open(path)
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
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            # JSON 파싱
            json_str = output_text.strip()
            if "```" in json_str:
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"): json_str = json_str[4:]
            
            tag_data = json.loads(json_str)
            
            # 파일별 저장
            asset_id = os.path.basename(os.path.dirname(path))
            out_file = os.path.join(output_dir, f"{asset_id}_tags.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(tag_data, f, indent=4, ensure_ascii=False)
            
            results.append({"asset_id": asset_id, "tags": tag_data})
            print(f"    [+] Saved: {out_file}")
            
        except Exception as e:
            print(f"    [!] Error processing {path}: {e}")

    # 최종 결과 요약 출력
    print("\n" + "="*50)
    print("[SUCCESS] 5 samples tagged. Final JSON previews:")
    for res in results:
        print(f"--- {res['asset_id']} ---")
        print(json.dumps(res['tags'], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_sample_tagging()
