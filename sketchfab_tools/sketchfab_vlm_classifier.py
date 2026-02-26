import os
import torch
import json
import time
from PIL import Image
from tqdm import tqdm
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from qwen_vl_utils import process_vision_info
import argparse
import pickle
import numpy as np

# --- Constants ---
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE_DIR = "sketchfab_data"
OUTPUT_DIR = "sketchfab_tools/sketchfab_classification_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDINGS_FILE = "sketchfab_tools/sketchfab_embeddings.pkl"

# --- Taxonomies ---
GENRES = ["Fantasy", "Sci-Fi", "Cyberpunk", "Modern", "Post-Apocalyptic", "Historical", "Horror", "Military", "Steampunk"]
STYLES = ["Photorealistic", "Stylized", "Low-Poly", "Toon / Anime", "Voxel", "Retro / PS1", "Abstract"]
SURFACES = ["Rusty", "Dusty", "Mossy", "Glow", "Blood-stained", "Wet", "Broken", "Clean"]
CATEGORIES = [
    "Animals & Creatures", "Architecture", "Art & Abstract", "Cars & Vehicles", "Characters & Creatures",
    "Cultural Heritage & History", "Electronics & Gadgets", "Fashion & Style", "Food & Drink",
    "Furniture & Home", "Music", "Nature & Plants", "News & Politics", "People",
    "Places & Locations", "Science & Technology", "Sports & Fitness", "Weapons & Military"
]

def load_embeddings():
    """DINOv2 임베딩 파일을 로드합니다."""
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"[!] Warning: Embeddings file {EMBEDDINGS_FILE} not found. RAG disabled.")
        return {}
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)

def find_visual_neighbors(embeddings, target_filename, top_n=3, threshold=0.8):
    """유사도가 높은 시각적 이웃을 찾습니다."""
    if target_filename not in embeddings:
        return []
    
    target_vec = embeddings[target_filename].flatten()
    target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    
    neighbors = []
    for filename, vec in embeddings.items():
        if filename == target_filename:
            continue
            
        vec = vec.flatten()
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        
        sim = np.dot(target_vec, vec)
        if sim >= threshold:
            neighbors.append((filename, sim))
    
    # 유사도 순 정렬
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return neighbors[:top_n]

def load_vlm():
    """Qwen3-VL 모델과 프로세서를 로드합니다."""
    print(f"[*] Loading {MODEL_ID} on {DEVICE}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if "cuda" in DEVICE else torch.float32,
        device_map="auto" if "cuda" in DEVICE else None,
        trust_remote_code=True
    )
    processor = Qwen3VLProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor

def analyze_image(image_path, model, processor, neighbors=None):
    """VLM을 사용하여 이미지를 분석합니다 (시각적 컨텍스트 RAG 포함)."""
    filename = os.path.basename(image_path)
    asset_name = filename.split("_", 1)[1] if "_" in filename else filename
    asset_name = os.path.splitext(asset_name)[0]
    
    neighbor_names = []
    neighbor_images = []
    
    if neighbors:
        for n_file, sim in neighbors:
            n_name = n_file.split("_", 1)[1] if "_" in n_file else n_file
            n_name = os.path.splitext(n_name)[0]
            neighbor_names.append(n_name)
            
            n_path = os.path.join(IMAGE_DIR, n_file)
            if os.path.exists(n_path):
                neighbor_images.append(n_path)
    
    neighbor_context = ""
    if neighbor_names:
        neighbor_context = f"\n[Visual Context] Visually similar assets to help your judgment: {', '.join(neighbor_names)}"

    prompt = f"""
    Analyze this 3D model thumbnail with high precision.
    The primary asset name is '{asset_name}'.{neighbor_context}
    
    1. **Category**: Choose the most accurate one from this list of popular standards: {CATEGORIES}
    2. **Genres**: Identify up to 5 top genres/atmospheres. Do NOT limit yourself to a list. Be specific (e.g., 'Gothic', 'Ancient Roman', 'Vibrant Fantasy'). Provide a confidence score (0.0 to 1.0) for each.
    3. **Styles**: Identify up to 5 top visual/artistic styles. Be specific (e.g., 'Hand-painted Stylized', 'Weathered Photorealistic', 'Voxel Art'). Provide a confidence score for each.
    4. **Surface Conditions**: Detect specific surface conditions (e.g., 'Rusty', 'Polished', 'Mossy', 'Dusty', 'Cracked').
    5. **Tags**: Provide 5-8 descriptive tags (lowercase, concise).
    
    Output strictly in JSON format:
    {{
        "category": "...",
        "genre": "best_genre_name",
        "top_genres": [
            {{"name": "...", "score": 0.9}},
            {{"name": "...", "score": 0.7}}
        ],
        "style": "best_style_name",
        "top_styles": [
            {{"name": "...", "score": 0.8}},
            {{"name": "...", "score": 0.6}}
        ],
        "surface_conditions": ["...", "..."],
        "tags": ["...", "...", "..."]
    }}
    """
    
    content = [
        {"type": "image", "image": f"file://{os.path.abspath(image_path)}"}
    ]
    
    # 이웃 이미지 추가 (최대 3개 정도로 제한하여 부하 방지)
    for n_path in neighbor_images[:3]:
        content.append({"type": "image", "image": f"file://{os.path.abspath(n_path)}"})
        
    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    # 전처리
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    # 생성
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    # JSON 파싱
    try:
        json_str = output_text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].strip()
        
        data = json.loads(json_str)
        
        # [추가] 0.3 미만 신뢰도 무시 및 최대 5개 유지 로직
        THRESHOLD = 0.3
        
        if "top_genres" in data:
            data["top_genres"] = [g for g in data["top_genres"] if float(g.get("score", 0)) >= THRESHOLD][:5]
            if data["top_genres"]:
                data["genre"] = data["top_genres"][0]["name"]
                
        if "top_styles" in data:
            data["top_styles"] = [s for s in data["top_styles"] if float(s.get("score", 0)) >= THRESHOLD][:5]
            if data["top_styles"]:
                data["style"] = data["top_styles"][0]["name"]

        # Validation: 카테고리가 목록에 없으면 기본값 매핑
        if data.get("category") not in CATEGORIES:
            data["category"] = "Architecture" # Default fallback
            
        return data
    except Exception as e:
        return {
            "category": "Architecture",
            "genre": "Fantasy",
            "top_genres": [],
            "style": "Photorealistic",
            "top_styles": [],
            "surface_conditions": [],
            "tags": []
        }

def process_all(limit=None):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    model, processor = load_vlm()
    embeddings = load_embeddings()
    
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"[*] Total files to process: {len(files)}")
    
    # Resume 지원을 위해 이미 처리된 파일 목록 확인
    processed = set([os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')])
    print(f"[*] Already processed: {len(processed)}")
    
    to_process = [f for f in files if os.path.splitext(f)[0] not in processed]
    
    # [추가] 제한(limit) 처리
    if limit:
        print(f"[*] Limiting processing to first {limit} files.")
        to_process = to_process[:limit]

    print(f"[*] Remaining files to process: {len(to_process)}")
    
    if not to_process:
        print("[+] Everything is already up to date.")
        return

    for filename in tqdm(to_process, desc="VLM 분석 중"):
        file_path = os.path.join(IMAGE_DIR, filename)
        file_id = os.path.splitext(filename)[0]
        
        # 이웃 찾기
        neighbors = find_visual_neighbors(embeddings, filename, top_n=3, threshold=0.8)
        
        try:
            result = analyze_image(file_path, model, processor, neighbors=neighbors)
            result["model_id_full"] = file_id
            
            # 파일명에 포함된 ID와 이름을 분리 저장
            if "_" in file_id:
                parts = file_id.split("_")
                result["sketchfab_id"] = parts[0]
                result["asset_name"] = "_".join(parts[1:])
            
            output_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"\n[!] Error processing {filename}: {e}")
            continue

    print(f"\n[+] Analysis complete. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of files to process")
    args = parser.parse_args()
    
    process_all(limit=args.limit)
