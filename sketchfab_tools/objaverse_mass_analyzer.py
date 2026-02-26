import os
import torch
import pickle
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import numpy as np

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/dinov2-base"
SOURCE_ROOT = "objaverse_mass_data"
OUTPUT_FILE = "sketchfab_tools/objaverse_mass_embeddings.pkl"

def extract_vector(image, processor, model):
    img = image.convert("RGB")
    # Style에 집중하기 위해 중앙 크롭 (분석기와 동일하게 70% 영역)
    w, h = img.size
    left, top, right, bottom = w * 0.15, h * 0.15, w * 0.85, h * 0.85
    img = img.crop((left, top, right, bottom))
    
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        # GAP (Global Average Pooling) 방식 사용
        patch_tokens = outputs.last_hidden_state[:, 1:, :] 
        return patch_tokens.mean(dim=1).cpu().numpy()

def analyze_mass_data():
    if not os.path.exists(SOURCE_ROOT):
        print(f"[!] {SOURCE_ROOT}를 찾을 수 없습니다.")
        return

    print(f"[*] 모델 로드 중: {MODEL_ID} ({DEVICE})")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    embeddings = {}
    
    # 모든 하위 폴더 순회
    all_files = []
    for root, dirs, files in os.walk(SOURCE_ROOT):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and "thumbnail" in f.lower():
                all_files.append(os.path.join(root, f))

    print(f"[*] 총 {len(all_files)}개의 썸네일을 분석합니다.")

    for file_path in tqdm(all_files, desc="DINO 특징 추출"):
        try:
            # 경로 구조: objaverse_mass_data/category_dir/uid/thumbnail.jpg
            parts = os.path.normpath(file_path).split(os.sep)
            if len(parts) >= 3:
                category = parts[-3]
                uid = parts[-2]
                
                image = Image.open(file_path).convert("RGB")
                vec = extract_vector(image, processor, model)
                
                # 저장 형식: { "uid": { "vector": vec, "category": cat, "rel_path": path } }
                # 또는 단순히 explorer 호환성을 위해 { path: vec }
                # 여기서는 확장성을 위해 dict 구조로 저장
                embeddings[file_path] = {
                    "vector": vec,
                    "category": category,
                    "uid": uid
                }
        except Exception as e:
            print(f"\n[!] {file_path} 처리 중 오류: {e}")

    print(f"[*] 결과를 {OUTPUT_FILE}에 저장 중...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print("[+] 모든 분석이 완료되었습니다!")

if __name__ == "__main__":
    analyze_mass_data()
