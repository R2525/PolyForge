import os
import torch
import pickle
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# 1. 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/dinov2-base"
IMAGE_DIR = "objaverse_chairs"
OUTPUT_FILE = "sketchfab_tools/objaverse_chair_embeddings.pkl"

def analyze_objaverse_chairs():
    """objaverse_chairs 폴더의 모든 이미지를 DINOv2로 분석하여 벡터를 저장합니다."""
    print(f"[*] 모델 로드 중: {MODEL_ID} ({DEVICE})")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    embeddings = {}
    
    if not os.path.exists(IMAGE_DIR):
        print(f"[!] 에러: {IMAGE_DIR} 폴더가 없습니다.")
        return

    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"[*] 총 {len(files)}개의 이미지를 분석합니다.")

    for filename in tqdm(files, desc="분석 중"):
        file_path = os.path.join(IMAGE_DIR, filename)
        
        try:
            # 이미지 전처리 및 벡터 추출
            image = Image.open(file_path).convert("RGB")
            
            # 중앙 크롭 (재질/스타일 집중)
            w, h = image.size
            margin = 0.1
            image = image.crop((w * margin, h * margin, w * (1-margin), h * (1-margin)))
            
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # 스타일 맵 생성을 위해 모든 패치의 평균을 사용
                patch_tokens = outputs.last_hidden_state[:, 1:, :] 
                embedding = patch_tokens.mean(dim=1).cpu().numpy()
            
            embeddings[filename] = embedding
            
        except Exception as e:
            print(f"\n[!] {filename} 처리 중 오류 발생: {e}")

    # 결과 저장
    print(f"[*] 결과를 {OUTPUT_FILE}에 저장 중...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print("[+] 모든 분석이 완료되었습니다!")

if __name__ == "__main__":
    analyze_objaverse_chairs()
