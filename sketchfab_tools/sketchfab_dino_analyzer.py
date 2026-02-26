import os
import torch
import pickle
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# 1. 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/dinov2-base"
IMAGE_DIR = "sketchfab_data"
OUTPUT_FILE = "sketchfab_tools/sketchfab_embeddings.pkl"

def analyze_dataset():
    """sketchfab_data 폴더의 모든 이미지를 DINOv2로 분석하여 벡터를 저장합니다."""
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
            
            # [추가] 재질(Style)에 더 집중하기 위해 중앙 크롭 시도 (선택 사항)
            # thumbnails의 경우 외곽 여백이 많아 중앙 70% 정도만 보는 것이 재질 분석에 유리할 수 있음
            w, h = image.size
            left = w * 0.15
            top = h * 0.15
            right = w * 0.85
            bottom = h * 0.85
            image = image.crop((left, top, right, bottom))
            
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # [수정] CLS 토큰 (index 0) 대신 모든 패치 토큰 (index 1~)의 평균값을 사용
                # 이는 '무엇(Shape)' 인지보다 '어떤 느낌(Texture/Style)' 인지를 더 잘 포착합니다.
                patch_tokens = outputs.last_hidden_state[:, 1:, :] # [1, 256, 768] (vit-base 기준)
                embedding = patch_tokens.mean(dim=1).cpu().numpy()
            
            # 정보 저장 (파일명에서 ID 추출 가능하므로 파일명 자체를 키로 사용)
            embeddings[filename] = embedding
            
        except Exception as e:
            print(f"\n[!] {filename} 처리 중 오류 발생: {e}")

    # 결과 저장
    print(f"[*] 결과를 {OUTPUT_FILE}에 저장 중...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print("[+] 모든 분석이 완료되었습니다!")

if __name__ == "__main__":
    analyze_dataset()
