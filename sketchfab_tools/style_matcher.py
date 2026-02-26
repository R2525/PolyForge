import os
import torch
import pickle
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torch.nn.functional import cosine_similarity

# 1. 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/dinov2-base"
EMBEDDINGS_FILE = "sketchfab_tools/sketchfab_embeddings.pkl"

def load_data():
    """미리 계산된 임베딩 데이터를 불러옵니다."""
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"[!] 에러: {EMBEDDINGS_FILE} 파일이 없습니다. analyzer를 먼저 실행하세요.")
        return None
    
    with open(EMBEDDINGS_FILE, 'rb') as f:
        return pickle.load(f)

def get_query_embedding(model, processor, image_path):
    """검색할 이미지의 DINOv2 벡터를 추출합니다."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu()
    return embedding

def find_matches(query_image_path, top_n=5):
    """대상 이미지와 가장 유사한 Sketchfab 모델을 찾습니다."""
    data = load_data()
    if not data: return

    print(f"[*] 모델 로드 중: {MODEL_ID}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    print(f"[*] '{query_image_path}' 스타일 분석 중...")
    query_vec = get_query_embedding(model, processor, query_image_path)

    scores = []
    
    for filename, vec_np in data.items():
        vec = torch.from_numpy(vec_np)
        similarity = cosine_similarity(query_vec, vec)
        scores.append((filename, similarity.item()))

    # 유사도 순으로 정렬
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- '{os.path.basename(query_image_path)}'와 가장 유사한 스타일 TOP {top_n} ---")
    for i in range(min(top_n, len(scores))):
        filename, score = scores[i]
        
        # 파일명에서 UUID 추출 (uuid_name.jpg 형식 가정)
        model_id = filename.split('_')[0]
        sketchfab_url = f"https://sketchfab.com/3d-models/{model_id}"
        
        print(f"{i+1}. [유사도: {score:.4f}] {filename}")
        print(f"   URL: {sketchfab_url}")
        print("-" * 30)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="스타일 기준이 되는 이미지 경로")
    parser.add_argument("--top_n", type=int, default=5, help="출력할 결과 개수")
    args = parser.parse_args()

    find_matches(args.query, args.top_n)
