import objaverse
import os
import json
import requests
from tqdm import tqdm

# --- Configuration ---
SAVE_DIR = "objaverse_chairs"
KEYWORD = "chair"
TARGET_COUNT = 300 # 분석에 적당한 양

def collect_chairs():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"[*] Objaverse에서 '{KEYWORD}' 검색 중...")
    
    # 1. 80만개 모델 정보 로드 (처음 실행 시 데이터 다운로드로 시간이 걸릴 수 있음)
    annotations = objaverse.load_annotations()
    
    found_uids = []
    print("[*] 키워드 필터링 중...")
    for uid, meta in tqdm(annotations.items()):
        name = meta.get('name', '').lower()
        tags = [t.get('name', '').lower() for t in meta.get('tags', [])]
        
        if KEYWORD in name or any(KEYWORD in t for t in tags):
            found_uids.append(uid)
            if len(found_uids) >= TARGET_COUNT:
                break
                
    print(f"[*] 총 {len(found_uids)}개의 '{KEYWORD}' 에셋을 찾았습니다.")
    
    # 2. 이미지(썸네일) 다운로드
    print("[*] 썸네일 다운로드 시작...")
    for uid in tqdm(found_uids):
        save_path = os.path.join(SAVE_DIR, f"{uid}.jpg")
        if os.path.exists(save_path):
            continue
            
        meta = annotations[uid]
        thumbnails = meta.get('thumbnails', {}).get('images', [])
        if thumbnails:
            # 가장 큰 이미지 선택
            best_img = max(thumbnails, key=lambda x: x.get('width', 0))
            url = best_img.get('url')
            
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    with open(save_path, "wb") as f:
                        f.write(resp.content)
            except Exception as e:
                pass
                
    print(f"\n[SUCCESS] {len([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')])}개의 이미지가 {SAVE_DIR}에 저장되었습니다.")

if __name__ == "__main__":
    collect_chairs()
