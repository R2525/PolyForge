import requests
import time
import os
import json
from tqdm import tqdm

# --- Configuration ---
MY_API_TOKEN = "3c1fd327428e42ecba47a6ce735c8103"
IMAGE_DIR = "sketchfab_data"
METADATA_DIR = "sketchfab_metadata"

def fetch_model_metadata(model_id, api_token):
    """Sketchfab API로부터 단일 모델의 전체 메타데이터를 가져옵니다."""
    url = f"https://api.sketchfab.com/v3/models/{model_id}"
    headers = {"Authorization": f"Token {api_token}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 429:
            print(f"\n[!] Rate Limit! 60초 대기...")
            time.sleep(60)
            return fetch_model_metadata(model_id, api_token) # 재시도
            
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"\n[!] {model_id} 메타데이터 가져오기 실패: {e}")
        return None

def harvest_metadata():
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)
        
    # 다운로드된 이미지 목록 확인
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"[*] 분석할 이미지 총 수: {len(files)}")
    
    # 이미 메타데이터가 있는 파일 확인
    existing_meta = set([os.path.splitext(f)[0] for f in os.listdir(METADATA_DIR) if f.endswith('.json')])
    
    to_process = []
    for f in files:
        file_id = f.split('_')[0]
        if file_id not in existing_meta:
            to_process.append((file_id, f))
            
    print(f"[*] 새로 수집할 메타데이터: {len(to_process)}개")
    
    if not to_process:
        print("[+] 모든 메타데이터가 최신 상태입니다.")
        return

    for model_id, filename in tqdm(to_process, desc="메타데이터 수집 중"):
        metadata = fetch_model_metadata(model_id, MY_API_TOKEN)
        
        if metadata:
            # 우리가 필요한 핵심 정보만 추출하거나 전체 저장
            # 여기서는 전체 정보를 저장하되 명명 규칙을 통일합니다.
            output_path = os.path.join(METADATA_DIR, f"{model_id}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            # API 부하 방지
            time.sleep(0.5)

    print(f"\n[SUCCESS] 메타데이터 수집 완료! 저장 경로: {METADATA_DIR}")

if __name__ == "__main__":
    harvest_metadata()
