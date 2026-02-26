import requests
import time
import os
import json
from pathvalidate import sanitize_filename
from sketchfab_batch_downloader import download_image, load_history, save_history

# --- Configuration ---
MY_API_TOKEN = "3c1fd327428e42ecba47a6ce735c8103"
SAVE_DIR = "sketchfab_data"
# 인기 있는 카테고리/태그 리스트
POPULAR_QUERIES = [
    # Genres
    "sci-fi", "fantasy", "horror", "cyberpunk", "steampunk", "post-apocalyptic",
    # Styles
    "stylized", "low-poly", "photorealistic", "anime", "voxel",
    # Fandoms
    "star wars", "pokemon", "warhammer", "marvel", "dc", "minecraft", "ghibli"
]
TARGET_PER_QUERY = 150 # 쿼리당 150개씩 수집 (약 18개 쿼리 * 150 = 2700개 추가 목표)

def get_popular_assets(query, api_token, target_count=100, save_dir="sketchfab_data"):
    """
    특정 쿼리에 대해 가장 인기 있는(Likes 기준) 에셋들을 수집합니다.
    """
    search_url = "https://api.sketchfab.com/v3/search"
    headers = {"Authorization": f"Token {api_token}"}
    
    save_dir = os.path.abspath(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    downloaded_ids = load_history(save_dir)
    
    # 팁: Sketchfab API에서 인기순 정렬은 '-like_count' 또는 '-view_count'가 일반적입니다.
    params = {
        "q": query,
        "type": "models",
        "count": 24,
        "sort_by": "-like_count" # 좋아요 순 정렬
    }
    
    session_downloaded = 0
    next_url = search_url
    
    print(f"\n[*] '{query}' 인기 에셋 수집 시작 (목표: {target_count}개)...")

    while next_url and session_downloaded < target_count:
        try:
            response = requests.get(next_url, params=params if next_url == search_url else None, headers=headers)
            
            if response.status_code == 429:
                print("[!] Rate Limit 도달. 60초 대기...")
                time.sleep(60)
                continue
                
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                print(f"[*] '{query}'에 대한 더 이상의 결과가 없습니다.")
                break

            for model in results:
                if session_downloaded >= target_count:
                    break
                
                model_id = model.get('uid')
                model_name = sanitize_filename(model.get('name', 'unknown'))

                if model_id in downloaded_ids:
                    continue
                
                thumbnails = model.get('thumbnails', {}).get('images', [])
                if not thumbnails:
                    continue
                
                # 고해상도 이미지 선택
                best_img = max(thumbnails, key=lambda x: x.get('width', 0))
                img_url = best_img.get('url')
                
                if not img_url:
                    continue

                file_name = f"{model_id}_{model_name}.jpg"
                save_path = os.path.join(save_dir, file_name)

                print(f"      [+] ({session_downloaded+1}/{target_count}) 다운로드: {model_name}")
                success = download_image(img_url, save_path)
                
                if success:
                    downloaded_ids.add(model_id)
                    session_downloaded += 1
                    save_history(save_dir, downloaded_ids)
                    time.sleep(0.2)

            next_url = data.get('next')
            time.sleep(1)

        except Exception as e:
            print(f"[!] 에러: {e}")
            time.sleep(5)
            continue

    print(f"[+] '{query}' 수집 완료: {session_downloaded}개 추가됨.")

def main():
    print(f"[*] 다양한 인기 장르/스타일/팬덤 에셋 수집을 시작합니다.")
    print(f"[*] 키워드 목록: {POPULAR_QUERIES}")
    
    for query in POPULAR_QUERIES:
        get_popular_assets(query, MY_API_TOKEN, target_count=TARGET_PER_QUERY, save_dir=SAVE_DIR)
        print("-" * 50)
        time.sleep(2) # 쿼리 간 휴식
        
    print("\n[SUCCESS] 모든 키워드에 대한 수집이 완료되었습니다!")

if __name__ == "__main__":
    main()
