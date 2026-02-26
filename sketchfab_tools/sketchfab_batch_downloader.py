import requests
import time
import os
import json
from pathvalidate import sanitize_filename

def download_image(url, save_path):
    """이미지 URL로부터 파일을 다운로드하여 저장합니다."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"      [!] 다운로드 실패: {e}")
        return False

def load_history(save_dir):
    """지정된 디렉토리의 다운로드 기록을 불러옵니다."""
    history_path = os.path.join(save_dir, "download_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_history(save_dir, history_set):
    """다운로드 기록을 파일로 저장합니다."""
    history_path = os.path.join(save_dir, "download_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(list(history_set), f, indent=4)

def get_sketchfab_batch(query, api_token, tags=None, target_count=3000, save_dir="sketchfab_data"):
    """
    Sketchfab에서 대량의 썸네일을 안전하게 가져오고 다운로드합니다. (중복 방지 기능 포함)
    """
    search_url = "https://api.sketchfab.com/v3/search"
    headers = {"Authorization": f"Token {api_token}"}
    
    # 저장 디렉토리 생성
    save_dir = os.path.abspath(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"[*] 저장 경로: {save_dir}")

    # 기존 다운로드 기록 로드
    downloaded_ids = load_history(save_dir)
    print(f"[*] 기존 다운로드 기록: {len(downloaded_ids)}개 확인됨.")

    params = {
        "q": query,
        "type": "models",
        "count": 24, # 페이지당 최대 개수
        "sort_by": "-relevance"
    }
    
    if tags:
        params["tags"] = tags

    # 현재 세션에서 추가로 다운로드한 개수
    session_downloaded = 0
    total_downloaded = len(downloaded_ids)
    
    next_url = search_url
    
    print(f"[*] '{query}'에 대한 다운로드 시작 (현재: {total_downloaded} / 목표: {target_count})...")

    while next_url and total_downloaded < target_count:
        try:
            # 1. API 요청
            print(f"[*] 페이지 요청 중... (진행도: {total_downloaded}/{target_count})")
            response = requests.get(next_url, params=params if next_url == search_url else None, headers=headers)
            
            if response.status_code == 429:
                print("[!] Rate Limit 도달 (429). 60초간 대기합니다...")
                time.sleep(60)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            results = data.get('results', [])
            for model in results:
                if total_downloaded >= target_count:
                    break
                
                model_id = model.get('uid')
                model_name = sanitize_filename(model.get('name', 'unknown'))

                # [중요] 이미 다운로드한 ID인지 체크
                if model_id in downloaded_ids:
                    # 파일이 실제로 존재하는지도 체크 (선택 사항)
                    expected_fname = f"{model_id}_{model_name}.jpg"
                    if os.path.exists(os.path.join(save_dir, expected_fname)):
                        continue # 완전히 스킵
                
                # 썸네일 정보 추출
                thumbnails = model.get('thumbnails', {}).get('images', [])
                if not thumbnails:
                    continue
                
                best_img = max(thumbnails, key=lambda x: x.get('width', 0))
                img_url = best_img.get('url')
                img_width = best_img.get('width', 0)

                if not img_url or img_width < 256:
                    continue

                # 파일 저장
                file_name = f"{model_id}_{model_name}.jpg"
                save_path = os.path.join(save_dir, file_name)

                print(f"      [+] 다운로드 중: {model_name} ({img_width}px)")
                success = download_image(img_url, save_path)
                
                if success:
                    downloaded_ids.add(model_id)
                    session_downloaded += 1
                    total_downloaded = len(downloaded_ids)
                    
                    # 10개마다 기록 저장 (안전성)
                    if session_downloaded % 10 == 0:
                        save_history(save_dir, downloaded_ids)
                    
                    time.sleep(0.3)

            # 다음 페이지 업데이트
            next_url = data.get('next')
            
            # 페이지 전환 전 기록 저장
            save_history(save_dir, downloaded_ids)
            time.sleep(1.5)

        except Exception as e:
            print(f"[!] 에러 발생: {e}. 10초 대기...")
            time.sleep(10)
            continue

    # 최종 기록 저장
    save_history(save_dir, downloaded_ids)
    print(f"\n[+] 완료! 이번 세션에 {session_downloaded}개를 추가로 받아 총 {total_downloaded}개가 되었습니다.")

if __name__ == "__main__":
    # 실행 전 API 토큰을 확인하세요.
    MY_API_TOKEN = "3c1fd327428e42ecba47a6ce735c8103"
    
    # 검색 키워드 및 설정
    QUERY = "medieval" 
    TAGS = "medieval" 
    TARGET = 1000 
    
    get_sketchfab_batch(QUERY, MY_API_TOKEN, tags=TAGS, target_count=TARGET)
