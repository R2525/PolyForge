import requests

def get_sketchfab_thumbnails(query, api_token, count=10):
    """
    Sketchfab에서 특정 키워드로 모델을 검색하고 썸네일 정보를 가져옵니다.
    """
    search_url = "https://api.sketchfab.com/v3/search"
    
    # 1. API 요청 헤더 및 파라미터 설정
    headers = {
        "Authorization": f"Token {api_token}"
    }
    params = {
        "q": query,           # 검색어 (예: 'cannon', 'pirate ship')
        "type": "models",     # 모델 데이터만 검색
        "count": count,       # 가져올 결과 개수 (최대 24, 페이지네이션 가능)
        "sort_by": "-relevance" # 관련성 높은 순서
    }

    try:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status() # 에러 발생 시 예외 처리
        
        data = response.json()
        model_list = []

        for model in data.get('results', []):
            # 모델 기본 정보 추출
            model_id = model.get('uid')
            model_name = model.get('name')
            
            # 썸네일 리스트 중 가장 큰 사이즈(보통 마지막 인덱스) 선택
            thumbnails = model.get('thumbnails', {}).get('images', [])
            if thumbnails:
                # 720p 혹은 가장 고화질 썸네일 URL 선택
                best_thumb = thumbnails[-1].get('url')
            else:
                best_thumb = None

            model_list.append({
                "id": model_id,
                "name": model_name,
                "thumbnail": best_thumb,
                "url": model.get('viewerUrl')
            })

        return model_list

    except Exception as e:
        print(f"API 요청 중 오류 발생: {e}")
        return []

if __name__ == "__main__":
    # --- 실행 및 테스트 ---
    # 실제 사용 시 이 부분을 사용자님의 토큰으로 수정하세요.
    MY_API_TOKEN = "3c1fd327428e42ecba47a6ce735c8103"
    search_keyword = "medieval cannon" # 중세 대포 검색

    results = get_sketchfab_thumbnails(search_keyword, MY_API_TOKEN, count=5)

    print(f"--- '{search_keyword}' 검색 결과 ---")
    for i, m in enumerate(results):
        print(f"{i+1}. {m['name']}")
        print(f"   ID: {m['id']}")
        print(f"   Thumbnail: {m['thumbnail']}")
        print("-" * 30)
