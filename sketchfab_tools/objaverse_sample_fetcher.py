import objaverse
import os
import json
import requests
import shutil

# --- Configuration ---
SAVE_ROOT = "objaverse_samples"
SAMPLE_COUNT = 4

def fetch_samples():
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)
        
    print("[*] Objaverse 애노테이션 로드 중...")
    # 전체 애노테이션을 로드하면 너무 무거우므로 일부만 가져오는 방식 확인
    # objaverse.load_annotations()는 전체 80만개를 로드하므로 메모리 주의
    # 대신 uids 리스트를 먼저 가져옵니다.
    uids = objaverse.load_uids()
    print(f"[*] 총 {len(uids)}개의 UIDs 확인됨.")
    
    # 샘플 4개 선택 (앞에서부터 4개)
    sample_uids = uids[:SAMPLE_COUNT]
    print(f"[*] 선택된 샘플 UIDs: {sample_uids}")
    
    # 1. 메타데이터 가져오기
    print("[*] 메타데이터 로드 중...")
    annotations = objaverse.load_annotations(sample_uids)
    
    # 2. 모델 다운로드
    print("[*] 모델(GLB) 다운로드 중...")
    objects = objaverse.load_objects(sample_uids)
    
    for uid in sample_uids:
        # 폴더 생성
        asset_folder = os.path.join(SAVE_ROOT, uid)
        if not os.path.exists(asset_folder):
            os.makedirs(asset_folder)
            
        # 메타데이터 저장
        meta = annotations.get(uid, {})
        with open(os.path.join(asset_folder, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
            
        # 모델 이동/복사 (objaverse는 기본적으로 ~/.objaverse에 저장함)
        downloaded_path = objects.get(uid)
        if downloaded_path and os.path.exists(downloaded_path):
            shutil.copy(downloaded_path, os.path.join(asset_folder, "model.glb"))
            print(f"      [+] 모델 저장 완료: {uid}")
            
        # 썸네일 다운로드
        # Objaverse 1.0 메타데이터에는 보통 'thumbnails' 관련 정보가 있음
        # 'thumbnails' -> 'images' -> 리스트
        # (참고: metadata['thumbnails']['images'][0]['url'])
        try:
            thumbnails = meta.get('thumbnails', {}).get('images', [])
            if thumbnails:
                thumb_url = thumbnails[0].get('url')
                if thumb_url:
                    resp = requests.get(thumb_url, timeout=10)
                    resp.raise_for_status()
                    with open(os.path.join(asset_folder, "thumbnail.jpg"), "wb") as f:
                        f.write(resp.content)
                    print(f"      [+] 썸네일 다운로드 완료: {uid}")
        except Exception as e:
            print(f"      [!] 썸네일 다운로드 실패 ({uid}): {e}")

    print(f"\n[SUCCESS] 4개의 샘플 에셋이 {SAVE_ROOT} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    fetch_samples()
