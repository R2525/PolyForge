import objaverse
import os
import json
import requests
import shutil
from tqdm import tqdm

# --- Configuration ---
SAVE_ROOT = "objaverse_mass_data"
# 수집할 LVIS 카테고리 (객체 중심)
TARGET_LVIS_CATEGORIES = [
    "chair", "sword", "helmet", "shield", "lamp", "armor", "axe", "chest", "bottle"
]
# 수집할 키워드 (장르/분위기 중심)
TARGET_KEYWORDS = [
    "medieval", "fantasy", "sci-fi", "modern", "cyberpunk", "stylized", "realistic"
]

LIMIT_PER_CATEGORY = 50 # 본격 수집을 위해 50으로 상향
DOWNLOAD_GLB = False # 용량 관계상 일단 False로 설정 (Thumbnail 우선)

def download_asset(uid, metadata, save_path):
    """UID와 메타데이터를 기반으로 에셋 저장"""
    asset_folder = os.path.join(save_path, uid)
    if not os.path.exists(asset_folder):
        os.makedirs(asset_folder)

    # 1. 메타데이터 저장
    with open(os.path.join(asset_folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    # 2. 썸네일 다운로드
    try:
        thumbnails = metadata.get('thumbnails', {}).get('images', [])
        if thumbnails:
            thumb_url = thumbnails[0].get('url')
            if thumb_url:
                resp = requests.get(thumb_url, timeout=10)
                resp.raise_for_status()
                with open(os.path.join(asset_folder, "thumbnail.jpg"), "wb") as f:
                    f.write(resp.content)
    except Exception as e:
        pass # 썸네일 실패는 일단 무시

    # 3. GLB 모델 다운로드 (필요시)
    if DOWNLOAD_GLB:
        try:
            objects = objaverse.load_objects([uid])
            downloaded_path = objects.get(uid)
            if downloaded_path:
                shutil.copy(downloaded_path, os.path.join(asset_folder, "model.glb"))
        except Exception:
            pass

def mass_collect():
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)

    print("[*] Objaverse LVIS 애노테이션 로드 중...")
    lvis_annotations = objaverse.load_lvis_annotations()
    
    total_collected = 0
    collection_stats = {}

    # 1. LVIS 카테고리 기반 수집
    print(f"[*] 객체별 수집 시작 (타겟: {len(TARGET_LVIS_CATEGORIES)}개 카테고리)")
    for category in TARGET_LVIS_CATEGORIES:
        uids = lvis_annotations.get(category, [])
        if not uids:
            continue
            
        target_uids = uids[:LIMIT_PER_CATEGORY]
        print(f"   > '{category}': {len(target_uids)}개 에셋 로드 중...")
        
        # 메타데이터 한꺼번에 로드
        metas = objaverse.load_annotations(target_uids)
        
        cat_path = os.path.join(SAVE_ROOT, f"obj_{category}")
        if not os.path.exists(cat_path): os.makedirs(cat_path)

        for uid in tqdm(target_uids, desc=f"obj_{category}", leave=False):
            download_asset(uid, metas.get(uid, {}), cat_path)
            total_collected += 1
            
        collection_stats[f"obj_{category}"] = len(target_uids)

    # 2. 키워드 검색 기반 수집 (장르)
    # 전체 메타데이터를 순회하는 대신, 일단 objaverse 1.0의 전체 UID 리스트를 활용
    # 실제로는 대량 검색이 필요하므로 여기서는 샘플링 방식으로 구현
    print(f"\n[*] 장르별 키워드 수집 시작 (타겟: {len(TARGET_KEYWORDS)}개 키워드)")
    # 주의: load_annotations()는 UID 리스트가 필요하므로, 
    # 대량 데이터에서는 uids 리스트를 청크 단위로 처리해야 함
    all_uids = objaverse.load_uids()
    
    # 키워드 검색을 위해 상위 50,000개만 샘플링하여 조사 (시간 절약)
    sample_uids = all_uids[:50000] 
    print(f"   > 샘플링된 {len(sample_uids)}개 메타데이터 분석 중...")
    
    # 메타데이터 청크 로드
    chunk_size = 5000
    keyword_matches = {kw: [] for kw in TARGET_KEYWORDS}
    
    for i in range(0, len(sample_uids), chunk_size):
        chunk = sample_uids[i:i+chunk_size]
        metas = objaverse.load_annotations(chunk)
        
        for uid, meta in metas.items():
            name = meta.get("name", "").lower()
            desc = meta.get("description", "").lower()
            
            # tags가 문자열 리스트일 수도, 객체(dict) 리스트일 수도 있음
            tags_raw = meta.get("tags", [])
            tags_clean = []
            for t in tags_raw:
                if isinstance(t, str):
                    tags_clean.append(t.lower())
                elif isinstance(t, dict) and "name" in t:
                    tags_clean.append(t["name"].lower())
            
            full_text = f"{name} {desc} {' '.join(tags_clean)}"
            for kw in TARGET_KEYWORDS:
                if kw in full_text and len(keyword_matches[kw]) < LIMIT_PER_CATEGORY:
                    keyword_matches[kw].append((uid, meta))

    for kw, matches in keyword_matches.items():
        if not matches: continue
        
        kw_path = os.path.join(SAVE_ROOT, f"genre_{kw}")
        if not os.path.exists(kw_path): os.makedirs(kw_path)
        
        print(f"   > '{kw}': {len(matches)}개 에셋 저장 중...")
        for uid, meta in tqdm(matches, desc=f"genre_{kw}", leave=False):
            download_asset(uid, meta, kw_path)
            total_collected += 1
            
        collection_stats[f"genre_{kw}"] = len(matches)

    print("\n" + "="*50)
    print(f"[SUCCESS] 총 {total_collected}개의 에셋 수집 완료!")
    print("="*50)
    for k, v in collection_stats.items():
        print(f" - {k}: {v} items")

if __name__ == "__main__":
    mass_collect()
