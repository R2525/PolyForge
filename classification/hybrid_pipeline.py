import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import sys
import subprocess
import torch
import numpy as np
from PIL import Image
import argparse
import json
import tempfile
import time
import glob
import threading

# 1. 초기화
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def parse_unity_mat(mat_path, texture_dir="assets/texture"):
    """Unity .mat 파일(YAML)에서 텍스처 경로를 추출합니다."""
    if not mat_path or not os.path.exists(mat_path):
        return None, None, None
    try:
        with open(mat_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        albedo, normal, rough = None, None, None
        
        # Unity YAML .mat 에서 텍스처 GUID 및 이름 추출
        # 패턴: "_MainTex:", "_BumpMap:", "_MetallicGlossMap:" 등의 텍스처 블록 파싱
        import re
        # 각 텍스처 타입 키워드와 내부 파일명 힌트 추출
        tex_blocks = re.findall(r'(\w+):\s*\{[^}]*fileID:\s*\d+[^}]*\}', content)
        
        # 파일명 기반 힌트 추출 (Unity가 GUID 대신 이름을 기록하는 경우)
        name_hints = {}
        for line in content.splitlines():
            if "m_Name:" in line:
                name = line.split("m_Name:")[-1].strip()
                if name:
                    name_hints["name"] = name

        # 텍스처 디렉토리에서 일치하는 파일 찾기 (서브폴더 포함)
        def find_in_texdir(keywords):
            if not os.path.exists(texture_dir):
                return None
            for f in glob.glob(os.path.join(texture_dir, "**", "*.*"), recursive=True):
                fn = os.path.basename(f).lower()
                # 확장자가 이미지인 것만
                if not any(fn.endswith(ext) for ext in [".tga", ".png", ".jpg", ".jpeg"]):
                    continue
                if any(kw.lower() in fn for kw in keywords):
                    return f
            return None
        
        # Albedo: _MainTex, _BaseColorMap, _AlbedoMap, _Albedo
        albedo = find_in_texdir(["albedo", "diffuse", "maintex", "basemap", "_d.", "food"])
        # Normal: _BumpMap, _NormalMap
        normal = find_in_texdir(["normal", "bump", "_n."])
        # Roughness: _MetallicGlossMap, _Roughness
        rough  = find_in_texdir(["roughness", "rough", "metallic", "_r."])
        
        return albedo, normal, rough
    except Exception as e:
        print(f"    [!] .mat parse error: {e}")
        return None, None, None

def find_matched_texture(model_path, texture_dir="assets/texture", mat_dir="assets/texture"):
    """모델 파일명과 유사한 텍스처를 검색합니다. .mat 파일을 우선 시도합니다."""
    base_name = os.path.splitext(os.path.basename(model_path))[0].lower()
    model_dir = os.path.normpath(os.path.dirname(model_path))
    model_dir_parent = os.path.dirname(model_dir)
    
    # 검색 우선순위 설정
    search_dirs = []
    # 1. 모델과 같은 폴더의 texture 폴더
    search_dirs.append(os.path.join(model_dir, "texture"))
    # 2. 모델 한 단계 위 폴더의 texture 폴더
    search_dirs.append(os.path.join(model_dir_parent, "texture"))
    # 3. 모델과 같은 폴더
    search_dirs.append(model_dir)
    # 4. 기본 텍스처 폴더
    search_dirs.append(texture_dir)
    
    search_dirs = [os.path.normpath(d) for d in search_dirs if os.path.exists(d)]

    # 1. .mat 파일 탐색
    for s_dir in search_dirs:
        mat_path = os.path.join(s_dir, f"{base_name}.mat")
        if os.path.exists(mat_path):
            albedo, normal, rough = parse_unity_mat(mat_path, texture_dir)
            if albedo:
                print(f"    - [.mat] Match: {os.path.basename(mat_path)} -> {os.path.basename(albedo)}", flush=True)
                return albedo, normal, rough
    
    # 2. 파일명 기반 직접 탐색 (비재귀로 속도 향상)
    albedo, normal, rough = None, None, None
    for s_dir in search_dirs:
        for f in os.listdir(s_dir):
            if not os.path.isfile(os.path.join(s_dir, f)): continue
            fn = f.lower()
            if not any(fn.endswith(ext) for ext in [".tga", ".png", ".jpg", ".jpeg"]): continue
            
            if base_name in fn:
                full_p = os.path.join(s_dir, f)
                if any(x in fn for x in ["albedo", "diffuse", "_d", "_col"]): albedo = full_p
                elif any(x in fn for x in ["normal", "_n", "norm"]): normal = full_p
                elif any(x in fn for x in ["roughness", "rough", "_r"]): rough = full_p
        if albedo: break

    # 3. 최후의 수단: Atlas 명칭 강제 매칭
    if not albedo:
        atlas_keywords = ["food", "loot_atlas", "atlas", "texture"]
        for s_dir in search_dirs:
            for kw in atlas_keywords:
                for ext in [".tga", ".png", ".jpg"]:
                    p = os.path.join(s_dir, kw + ext)
                    if os.path.exists(p):
                        print(f"    - [Atlas] Match: {kw+ext}", flush=True)
                        albedo = p; break
                if albedo: break
            if albedo: break
                
    return albedo, normal, rough

def load_o3d_img(path):
    if not path: return None
    try:
        img_pil = Image.open(path).convert("RGB")
        return o3d.geometry.Image(np.asarray(img_pil))
    except: return None

def process_batch(asset_list, output_dir="output"):
    """여러 에셋을 하나의 GUI 세션에서 순차적으로 처리합니다."""
    if not asset_list:
        print("[!] No assets to process.")
        return

    # Blender 파일 경로 (config.json 참조)
    with open("config.json", "r") as f:
        config = json.load(f)
    blender_exe = config.get("blender", {}).get("executable", "blender")
    
    image_out_dir = os.path.join(output_dir, "image")
    os.makedirs(image_out_dir, exist_ok=True)

    def process_next_asset(idx):
        if idx >= len(asset_list):
            print("\n[*] All batch items complete. Quitting...")
            return
            
        file_path = asset_list[idx]
        base_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        
        # 변수 초기화 (NameError 방지)
        dims_str = "0,0,0"
        textures_str = ""
        poly_count = 0
        
        print(f"\n[*] Processing [{idx+1}/{len(asset_list)}]: {base_name}")
        
        # [Blender Rendering] 절대 경로 사용으로 경로 불일치 방지
        abs_file_path = os.path.abspath(file_path)
        abs_output_prefix = os.path.abspath(os.path.join(image_out_dir, file_name_without_ext))
        
        print(f"    [*] Rendering with Blender...")
        dims_str = "0,0,0"
        try:
            # subprocess.run을 사용하여 블렌더 백그라운드 렌더링 실행
            # UnicodeDecodeError 방지를 위해 encoding/errors 설정 추가
            res = subprocess.run([
                blender_exe, "--background", "--python", "bl_render.py", "--",
                "--input", abs_file_path, "--output", abs_output_prefix
            ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if "RENDER_SUCCESS" in res.stdout:
                print(f"    [*] Blender Render Complete.")
                # Dimensions & Textures 추출
                for line in res.stdout.splitlines():
                    if "DIMENSIONS:" in line:
                        dims_str = line.split("DIMENSIONS:")[1].strip()
                        print(f"    [*] Extracted Dimensions: {dims_str}")
                    elif "TEXTURES:" in line:
                        textures_str = line.split("TEXTURES:")[1].strip()
                        print(f"    [*] Extracted Textures info.")
            else:
                print(f"    [!] Blender Render Error: {res.stderr}")
        except Exception as e:
            print(f"    [!] Blender subprocess error for {base_name}: {e}")
        
        # 분석 서비스 호출 (JSON 생성 담당)
        print(f"    [*] Launching Analysis (JSON Generation)...")
        try:
            abs_output_dir = os.path.abspath(output_dir)
            analysis_cmd = [
                sys.executable, "analysis_service.py", 
                abs_file_path, 
                "--output", abs_output_dir,
                "--poly_count", str(poly_count),
                "--dimensions", dims_str,
                "--textures", textures_str
            ]
            subprocess.run(analysis_cmd, check=True)
        except Exception as e:
            print(f"    [!] Analysis service failed for {base_name}: {e}")
        
        # 재귀적으로 다음 에셋 처리 (GUI 세션이 아니므로 직접 루프 가능)
        process_next_asset(idx + 1)

    # 배치 시작
    process_next_asset(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process 3D assets with automated capture.")
    parser.add_argument("path", nargs="?", default="assets/model", help="Path to a single file or a directory containing models")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()
    
    VALID_EXTS = {".fbx", ".obj"}
    asset_list = []
    if os.path.isdir(args.path):
        for ext in ["**/*.fbx", "**/*.obj"]:
            for f in glob.glob(os.path.join(args.path, ext), recursive=True):
                # .meta, .fbx.meta 등 Unity 메타 파일 및 지원하지 않는 파일 제외
                if os.path.splitext(f)[1].lower() in VALID_EXTS:
                    asset_list.append(f)
    elif os.path.exists(args.path):
        if os.path.splitext(args.path)[1].lower() in VALID_EXTS:
            asset_list = [args.path]
        else:
            print(f"Error: File type not supported: {args.path}")
            sys.exit(1)
    else:
        print(f"Error: Path not found {args.path}")
        sys.exit(1)
        
    process_batch(sorted(asset_list), args.output)
    print("\n[Batch Processing Complete]")
