import os
import sys
import subprocess
import torch
import json
import argparse
import traceback
import time
import glob
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import open3d as o3d
import open3d.visualization.rendering as rendering
from categorize_lvis import LVIS_TAXONOMY_DICT

# Ensure Open3D can run headlessly if needed (optional depending on env, but good practice)
os.environ["O3D_WEBRTC_PORT"] = "8888"

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def load_config():
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("blender", {}).get("executable", "blender")
    return "blender"

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_categories():
    try:
        from classification_config import CATEGORY_PROMPTS, GENRE_TAXONOMY, STYLE_TAXONOMY, SURFACE_CONDITION_TAXONOMY
        return list(CATEGORY_PROMPTS.keys()), GENRE_TAXONOMY, STYLE_TAXONOMY, SURFACE_CONDITION_TAXONOMY
    except ImportError:
        print("[!] classification_config.py not found. Using fallback categories.")
        cat = ["Food & Drink", "Weapons & Military", "Furniture & Home", "Animals & Pets", "Other"]
        genre = {"Fantasy": "", "Sci-Fi": "", "Modern": "", "Post-Apocalyptic": "", "Military": ""}
        style = {"Photorealistic": "", "Stylized": "", "Low-Poly": ""}
        surface = {"Rusty": "", "Dusty": "", "Clean": ""}
        return cat, genre, style, surface

def render_fast_blender(blender_exe, fbx_path, output_prefix):
    """Blender Workbench 모델을 사용하여 고속 렌더링 수행 및 메타데이터 추출"""
    script_path = os.path.join(os.path.dirname(__file__), "bl_render_fast.py")
    abs_fbx = os.path.abspath(fbx_path)
    abs_prefix = os.path.abspath(output_prefix)
    
    cmd = [
        blender_exe, "--background", "--python", script_path, "--",
        "--input", abs_fbx, "--output", abs_prefix
    ]
    
    mesh_info = {"faces": 0, "vertices": 0, "size": [0, 0, 0], "textures": []}
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = res.stdout
        
        # MESH_DATA 파싱: "MESH_DATA: faces=..., vertices=..., size_x=..., size_y=..., size_z=..."
        for line in stdout.splitlines():
            if "MESH_DATA:" in line:
                try:
                    parts = line.split("MESH_DATA:")[1].split(",")
                    for p in parts:
                        k, v = p.strip().split("=")
                        if k == "faces": mesh_info["faces"] = int(v)
                        elif k == "vertices": mesh_info["vertices"] = int(v)
                        elif k == "size_x": mesh_info["size"][0] = float(v)
                        elif k == "size_y": mesh_info["size"][1] = float(v)
                        elif k == "size_z": mesh_info["size"][2] = float(v)
                except:
                    pass
            elif "TEXTURES:" in line:
                try:
                    tex_str = line.split("TEXTURES:")[1].strip()
                    mesh_info["textures"] = [t.strip() for t in tex_str.split(",") if t.strip()]
                except:
                    pass
            elif "NORMAL_MAP:" in line:
                mesh_info["normal_map"] = line.split("NORMAL_MAP:")[1].strip()
            elif "ROUGHNESS_MAP:" in line:
                mesh_info["roughness_map"] = line.split("ROUGHNESS_MAP:")[1].strip()
            elif "TEXTURE_STATUS:" in line:
                try:
                    mesh_info["texture_status"] = line.split("TEXTURE_STATUS:")[1].strip()
                except:
                    pass
        
        success = "RENDER_SUCCESS" in stdout or os.path.exists(f"{abs_prefix}_view4.png")
        return success, mesh_info
    except subprocess.CalledProcessError as e:
        print(f"      [!] Blender render error: {e.stderr}")
        return False, mesh_info

# ---------------------------------------------------------
# AI Analysis
# ---------------------------------------------------------

def extract_dino_features(image_path, device, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[0, 0, :]
    return features.cpu().numpy().tolist()

def map_tags_to_lvis_category(tags, current_category):
    """VLM이 출력한 태그들을 기반으로 LVIS Taxonomy와 대조하여 카테고리 보정"""
    if not tags: return current_category
    
    match_scores = {cat: 0 for cat in LVIS_TAXONOMY_DICT.keys()}
    
    for tag in tags:
        tag_clean = tag.lower().strip()
        for cat, keywords in LVIS_TAXONOMY_DICT.items():
            if tag_clean in keywords:
                match_scores[cat] += 2 # 명확한 매칭 가점
            elif any(kw in tag_clean or tag_clean in kw for kw in keywords):
                match_scores[cat] += 1 # 부분 매칭 가점
                
    # 점수가 가장 높은 카테고리 추출
    max_score = max(match_scores.values())
    if max_score > 0:
        best_cats = [cat for cat, score in match_scores.items() if score == max_score]
        # 만약 동점인 카테고리 중 현재 카테고리가 있다면, AI의 판단을 존중하여 유지
        if current_category in best_cats:
            return current_category
        return best_cats[0]
        
    return current_category

def run_vlm_analysis(image_paths, file_name, mesh_info, cat_tuple, device, model, processor):
    categories, genre_tax, style_tax, surface_tax = cat_tuple
    
    cat_list_str = ", ".join([f'"{c}"' for c in categories])
    genre_list_str = ", ".join([f'"{g}"' for g in genre_tax.keys()])
    style_list_str = ", ".join([f'"{s}"' for s in style_tax.keys()])
    surface_list_str = ", ".join([f'"{s}"' for s in surface_tax.keys()])
    
    genre_desc = "\n".join([f"- {g}: {d}" for g, d in genre_tax.items()])
    style_desc = "\n".join([f"- {s}: {d}" for s, d in style_tax.items()])
    surface_desc = "\n".join([f"- {s}: {d}" for s, d in surface_tax.items()])

    # LVIS Taxonomy 힌트 생성
    lvis_hints = []
    for cat in categories:
        if cat in LVIS_TAXONOMY_DICT:
            keywords = ", ".join(LVIS_TAXONOMY_DICT[cat][:10]) # 상위 10개만
            lvis_hints.append(f"- {cat}: related to {keywords}...")
    lvis_hint_str = "\n".join(lvis_hints)

    poly_count = mesh_info.get("faces", 0)
    size = mesh_info.get("size", [0, 0, 0])
    size_str = f"{size[0]:.2f}m x {size[1]:.2f}m x {size[2]:.2f}m"
    textures = mesh_info.get("textures", [])
    texture_str = ", ".join(textures) if textures else "None"

    prompt_stage1 = f"""
    Asset Details:
    - File Name: {file_name}
    - Physical Size: {size_str}
    - Used Textures: {texture_str}
    
    Analyze the provided 4 views (Front, Right, Back, Isometric) of the 3D model.
    Focus on extracting visual features, styles, and surface conditions.
    
    1. STYLES: Identify the Top 4 most relevant styles from: {style_list_str}.
       For each style, provide a "name" and a "score" (a probability between 0.0 and 1.0 indicating relevance).
       Reference:
       {style_desc}
    2. SURFACE CONDITIONS (Material/State): Select 0 to 4 most relevant conditions from: {surface_list_str}.
       Reference:
       {surface_desc}
       If none apply, return an empty list [].
    3. TAGS: Output 1 to 4 simple, one-word tags describing the object's function or form (e.g., rusty, armor, sword).
    
    Output strictly in this JSON format:
    {{
        "styles": [
            {{"name": "...", "score": 0.0}},
            {{"name": "...", "score": 0.0}}
        ],
        "surface_conditions": ["..."],
        "tags": ["tag1", "tag2"]
    }}
    """
    
    # -----------------------------------------------------
    # STAGE 1: Visual Feature Extraction
    # -----------------------------------------------------
    image_contents = [{"type": "image", "image": img} for img in image_paths]
    messages_stage1 = [
        {
            "role": "user",
            "content": image_contents + [{"type": "text", "text": prompt_stage1}],
        }
    ]
    
    text_st1 = processor.apply_chat_template(messages_stage1, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages_stage1)
    inputs_st1 = processor(
        text=[text_st1],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids_st1 = model.generate(**inputs_st1, max_new_tokens=256)
        generated_ids_trimmed_st1 = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_st1.input_ids, generated_ids_st1)
        ]
        output_text_st1 = processor.batch_decode(
            generated_ids_trimmed_st1, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    try:
        json_str = output_text_st1.strip()
        if "```" in json_str:
            parts = json_str.split("```")
            for p in parts:
                if "{" in p and "}" in p:
                    json_str = p
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                    break
        res_st1 = json.loads(json_str)
    except:
        res_st1 = {"styles": [{"name": "Other", "score": 1.0}], "surface_conditions": [], "tags": []}

    styles_data = res_st1.get("styles", [])
    if not isinstance(styles_data, list):
        styles_data = []

    valid_styles = []
    for s in styles_data:
        name = s.get("name", "Other")
        if name not in style_tax:
            name = "Other"
        try:
            score = float(s.get("score", 0.0))
        except:
            score = 0.0
        valid_styles.append({"name": name, "score": score})
    
    valid_styles = sorted(valid_styles, key=lambda x: x["score"], reverse=True)

    if not valid_styles:
         old_style = res_st1.get("style", "Other")
         if old_style not in style_tax: old_style = "Other"
         valid_styles = [{"name": old_style, "score": 1.0}]

    top_style = valid_styles[0]["name"] if valid_styles else "Other"

    extracted_surface = res_st1.get("surface_conditions", [])
    extracted_tags = res_st1.get("tags", [])

    # Compose the descriptive phrase
    features = extracted_surface + extracted_tags
    if top_style and top_style != "Other":
        features.append(f"{top_style} style")
    descriptive_phrase = ", ".join(features) if features else "Unknown object"

    # -----------------------------------------------------
    # STAGE 2: Text-Based Category & Genre Inference
    # -----------------------------------------------------
    prompt_stage2 = f"""
    Based on the visual analysis of a 3D model, the following descriptive phrase was generated:
    "{descriptive_phrase}"
    
    Asset Metadata:
    - Physical Size: {size_str}
    
    Using the descriptive phrase and metadata, infer the most appropriate Category and Genre.
    
    1. CATEGORY: Choose EXACTLY ONE from: {cat_list_str}.
       Reference Keywords (LVIS):
       {lvis_hint_str}
    2. GENRES: Identify the Top 3 most relevant genres from: {genre_list_str}.
       For each genre, provide a "name" and a "score" (a probability between 0.0 and 1.0 indicating relevance).
       Reference:
       {genre_desc}
       (Note: e.g., "Rusty, blood-stained armor" should map to Fantasy or Military depending on context, while "Clean bread" is Modern).
    
    Output strictly in this JSON format:
    {{
        "category": "...",
        "genres": [
            {{"name": "...", "score": 0.0}},
            {{"name": "...", "score": 0.0}},
            {{"name": "...", "score": 0.0}}
        ]
    }}
    """

    messages_stage2 = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_stage2}],
        }
    ]
    
    text_st2 = processor.apply_chat_template(messages_stage2, tokenize=False, add_generation_prompt=True)
    inputs_st2 = processor(
        text=[text_st2],
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids_st2 = model.generate(**inputs_st2, max_new_tokens=256)
        generated_ids_trimmed_st2 = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_st2.input_ids, generated_ids_st2)
        ]
        output_text_st2 = processor.batch_decode(
            generated_ids_trimmed_st2, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    try:
        json_str = output_text_st2.strip()
        if "```" in json_str:
            parts = json_str.split("```")
            for p in parts:
                if "{" in p and "}" in p:
                    json_str = p
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                    break
        res_st2 = json.loads(json_str)
        
        # Extract and validate genres
        genres_data = res_st2.get("genres", [])
        if not isinstance(genres_data, list):
            genres_data = []

        valid_genres = []
        for g in genres_data:
            name = g.get("name", "Other")
            if name not in genre_tax:
                name = "Other"
            try:
                score = float(g.get("score", 0.0))
            except:
                score = 0.0
            valid_genres.append({"name": name, "score": score})
        
        valid_genres = sorted(valid_genres, key=lambda x: x["score"], reverse=True)

        if not valid_genres:
             old_genre = res_st2.get("genre", "Other")
             if old_genre not in genre_tax: old_genre = "Other"
             valid_genres = [{"name": old_genre, "score": 1.0}]

        top_genre = valid_genres[0]["name"] if valid_genres else "Other"

        # Merge results
        final_res = {
            "category": res_st2.get("category", "Other"),
            "genre": top_genre,
            "top_genres": valid_genres,
            "style": top_style,
            "top_styles": valid_styles,
            "surface_conditions": extracted_surface,
            "tags": extracted_tags
        }
        
        # Python-side Refinement using LVIS tags
        vlm_cat = final_res["category"]
        refined_cat = map_tags_to_lvis_category(final_res["tags"], vlm_cat)
        final_res["category"] = refined_cat

        if final_res.get("category") not in categories: final_res["category"] = "Other"
            
        return final_res
    except:
        return {"category": "Other", "genre": "Other", "top_genres": [{"name": "Other", "score": 1.0}], "style": top_style, "top_styles": valid_styles, "surface_conditions": extracted_surface, "tags": extracted_tags}


# ---------------------------------------------------------
# Main Pipeline Workflow
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to FBX file or directory")
    parser.add_argument("--output_dir", default="output/iso_analysis", help="Output directory")
    args = parser.parse_args()

    blender_exe = load_config()
    device = get_device()
    cat_data = load_categories() # (cat_list, genre_dict, style_dict)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    image_out_dir = os.path.abspath(os.path.join("output/image_fast"))
    if not os.path.exists(image_out_dir):
        os.makedirs(image_out_dir)

    def get_best_texture_for_dino(tex_list):
        if not tex_list: return None
        # Prioritize based on keywords
        albedo_keywords = ["alb", "diff", "base", "color", "main"]
        skip_keywords = ["nrm", "norm", "bump", "rough", "gloss", "spec", "met", "rma", "ao", "emm"]
        
        # 1. Best candidates
        for t in tex_list:
            lower = t.lower()
            if any(k in lower for k in albedo_keywords) and not any(k in lower for k in skip_keywords):
                return t
        
        # 2. Any that doesn't look like a technical map
        for t in tex_list:
            lower = t.lower()
            if not any(k in lower for k in skip_keywords):
                return t
        
        # 3. Fallback to first if nothing else
        return tex_list[0]

    target_files = []
    if os.path.isdir(args.input_path):
        target_files.extend(glob.glob(os.path.join(args.input_path, "*.fbx")))
    elif os.path.isfile(args.input_path) and args.input_path.lower().endswith('.fbx'):
        target_files.append(args.input_path)
    
    if not target_files:
        print(f"[!] No .fbx files found in: {args.input_path}")
        return

    print(f"\n{'='*50}")
    print(f"[*] Starting Multi-Stage Pipeline on {len(target_files)} assets.")
    print(f"{'='*50}\n")
    
    asset_metadata = {} # fbx_name -> mesh_info
    asset_images = {}   # fbx_name -> [view1, view2, view3, view4]
    successful_assets = []
    failed_assets = []
    global_start = time.time()

    # -----------------------------------------------------
    # STAGE 1: Fast Pre-Rendering & Metadata Extraction
    # -----------------------------------------------------
    print(f"[*] STAGE 1: Fast Rendering & Metadata Extraction")
    for idx, fbx_path in enumerate(target_files, 1):
        base_name = os.path.basename(fbx_path)
        base_stem = os.path.splitext(base_name)[0]
        
        print(f"    [{idx}/{len(target_files)}] Processing {base_name}...")
        image_prefix = os.path.join(image_out_dir, base_stem)
        success, mesh_info = render_fast_blender(blender_exe, fbx_path, image_prefix)
        
        if success:
            asset_metadata[base_name] = mesh_info
            # 4개 뷰 경로 저장
            asset_images[base_name] = [f"{image_prefix}_view{i}.png" for i in range(1, 5)]
        else:
            print(f"      [!] Process failed for {base_name}")
            failed_assets.append(base_name)

    print(f"[+] Stage 1 Complete.\n")

    # -----------------------------------------------------
    # STAGE 2: AI Analysis (DINO + Multi-View Qwen3-VL)
    # -----------------------------------------------------
    if asset_images:
        print(f"[*] STAGE 2: AI Analysis ({len(asset_images)} assets)")
        t0 = time.time()
        print("    [*] Pre-loading AI models to GPU...")
        try:
            dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
            
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
            from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
            
            model_id = "Qwen/Qwen3-VL-2B-Instruct"
            vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
            )
            vlm_processor = Qwen3VLProcessor.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            print(f"[!] Failed to load AI models: {e}")
            sys.exit(1)
        
        stage2_start = time.time()
        for count, (base_name, images) in enumerate(asset_images.items(), 1):
            base_stem = os.path.splitext(base_name)[0]
            print(f"    [{count}/{len(asset_images)}] Analyzing {base_stem}...")
            
            # DINO (텍스처 이미지 우선, 없으면 ISO 뷰 사용)
            mesh_info = asset_metadata.get(base_name, {})
            textures = mesh_info.get("textures", [])
            
            dino_vec = []
            dino_mode = "unknown"
            
            # 텍스처가 있다면 텍스처로 DINO 특징 추출
            tex_path = get_best_texture_for_dino(textures)
            if tex_path and os.path.exists(tex_path):
                print(f"      [*] Using texture for DINO: {tex_path}")
                try:
                    dino_vec = extract_dino_features(tex_path, device, dino_model, dino_processor)
                    dino_mode = "texture"
                except Exception as e:
                    print(f"      [!] Failed to extract DINO from texture {tex_path}: {e}")
            
            # 텍스처 추출 실패 시 ISO 렌더링 이미지 사용 (Fallback)
            if not dino_vec:
                iso_path = images[3]
                print(f"      [*] Fallback: Using ISO view for DINO: {iso_path}")
                try:
                    dino_vec = extract_dino_features(iso_path, device, dino_model, dino_processor)
                    dino_mode = "iso_fallback"
                except Exception as e:
                    print(f"      [!] Failed to extract DINO from ISO view: {e}")
                    dino_vec = []
                    dino_mode = "failed"
            
            # VLM Analysis (Multi-View + Metadata)
            try:
                mesh_info = asset_metadata.get(base_name, {})
                vlm_res = run_vlm_analysis(images, base_name, mesh_info, cat_data, device, vlm_model, vlm_processor)
            except Exception as e:
                print(f"      [!] VLM failed: {e}")
                vlm_res = {"category": "Other", "genre": "Other", "style": "Other", "tags": []}
            
            # Save final metadata in asset_metadata for summary
            asset_metadata[base_name].update({
                "category": vlm_res.get("category"),
                "genre": vlm_res.get("genre"),
                "top_genres": vlm_res.get("top_genres", []),
                "style": vlm_res.get("style"),
                "top_styles": vlm_res.get("top_styles", []),
                "surface_conditions": vlm_res.get("surface_conditions", []),
                "tags": vlm_res.get("tags", []),
                "texture_status": mesh_info.get("texture_status", "UNKNOWN")
            })

            # Save JSON
            clean_mesh_info = {
                "faces": mesh_info.get("faces", 0),
                "vertices": mesh_info.get("vertices", 0),
                "size": mesh_info.get("size", [0, 0, 0]),
                "texture_status": mesh_info.get("texture_status", "UNKNOWN"),
                "normal_map": mesh_info.get("normal_map"),
                "roughness_map": mesh_info.get("roughness_map")
            }
            if "textures" in mesh_info:
                clean_mesh_info["textures"] = mesh_info["textures"]

            final_res = {
                "asset_id": base_name,
                "images": images,
                "mesh_info": clean_mesh_info,
                "category": vlm_res.get("category"),
                "top_genres": vlm_res.get("top_genres", []),
                "style": vlm_res.get("style"),
                "top_styles": vlm_res.get("top_styles", []),
                "surface_conditions": vlm_res.get("surface_conditions", []),
                "tags": vlm_res.get("tags", []),
                "genre": vlm_res.get("genre", "Other"), # Legacy fallback
                "dino_mode": dino_mode,
                "dino_vector": dino_vec
            }
            
            out_json = os.path.join(args.output_dir, f"{base_stem}_analysis.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(final_res, f, indent=4, ensure_ascii=False)
                
            successful_assets.append(base_name)
            
        print(f"[+] Stage 2 Complete.\n")
    else:
        print("[!] No images successfully rendered. Skipping AI Analysis.")

    # -----------------------------------------------------
    # Generate Summary Report
    # -----------------------------------------------------
    summary_path = os.path.join(args.output_dir, "pipeline_summary.txt")
    total_elapsed = time.time() - global_start
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(" Multi-View Metadata-Informed VLM Pipeline Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Time: {total_elapsed:.2f}s | Assets: {len(target_files)} | Success: {len(successful_assets)}\n\n")
        
        f.write("-"*98 + "\n")
        f.write(f"{'Asset Name':<25} | {'Category':<15} | {'Genre':<10} | {'Surface':<15} | {'TexStatus':<9} | {'Poly':<10}\n")
        f.write("-"*98 + "\n")
        for asset in successful_assets:
            m = asset_metadata.get(asset, {})
            surf = ", ".join(m.get("surface_conditions", []))[:15]
            tex_stat = m.get("texture_status", "UNKNOWN")
            f.write(f"{asset[:25]:<25} | {m.get('category', 'N/A')[:15]:<15} | {m.get('genre', 'N/A')[:10]:<10} | {surf:<15} | {tex_stat:<9} | {m.get('faces', 0):>10,}\n")
            
        if failed_assets:
            f.write("\n" + "-"*60 + "\n")
            f.write(" Failed Assets\n")
            f.write("-"*60 + "\n")
            for asset in failed_assets:
                f.write(f" - {asset}\n")
                
    print(f"\n[*] Pipeline finished successfully. Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
