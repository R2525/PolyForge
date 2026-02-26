import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import json
import traceback
import re
import datetime

MISSING_LABELS_FILE = "missing_labels.json"

class MissingTagsManager:
    """
    택사노미에 없는 새로운 단어들을 추적하고 저장합니다.
    """
    def __init__(self):
        self.labels = []
        self._load()

    def _load(self):
        if os.path.exists(MISSING_LABELS_FILE):
            try:
                with open(MISSING_LABELS_FILE, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
            except:
                self.labels = []

    def log_label(self, label, source="unknown"):
        if not label: return
        label = label.lower().strip().replace('_', ' ')
        
        # 이미 존재하는지 확인 (대소문자 무시)
        for item in self.labels:
            if item["label"] == label: return

        self.labels.append({
            "label": label,
            "source": source,
            "detected_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self._save()

    def _save(self):
        with open(MISSING_LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=4, ensure_ascii=False)

missing_manager = MissingTagsManager()

# 외부 설정 및 택사노미 로드
from classification_config import (
    GENRE_CONFIG, STYLE_CONFIG, CONDITION_CONFIG, 
    CATEGORY_PROMPTS, LVIS_PROMPT_TEMPLATE
)

TAXONOMY_PATH = "lvis_taxonomy.json"

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_taxonomy():
    if os.path.exists(TAXONOMY_PATH):
        with open(TAXONOMY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# --- Stage 1: Keyword Filtering ---
def stage1_keyword_search(file_name, taxonomy, textures=None):
    """
    파일명과 텍스처 파일명에서 키워드를 추출하여 유력한 카테고리를 추측합니다.
    """
    if not taxonomy: return [], []
    
    # 분석 대상 텍스트 수집 (파일명 + 텍스처 파일명들)
    search_texts = [file_name.lower().replace('-', '_').replace(' ', '_')]
    if textures:
        if isinstance(textures, str):
            search_texts.extend([t.lower().replace('-', '_').replace(' ', '_') for t in textures.split(',')])
        else:
            search_texts.extend([t.lower().replace('-', '_').replace(' ', '_') for t in textures])

    matched_categories = []
    matched_tags = []
    
    for category, tags in taxonomy["taxonomy"].items():
        # 카테고리명을 구성하는 단어들 (예: "Food", "Drink")
        cat_words = re.findall(r'[a-z]+', category.lower())
        
        for text in search_texts:
            # 카테고리 단어 중 하나라도 포함되면 매칭 (3글자 이상만)
            for cw in cat_words:
                if len(cw) > 2 and cw in text:
                    matched_categories.append(category)
                    break
                
            for tag in tags:
                tag_clean = tag.lower().replace(' ', '_')
                pattern = rf"(^|[^a-z]){tag_clean}s?([^a-z]|$)"
                if re.search(pattern, text):
                    matched_categories.append(category)
                    matched_tags.append(tag_clean)
    
    # [추가] 분석 대상들에서 단어를 추출하되 택사노미에 없는 경우 로깅
    all_known_tags = []
    for t_list in taxonomy["taxonomy"].values():
        all_known_tags.extend([t.lower().replace(' ', '_') for t in t_list])
    all_known_cats = [c.lower().replace(' & ', '_').replace(' ', '_') for c in taxonomy["taxonomy"].keys()]
    
    for text in search_texts:
        words = re.findall(r'[a-z]+', text)
        for w in words:
            if len(w) > 2 and w not in all_known_tags and w not in all_known_cats:
                if w not in ["the", "and", "model", "fbx", "obj", "view", "part", "diff", "spec", "norm", "metal", "rough"]:
                    missing_manager.log_label(w, source=f"keyword_search: {text}")
                
    return list(set(matched_categories)), list(set(matched_tags))

# --- VLM Fallback Helper (Stage 4) ---
def run_vlm_analysis(image_path, category_name, labels, device):
    """
    신뢰도가 낮을 때 Qwen3-VL을 사용하여 정밀 재분석
    """
    print(f"    [VLM] Low confidence in '{category_name}'. Starting VLM Fallback...")
    
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True) if "cuda" in device else None
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            quantization_config=bnb_config,
            device_map="auto" if "cuda" in device else None,
            trust_remote_code=True
        )
        if "cuda" not in device:
            model = model.to(device)
            
        processor = AutoProcessor.from_pretrained(model_id)

        # labels가 문자열 리스트인지 보장
        safe_labels = []
        for l in labels:
            if isinstance(l, list):
                safe_labels.append(", ".join([str(x) for x in l]))
            else:
                safe_labels.append(str(l))
        
        labels_str = ", ".join(safe_labels)
        print(f"    [VLM] Joining labels: {safe_labels}", flush=True)
        
        prompt = f"Observe this 3D model carefully. Which of the following tags best describes its '{category_name}'? \nTags: {labels_str}\nAnswer with ONLY the most accurate tag name from the list and a very brief reason."
        
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

        # generate 호출 (격리 테스트에서 검증된 방식)
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(f"    [VLM] Result: {output_text}", flush=True)
        
        vlm_tag = None
        for label in labels:
            if label.lower() in output_text.lower():
                vlm_tag = label
                break
        
        if not vlm_tag:
            # 쉼표나 마침표 제거 후 첫 단어 등을 추출하거나 전체 출력 텍스트 로깅 고려
            # 여기서는 VLM이 내뱉은 단어가 택사노미에 없을 가능성이 높으므로 로깅
            potential_new_tag = output_text.split('.')[0].split(',')[0].strip()
            if len(potential_new_tag) > 2:
                missing_manager.log_label(potential_new_tag, source="vlm_output")
            vlm_tag = labels[0] # 기본값

        return vlm_tag, output_text
    except Exception as e:
        print(f"    [VLM] Failed: {e}")
        return None, str(e)
    finally:
        if 'model' in locals():
            del model
            torch.cuda.empty_cache()

def run_analysis(file_path: str, output_root: str, category: str, poly_count: int = 0, dimensions: list = None, textures: list = None):
    """
    Hierarchical Analysis Pipeline
    """
    try:
        device = get_device()
        print(f"[*] Analysis Service | device: {device}")

        file_name = os.path.basename(file_path)
        base_stem = os.path.splitext(file_name)[0]
        taxonomy = load_taxonomy()

        img_dir  = os.path.join(output_root, "image")
        json_dir = os.path.join(output_root, "json")
        os.makedirs(json_dir, exist_ok=True)

        # ── 0. 모델 Load ──────────────────────────
        print("[*] Loading CLIP model...")
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        dino_model     = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

        # ── 1. 이미지 Load ────────────────────────
        analysis_views = []
        saved_view_paths = []
        for i in range(1, 11):
            image_path = os.path.join(img_dir, f"{base_stem}_view{i}.png")
            if not os.path.exists(image_path): continue
            img = Image.open(image_path).convert("RGB")
            analysis_views.append(img)
            saved_view_paths.append(image_path)

        if not analysis_views:
            print(f"    [!] No images found in {img_dir} for {base_stem}. Check if rendering succeeded.")
            return

        pixel_values = torch.cat([clip_processor(images=img, return_tensors="pt")["pixel_values"] for img in analysis_views], dim=0).to(device)

        detected_tags = {}
        tag_candidates = {}

        # ── [STAGE 1] Keyword Matching ───────────
        kw_categories, kw_tags = stage1_keyword_search(file_name, taxonomy, textures=textures)
        if kw_categories:
            print(f"[*] Stage 1 (Keywords matched): Cats={kw_categories}, Tags={kw_tags}")
        else:
            print(f"[*] Stage 1 (Keywords): No match found for '{file_name}'")

        # ── [STAGE 2] Top-level Category CLIP ─────
        cat_labels = list(CATEGORY_PROMPTS.keys())
        cat_prompts = list(CATEGORY_PROMPTS.values())
        
        text_inputs = clip_processor(text=cat_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(pixel_values=pixel_values, **text_inputs)
            probs = outputs.logits_per_image.softmax(dim=-1).mean(dim=0)
            
            # 키워드 매칭된 카테고리에 강력한 가중치 부여 (0.5로 상향)
            for i, label in enumerate(cat_labels):
                if label in kw_categories:
                    probs[i] += 0.5 
            
            probs = probs / probs.sum() # Normalize
            top_vals, top_indices = torch.topk(probs, min(3, len(cat_labels)))
            
        top_3_categories = [cat_labels[idx.item()] for idx in top_indices]
        print(f"[*] Stage 2 (Category CLIP): Top 1 Result -> {top_3_categories[0]} ({top_vals[0].item():.4f})")
        
        detected_tags["category"] = top_3_categories[0]
        tag_candidates["category"] = [
            {"label": cat, "confidence": round(val.item(), 4)} 
            for val, cat in zip(top_vals, top_3_categories)
        ]

        # ── [STAGE 3] Hierarchical LVIS Analysis (Restored) ──
        print(f"[*] Stage 3 (Fine-grained LVIS Tagging)...")
        potential_tags = []
        for cat in top_3_categories:
            potential_tags.extend(taxonomy["taxonomy"].get(cat, []))
        
        potential_tags = list(set(potential_tags))[:200]
        
        if potential_tags:
            lvis_prompts = [LVIS_PROMPT_TEMPLATE.format(tag.replace('_', ' ')) for tag in potential_tags]
            text_inputs_lvis = clip_processor(text=lvis_prompts, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = clip_model(pixel_values=pixel_values, **text_inputs_lvis)
                lvis_probs = outputs.logits_per_image.softmax(dim=-1).mean(dim=0)
                
                k = min(5, len(potential_tags))
                top_lvis_vals, top_lvis_indices = torch.topk(lvis_probs, k)
                
                best_lvis_tag = potential_tags[top_lvis_indices[0].item()]
                best_lvis_conf = top_lvis_vals[0].item()

                # Stage 1 키워드 우선순위 복원
                if kw_tags:
                    for kw in kw_tags:
                        # kw가 potential_tags에 있거나 혹은 카테고리가 일치하는지 확인
                        # 여기서는 단순하게 kw가 taxonomy 전체에 있는지 확인하여 신뢰도 보정
                        for cat_name, tag_list in taxonomy["taxonomy"].items():
                            if kw in [t.lower().replace(' ', '_') for t in tag_list]:
                                if cat_name in top_3_categories:
                                    print(f"[*] Keyword Priority: Overriding '{best_lvis_tag}' with matched keyword '{kw}'")
                                    best_lvis_tag = kw
                                    best_lvis_conf = 1.0 # VLM Skip
                                    break
                
                detected_tags["object_type"] = best_lvis_tag
                tag_candidates["object_type"] = [
                    {"label": potential_tags[idx.item()], "confidence": round(val.item(), 4)} 
                    for val, idx in zip(top_lvis_vals, top_lvis_indices)
                ]

                # ── [STAGE 4] VLM Fallback (Restored) ──
                if best_lvis_conf < 0.5:
                    print(f"    [VLM] Low confidence ({best_lvis_conf:.4f}). Triggering Stage 4...")
                    vlm_labels = [c["label"] for c in tag_candidates["object_type"]]
                    vlm_tag, vlm_reason = run_vlm_analysis(saved_view_paths[0], "object_type", vlm_labels, device)
                    if vlm_tag:
                        detected_tags["object_type"] = vlm_tag
                        detected_tags["object_type_vlm_reason"] = vlm_reason
        else:
            print(f"    [!] No potential tags found for categories: {top_3_categories}")

        # ── 기타 고정 카테고리 (Genre, Style, Condition) ──
        for cat_name, config in [("genre", GENRE_CONFIG), ("visual_style", STYLE_CONFIG), ("condition", CONDITION_CONFIG)]:
            labels, prompts = config["labels"], config["prompts"]
            text_in = clip_processor(text=prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = clip_model(pixel_values=pixel_values, **text_in)
                p = out.logits_per_image.softmax(dim=-1).mean(dim=0)
                idx = torch.argmax(p).item()
                detected_tags[cat_name] = labels[idx]
                tag_candidates[cat_name] = [{"label": labels[i], "confidence": round(p[i].item(), 4)} for i in range(len(labels))]

        # ── DINO Vector & JSON Save (기존 유지) ──
        print("[*] DINO Style Vector...")
        all_embs = []
        for img in analysis_views:
            inputs_dino = dino_processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = dino_model(**inputs_dino)
                all_embs.append(out.last_hidden_state[0, 0, :].cpu())
        final_vector = torch.stack(all_embs).mean(dim=0).numpy()

        db_entry = {
            "asset_id": file_name,
            "category": detected_tags.get("category", "unknown"),
            "object_type": detected_tags.get("object_type", "unknown"),
            "dimensions": dimensions, # [x, y, z] in meters
            "poly_count": poly_count,
            "tags": detected_tags,
            "tag_candidates": tag_candidates,
            "full_style_vector": final_vector.tolist(),
        }
        out_json = os.path.join(json_dir, f"{base_stem}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(db_entry, f, indent=4, ensure_ascii=False)
        print(f"[*] Analysis Complete → {out_json}", flush=True)

    except Exception as e:
        print(f"\n[!] ERROR: {e}", flush=True)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--category", default="root")
    parser.add_argument("--poly_count", type=int, default=0)
    parser.add_argument("--dimensions", default="0,0,0", help="x,y,z size in meters")
    parser.add_argument("--textures", default="", help="comma separated texture filenames")
    args = parser.parse_args()
    
    # dimensions 파싱
    try:
        dim_list = [float(x) for x in args.dimensions.split(",")]
    except:
        dim_list = [0.0, 0.0, 0.0]

    # textures 파싱
    tex_list = args.textures.split(",") if args.textures else []

    run_analysis(args.file_path, args.output, args.category, args.poly_count, dim_list, textures=tex_list)
