"""
Qwen2.5-VL-3B vs Qwen2.5-VL-7B 비교 스크립트
- 동일한 이미지에 대해 두 모델의 결과를 나란히 출력
- 4-bit 양자화 적용
"""

import torch
import json
import time
import argparse
from PIL import Image
from transformers import BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from classification_config import CATEGORY_PROMPTS

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_quant_config(quant_type):
    if quant_type == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quant_type == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None # FP16/BF16

def load_model_variant(model_id, quant_type):
    print(f"\n  [*] Loading {model_id} ({quant_type})...")
    t0 = time.time()
    try:
        q_config = get_quant_config(quant_type)
        
        # Qwen3-VL과 Qwen2.5-VL은 클래스가 다를 수 있으므로 자동 감지 사용
        from transformers import AutoModelForVision2Seq
        
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True
        }
        if q_config:
            load_kwargs["quantization_config"] = q_config
        else:
            load_kwargs["torch_dtype"] = torch.float16
            
        model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs)
        processor = AutoProcessor.from_pretrained(model_id)
        
        # VRAM 점유량 체크
        vram = torch.cuda.memory_allocated() / (1024**3)
        print(f"  [+] Loaded in {time.time() - t0:.2f}s (Current VRAM: {vram:.2f} GB)")
        return model, processor, vram
    except Exception as e:
        print(f"  [!] Failed to load {model_id} in {quant_type}: {e}")
        return None, None, 0

def run_vlm(image_path, model, processor, categories, file_name):
    cat_list_str = ", ".join([f'"{c}"' for c in categories])
    prompt = f"""Asset File Name: {file_name}
Analyze the provided Isometric (ISO) view image of the 3D model.

1. CATEGORY: You MUST choose EXACTLY ONE category from this strict list: {cat_list_str}. Do not invent new categories.
2. TAGS: Output a json array of 1 to 3 simple, one-word tags identifying exactly what objects are in the scene (e.g. ["fish", "dish"], ["sword"], ["bread", "loaf"]). Do not use generic words like "object" or "model".

Output strictly in this JSON format:
{{
    "category": "...",
    "tags": ["tag1", "tag2"]
}}"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    elapsed = time.time() - t0

    try:
        js = output_text.strip()
        if "```" in js:
            js = js.split("```")[1]
            if js.startswith("json"): js = js[4:]
        result = json.loads(js)
    except:
        result = {"category": "Parse Error", "tags": [], "raw": output_text}

    result["_inference_sec"] = round(elapsed, 2)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to ISO view PNG image")
    args = parser.parse_args()

    image_path = args.image_path
    file_name = os.path.basename(image_path)
    categories = list(CATEGORY_PROMPTS.keys())

    # 비교 대상 모델 리스트
    MODELS = [
        "Qwen/Qwen2-VL-2B-Instruct",      # 현재 모델
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct"
    ]
    
    QUANTS = ["4bit", "8bit", "fp16"]

    print("=" * 60)
    print(" Qwen 모델별/양자화별 통합 비교 벤치마크")
    print("=" * 60)
    print(f"  Target Image: {image_path}")

    all_results = []
    
    for model_id in MODELS:
        for q_type in QUANTS:
            # 8B 모델은 fp16으로 로드 시 VRAM 부족 가능성 큼 -> 예외 처리
            model, processor, vram = load_model_variant(model_id, q_type)
            if model:
                res = run_vlm(image_path, model, processor, categories, file_name)
                res["model"] = model_id
                res["quant"] = q_type
                res["vram_gb"] = round(vram, 2)
                all_results.append(res)
                
                # 메모리 정리
                del model, processor
                torch.cuda.empty_cache()
                time.sleep(2)
            else:
                print(f"  [!] Skipped {model_id} ({q_type}) due to loading error.")

    # 결과 테이블 출력
    print("\n" + "=" * 85)
    print(f"{'Model Name':<25} | {'Quant':<6} | {'VRAM':<7} | {'Speed':<7} | {'Category'}")
    print("-" * 85)
    for r in all_results:
        m_name = r['model'].split('/')[-1]
        print(f"{m_name:<25} | {r['quant']:<6} | {r['vram_gb']:>5.2f}G | {r['_inference_sec']:>5.2f}s | {r['category']}")
    print("=" * 85)

    out_path = "output/vlm_quant_comparison.json"
    os.makedirs("output", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n[+] Benchmark finished. Data saved to: {out_path}")

if __name__ == "__main__":
    main()
    import os; os.makedirs("output", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n[+] 결과 저장: {out_path}")

if __name__ == "__main__":
    main()
