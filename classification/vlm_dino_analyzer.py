import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import json
import traceback

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def extract_dino_features(image_path, device, model, processor):
    """DINOv2를 사용하여 768차원 피처 벡터 추출"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Class token [CLS] 추출
        features = outputs.last_hidden_state[0, 0, :]
        
    return features.cpu().numpy().tolist()

def run_vlm_analysis(image_path, file_name, device, model, processor):
    """Qwen2-VL을 사용하여 요약 및 카테고리 분류"""
    prompt = f"""
    Asset File Name: {file_name}
    Analyze the provided image and the file name.
    1. Provide a one-sentence concise summary of the object in the image.
    2. Suggest the most appropriate category for this object.
    
    Output format (JSON):
    {{
        "summary": "one sentence summary here",
        "category": "category name here"
    }}
    """
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # 전처리
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 생성
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    # JSON 파싱 시도
    try:
        # ```json ... ``` 블록 제거 시도
        json_str = output_text.strip()
        if "```" in json_str:
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        
        return json.loads(json_str)
    except:
        return {"summary": output_text, "category": "unknown", "raw_output": output_text}

def main():
    parser = argparse.ArgumentParser(description="VLM-DINO Independent Analyzer")
    parser.add_argument("model_path", help="Path to the 3D model file (for context)")
    parser.add_argument("image_path", help="Path to the rendered image")
    parser.add_argument("--output", default="analysis_result.json", help="Output JSON path")
    args = parser.parse_args()

    device = get_device()
    print(f"[*] Starting Analysis on {device}...")

    try:
        # 1. DINOv2 모델로드
        print("[*] Loading DINOv2...")
        dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

        # 2. Qwen2-VL 모델로드
        print("[*] Loading Qwen2-VL...")
        vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
        )
        vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

        # 3. 피처 추출
        print("[*] Extracting DINO features...")
        dino_vector = extract_dino_features(args.image_path, device, dino_model, dino_processor)

        # 4. VLM 분석
        print("[*] Running VLM semantic analysis...")
        vlm_result = run_vlm_analysis(args.image_path, os.path.basename(args.model_path), device, vlm_model, vlm_processor)

        # 5. 결과 저장
        result = {
            "model_name": os.path.basename(args.model_path),
            "image_path": args.image_path,
            "dino_vector": dino_vector,
            "vlm_analysis": vlm_result
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"[*] Analysis Complete! Result saved to: {args.output}")

    except Exception as e:
        print(f"[!] Critical Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
