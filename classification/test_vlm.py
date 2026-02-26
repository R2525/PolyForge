import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

def test_vlm():
    model_id = "Qwen/Qwen2-VL-2B-Instruct" # Using Qwen2-VL as per standard
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model on {device}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device_map="auto" if "cuda" in device else None
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    prompt = "Describe this image."
    # Mock image path (use one from output)
    image_path = "debug_output/image/sausages1_view4.png"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    print("Attempting generate...")
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        print("Success with **inputs!")
    except Exception as e:
        print(f"Failed with **inputs: {e}")
        
    try:
        gen_kwargs = {k: v for k, v in inputs.items()}
        gen_kwargs["max_new_tokens"] = 128
        generated_ids = model.generate(**gen_kwargs)
        print("Success with items copy!")
    except Exception as e:
        print(f"Failed with items copy: {e}")

if __name__ == "__main__":
    test_vlm()
