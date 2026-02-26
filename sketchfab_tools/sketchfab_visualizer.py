import os
import pickle
import base64
import io
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.manifold import TSNE
from transformers import AutoImageProcessor, AutoModel

# 1. 설정
EMBEDDINGS_FILE = "sketchfab_tools/sketchfab_embeddings.pkl"
IMAGE_DIR = "sketchfab_data"
OUTPUT_HTML = "sketchfab_tools/style_cluster_2d.html"
MODEL_ID = "facebook/dinov2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_image_base64(path, size=(128, 128)):
    """이미지를 base64 문장으로 변환합니다 (툴팁용)."""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(size)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    except:
        return ""

def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}
    with open(EMBEDDINGS_FILE, 'rb') as f:
        return pickle.load(f)

def get_query_vector(image_path):
    """새로운 이미지의 DINOv2 벡터를 추출합니다."""
    print(f"[*] 쿼리 이미지 분석 중: {image_path}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def visualize_style_space(query_image=None):
    """
    DINOv2 벡터를 시각화합니다.
    query_image: 사용자가 입력한 새로운 사진 경로 (선택 사항)
    """
    embeddings_dict = load_embeddings()
    if not embeddings_dict:
        print("[!] 불러올 임베딩 데이터가 없습니다.")
        return

    filenames = list(embeddings_dict.keys())
    vectors = [v.squeeze() for v in embeddings_dict.values()]
    types = ["Dataset"] * len(filenames)
    image_paths = [os.path.join(IMAGE_DIR, f) for f in filenames]

    # 쿼리 이미지가 있는 경우 추가
    if query_image and os.path.exists(query_image):
        q_vec = get_query_vector(query_image)
        filenames.append("QUERY_TARGET")
        vectors.append(q_vec.squeeze())
        types.append("YOUR_IMAGE")
        image_paths.append(query_image)

    vectors = np.array(vectors)

    print(f"[*] t-SNE 계산 중 (대상: {len(filenames)}개)...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(filenames)-1), random_state=42, init='pca', learning_rate='auto')
    vectors_2d = tsne.fit_transform(vectors)

    # 툴팁용 base64 리스트 제작
    print("[*] 툴팁용 썸네일 생성 중 (시간이 조금 걸릴 수 있습니다)...")
    b64_images = [get_image_base64(p) for p in image_paths]

    # 그래프 데이터 준비
    data = {
        "x": vectors_2d[:, 0],
        "y": vectors_2d[:, 1],
        "name": [f.split("_")[-1].replace(".jpg", "") for f in filenames],
        "type": types,
        "img_b64": b64_images
    }

    fig = px.scatter(
        data, x="x", y="y", color="type",
        hover_name="name",
        color_discrete_map={"Dataset": "#636EFA", "YOUR_IMAGE": "#EF553B"},
        title="Sketchfab Medieval Style Map (Interactive)",
        labels={"x": "Style A", "y": "Style B"},
        template="plotly_dark"
    )

    # 커스텀 툴팁 설정 (이미지 포함)
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br><br><img src='data:image/jpeg;base64,%{customdata[0]}' width='150'><extra></extra>",
        customdata=np.expand_dims(data["img_b64"], axis=-1)
    )

    # 쿼리 타겟(사용자 사진) 강조
    if "YOUR_IMAGE" in types:
        fig.update_traces(
            selector=dict(name="YOUR_IMAGE"),
            marker=dict(size=25, symbol="star", line=dict(width=3, color="white"))
        )

    fig.update_traces(selector=dict(name="Dataset"), marker=dict(size=10, opacity=0.7))

    fig.update_layout(
        width=1200, height=800,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        hoverlabel=dict(bgcolor="black", font_size=12)
    )

    print(f"[*] 결과를 {OUTPUT_HTML}에 저장했습니다.")
    fig.write_html(OUTPUT_HTML)
    print(f"\n[+] 시각화 완료! 브라우저에서 확인하세요: {os.path.abspath(OUTPUT_HTML)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="시각화에 배치해볼 사용자 사진 경로")
    args = parser.parse_args()
    
    visualize_style_space(query_image=args.query)
