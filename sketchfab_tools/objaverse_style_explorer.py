import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import torch
from PIL import Image
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoImageProcessor, AutoModel
import base64

# --- 기본 설정 ---
st.set_page_config(page_title="Objaverse Mass Style Explorer", layout="wide", page_icon="📦")

EMBEDDINGS_FILE = "sketchfab_tools/objaverse_mass_embeddings.pkl"
IMAGE_DIR = "objaverse_mass_data"
MODEL_ID = "facebook/dinov2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_dino_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    return processor, model

@st.cache_data
def load_data(mtime):
    if not os.path.exists(EMBEDDINGS_FILE):
        return None
    with open(EMBEDDINGS_FILE, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def extract_vector(image, processor, model):
    img = image.convert("RGB")
    w, h = img.size
    margin = 0.15
    img = img.crop((w * margin, h * margin, w * (1-margin), h * (1-margin)))
    
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        patch_tokens = outputs.last_hidden_state[:, 1:, :] 
        return patch_tokens.mean(dim=1).cpu().numpy()

def main():
    st.title("📦 Objaverse Mass Style Explorer")
    st.markdown("수천 개의 Objaverse 에셋을 장르/객체별로 분류하여 스타일 지도를 생성했습니다.")

    # 모델 및 데이터 로드
    processor, model = load_dino_model()
    mtime = os.path.getmtime(EMBEDDINGS_FILE) if os.path.exists(EMBEDDINGS_FILE) else 0
    embeddings_raw = load_data(mtime)

    if embeddings_raw is None:
        st.warning("먼저 `objaverse_mass_analyzer.py`를 실행하여 벡터를 추출해주세요.")
        return

    # 업로드 섹션
    st.sidebar.header("📁 Context Upload")
    uploaded_file = st.sidebar.file_uploader("참고용 스타일 이미지 업로드", type=["jpg", "jpeg", "png"])
    
    all_vectors = []
    all_uids = []
    all_categories = []
    all_paths = []
    all_types = []

    # 기존 데이터셋 추가 (objaverse_mass_embeddings.pkl 구조 반영)
    for path, data in embeddings_raw.items():
        all_vectors.append(data["vector"].flatten())
        all_uids.append(data["uid"])
        all_categories.append(data["category"])
        all_paths.append(path)
        all_types.append("Objaverse Collection")

    # 신규 이미지 처리
    query_idx = None
    if uploaded_file:
        with st.spinner("이미지 분석 중..."):
            query_img = Image.open(uploaded_file)
            query_vec = extract_vector(query_img, processor, model)
            all_vectors.append(query_vec.flatten())
            all_uids.append("UPLOAD")
            all_categories.append("USER_IMAGE")
            all_paths.append(uploaded_file)
            all_types.append("YOUR IMAGE")
            query_idx = len(all_vectors) - 1

    # t-SNE 계산
    with st.spinner("스타일 지도 생성 중 (t-SNE)..."):
        vectors_np = np.array(all_vectors)
        # 데이터가 많으므로 perplexity 상향
        perplexity = min(30, max(5, len(all_vectors) // 10))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        vectors_2d = tsne.fit_transform(vectors_np)

    df = pd.DataFrame({
        'x': vectors_2d[:, 0],
        'y': vectors_2d[:, 1],
        'uid': all_uids,
        'category': all_categories,
        'type': all_types,
        'index': range(len(all_uids))
    })

    # 시각화
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Mass Style Similarity Map")
        fig = px.scatter(df, x='x', y='y', color='category',
                         hover_data=['uid', 'type'],
                         custom_data=['index'],
                         template="plotly_dark",
                         title="Objaverse Categorized Style Space")
        
        fig.update_traces(marker=dict(size=12, opacity=0.7))
        
        # 업로드 이미지 특별 표시
        if query_idx is not None:
            fig.update_traces(
                selector=dict(category="USER_IMAGE"),
                marker=dict(size=30, symbol="star", color="red", line=dict(width=2, color="white"))
            )
            
        fig.update_layout(height=650, margin=dict(l=0, r=0, b=0, t=0))
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

    with col2:
        st.subheader("Mass Asset Inspection")
        selected_idx = None
        
        if event and "selection" in event and event["selection"]["points"]:
            selected_idx = int(event["selection"]["points"][0]["customdata"][0])
        elif query_idx is not None:
            selected_idx = query_idx
            
        if selected_idx is not None:
            uid = all_uids[selected_idx]
            cat = all_categories[selected_idx]
            path = all_paths[selected_idx]
            
            st.info(f"Category: {cat}")
            st.write(f"UID: `{uid}`")
            
            if uid == "UPLOAD":
                st.image(path, use_container_width=True, caption="Uploaded Image")
            else:
                if os.path.exists(path):
                    st.image(Image.open(path), use_container_width=True)
                    st.markdown(f"[Sketchfab에서 보기](https://sketchfab.com/3d-models/{uid})")
        else:
            st.write("지도의 점을 클릭하거나 이미지를 업로드하세요.")

if __name__ == "__main__":
    main()
