import streamlit as st
import os
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.manifold import TSNE
import torch
from transformers import AutoImageProcessor, AutoModel
import io
import base64

# --- Page Config ---
st.set_page_config(page_title="Sketchfab Style Explorer", layout="wide", page_icon="🦖")

# --- Constants ---
EMBEDDINGS_FILE = "sketchfab_tools/sketchfab_embeddings.pkl"
IMAGE_DIR = "sketchfab_data"
MODEL_ID = "facebook/dinov2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Cached Functions ---
@st.cache_resource
def load_dino_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    return processor, model

@st.cache_data
def load_dataset_embeddings(mtime):
    """파일 수정 시간을 감지하여 캐시를 자동 갱신합니다."""
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}
    with open(EMBEDDINGS_FILE, 'rb') as f:
        return pickle.load(f)

def extract_vector(image, processor, model):
    img = image.convert("RGB")
    
    # [추가] 재질(Style)에 더 집중하기 위해 중앙 크롭 (분석기와 동일하게 70% 영역)
    w, h = img.size
    left, top, right, bottom = w * 0.15, h * 0.15, w * 0.85, h * 0.85
    img = img.crop((left, top, right, bottom))
    
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        # [수정] CLS 대신 패치 토큰의 평균(GAP) 사용
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        return patch_tokens.mean(dim=1).cpu().numpy()

# --- App Structure ---
st.title("🦖 Sketchfab Style Explorer")
st.markdown("나만의 사진을 업로드하여 Sketchfab 중세 자산들과의 스타일 유사도를 확인해보세요.")

sidebar = st.sidebar
sidebar.header("📁 데이터 설정")

# 1. 모델 및 데이터 로드
processor, model = load_dino_model()
file_mtime = os.path.getmtime(EMBEDDINGS_FILE) if os.path.exists(EMBEDDINGS_FILE) else 0
embeddings_dict = load_dataset_embeddings(file_mtime)

if not embeddings_dict:
    st.error(f"임베딩 데이터({EMBEDDINGS_FILE})를 찾을 수 없습니다. 분석을 먼저 실행해주세요.")
    st.stop()

# 2. 이미지 업로드 섹션 (메인 페이지 상단에 배치)
st.info("💡 이미지를 업로드하면 자동으로 분석되어 그래프에 **빨간 별(★)**로 표시됩니다.")
uploaded_file = st.file_uploader("새로운 스타일 사진 업로드 (디자인 시안, 참고 사진 등)", type=["jpg", "jpeg", "png"])

all_vectors = []
all_names = []
all_types = []
all_paths = []

# 데이터셋 임베딩 추가
for filename, vec_np in embeddings_dict.items():
    all_vectors.append(vec_np.squeeze())
    all_names.append(filename.split("_")[-1].replace(".jpg", ""))
    all_types.append("Sketchfab Dataset")
    all_paths.append(os.path.join(IMAGE_DIR, filename))

# 업로드된 이미지가 있으면 임베딩 추출 및 추가
query_vec = None
if uploaded_file:
    with st.spinner("이미지 분석 중..."):
        query_image = Image.open(uploaded_file)
        query_vec = extract_vector(query_image, processor, model)
        
        all_vectors.append(query_vec.squeeze())
        all_names.append("YOUR_UPLOAD")
        all_types.append("YOUR IMAGE")
        # 쿼리 이미지는 경로 대신 PIL 이미지를 직접 처리하기 위해 None 처리하거나 임시 저장
        all_paths.append(uploaded_file)

# 3. t-SNE 계산
with st.spinner("유사도 공간 계산 중..."):
    vectors_array = np.array(all_vectors)
    perplexity = min(30, len(all_vectors) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    vectors_2d = tsne.fit_transform(vectors_array)

# 4. 시각화 및 인터랙션
if "highlighted_idx" not in st.session_state:
    st.session_state.highlighted_idx = None

# 프로젝트 루트 경로 확보 (이미지 로딩용)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABS_IMAGE_DIR = os.path.join(PROJECT_ROOT, "sketchfab_data")

df_plot = {
    "index": list(range(len(all_names))),
    "x": vectors_2d[:, 0],
    "y": vectors_2d[:, 1],
    "Name": all_names,
    "Type": all_types,
    "Path": []
}

# 경로 재설정 (확실한 절대경로)
for i, p in enumerate(all_paths):
    if isinstance(p, str):
        # 파일명만 추출해서 ABS_IMAGE_DIR과 결합
        fname = os.path.basename(p)
        df_plot["Path"].append(os.path.join(ABS_IMAGE_DIR, fname))
    else:
        df_plot["Path"].append(p)

st.subheader("📍 Style Similarity Map")
st.caption("Lasso Select(올가미)나 Box Select를 사용하여 영역을 선택하면 하단에 썸네일이 나타납니다.")

fig = px.scatter(
    df_plot, x="x", y="y", color="Type",
    hover_name="Name",
    color_discrete_map={"Sketchfab Dataset": "#636EFA", "YOUR IMAGE": "#EF553B"},
    height=600,
    template="plotly_dark",
    labels={"x": "Style A", "y": "Style B"},
    custom_data=["index"]
)

fig.update_traces(marker=dict(size=14, opacity=0.8))
if query_vec is not None:
    fig.update_traces(
        selector=dict(name="YOUR IMAGE"),
        marker=dict(size=35, symbol="star", line=dict(width=3, color="white"))
    )

if st.session_state.highlighted_idx is not None:
    idx = st.session_state.highlighted_idx
    if idx < len(vectors_2d): # Ensure index is within bounds
        fig.add_trace(go.Scatter(
            x=[vectors_2d[idx, 0]], y=[vectors_2d[idx, 1]],
            mode='markers',
            marker=dict(
                size=30, 
                color='#FFFF00', # Bright Yellow
                symbol='star', 
                line=dict(width=2, color='white')
            ),
            name="Selected Location",
            showlegend=False,
            hoverinfo='skip'
        ))

# 디버그용 (필요시 주석 해제)
# st.write(f"Debug: Project Root = {PROJECT_ROOT}")
event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="style_map")

# 5. 선택된 항목 갤러리 (아래쪽)
st.divider()
st.subheader("🖼️ Selected Assets Gallery")

def img_to_base64(path_or_file):
    """이미지를 base64로 변환하여 브라우저에서 직접 표시되게 합니다."""
    try:
        if isinstance(path_or_file, str):
            if not os.path.exists(path_or_file):
                return None
            with open(path_or_file, "rb") as f:
                data = f.read()
        else: # UploadedFile
            data = path_or_file.getvalue()
        return base64.b64encode(data).decode()
    except Exception as e:
        return None

# 이벤트 캡처 디버그 및 인덱스 추출
selected_indices = []
if event and "selection" in event:
    pts = event["selection"].get("points", [])
    if pts:
        st.success(f"DEBUG: {len(pts)}개의 포인트가 선택됨")
        for p in pts:
            # 1. custom_data 우선 확인
            if "custom_data" in p and p["custom_data"] is not None:
                selected_indices.append(p["custom_data"][0])
            # 2. point_index를 차선책으로 사용 (단일 트레이스일 때 유리)
            elif "point_index" in p:
                selected_indices.append(p["point_index"])

if not selected_indices:
    st.info("그래프에서 점을 클릭하거나 마우스로 영역을 드래그(Box/Lasso)하여 선택해주세요.")
else:
    # 중복 제거 및 유효 범위 확인
    unique_indices = list(dict.fromkeys(selected_indices))
    valid_indices = [idx for idx in unique_indices if idx < len(all_names)]
    
    # 갤러리 그리드 구성
    cols = st.columns(5)
    for i, idx in enumerate(valid_indices):
        with cols[i % 5]:
            path = df_plot["Path"][idx]
            name = df_plot["Name"][idx]
            
            # Base64 변환 및 표시
            b64_img = img_to_base64(path)
            
            if b64_img:
                # 캡션과 이미지를 묶어서 표시
                st.markdown(f"**{name[:15]}**")
                st.markdown(f'<img src="data:image/jpeg;base64,{b64_img}" style="width:100%; border-radius:10px; border: 2px solid #444;">', unsafe_allow_html=True)
            else:
                st.warning(f"Image Missing: {name}")
            
            # 하이라이트 버튼
            if st.button(f"🔍 위치 찾기", key=f"gal_btn_{idx}_{i}"):
                st.session_state.highlighted_idx = idx
                st.rerun()
            
            # 외부 링크
            if df_plot["Type"][idx] == "Sketchfab Dataset" and isinstance(path, str):
                model_id = os.path.basename(path).split("_")[0]
                st.caption(f"[Sketchfab에서 보기](https://sketchfab.com/3d-models/{model_id})")

# 푸터
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.divider()
st.caption("DINOv2 GAP Features + t-SNE | Interactive Style Explorer | PolyForge")
