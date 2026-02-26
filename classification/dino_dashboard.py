import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import cv2
import glob

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PolyForge DINO Dashboard", layout="wide", page_icon="🦖")

DINO_MODEL_ID = "facebook/dinov2-base"
INPUT_ANALYSIS_DIR = os.path.join(os.getcwd(), "output", "iso_analysis")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Cached Model & Data Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_dino_model():
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    model = AutoModel.from_pretrained(DINO_MODEL_ID).to(DEVICE)
    model.eval()
    return processor, model

def load_analysis_data(input_dir):
    data = []
    if not os.path.exists(input_dir):
        return []
    
    for filename in os.listdir(input_dir):
        if filename.endswith("_analysis.json"):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                    asset_id = entry.get("asset_id", filename)
                    vec = entry.get("dino_vector", [])
                    if vec:
                        data.append({
                            "index": len(data),
                            "asset_id": asset_id,
                            "category": entry.get("category", "Unknown"),
                            "genre": entry.get("genre", "Unknown"),
                            "style": entry.get("style", "Unknown"),
                            "dino_mode": entry.get("dino_mode", "unknown"),
                            "vector": np.array(vec),
                            "source_json": filename,
                            "mesh_info": entry.get("mesh_info", {}),
                            "images": entry.get("images", [])
                        })
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
    return data

def extract_features(image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        # Assuming dinov2-base CLS token is at index 0
        features = outputs.last_hidden_state[0, 0, :]
    return features.cpu().numpy()

def generate_attention_map(image, processor, model):
    # This is a simplified version of attention map extraction
    # DINOv2 uses VitLayer.attention.attention.query/key/value or similar.
    # For AutoModel, we need to access the appropriate blocks.
    
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    
    # We need to get attention outputs. AutoModel might not return them by default.
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    if not outputs.attentions:
        return None
    
    # Get last layer attention: (batch, heads, seq_len, seq_len)
    last_attn = outputs.attentions[-1] 
    # Shape: [1, 12, 257, 257] for base (256 patches + 1 CLS)
    
    # Mean across heads
    avg_attn = last_attn[0].mean(dim=0) # [257, 257]
    
    # Attention from CLS token to all other patches
    cls_attn = avg_attn[0, 1:].cpu().numpy() # [256]
    
    # Reshape to 16x16 (for 224x224 input with 14x14 patch)
    # Actually, dinov2-base often uses 14x14 patch size. 
    # seq_len - 1 = total patches.
    patch_size = int(np.sqrt(len(cls_attn)))
    attn_grid = cls_attn.reshape(patch_size, patch_size)
    
    # Normalize
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
    
    # Resize to original image size
    img_array = np.array(image)
    heatmap = cv2.resize(attn_grid, (img_array.shape[1], img_array.shape[0]))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    alpha = 0.5
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(overlay)

def reconstruct_normal_map(image):
    """
    Detects and fixes Unity 'swizzled' normal maps (DXT5nm style) for visualization.
    Standard purple: [128, 128, 255]
    Swizzled teal: [Low, High, High, High(X)]
    """
    if image.mode != "RGBA":
        # If it's already RGB and looks purple, keep it. 
        # If it's RGB and looks teal, we might be missing the X channel in Alpha.
        return image
        
    img_array = np.array(image).astype(np.float32)
    r, g, b, a = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2], img_array[:,:,3]
    
    # Check if it's likely swizzled (Low Red, High Green/Blue/Alpha)
    avg_r = np.mean(r)
    if avg_r < 100:
        # Reconstruct standard purple map
        # Unity DXT5nm: X is in Alpha, Y is in Green. 
        # We'll map them back to R and G.
        new_img = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
        
        # X -> Red (Alpha channel)
        # Y -> Green (Green channel)
        new_img[:,:,0] = a.astype(np.uint8)
        new_img[:,:,1] = g.astype(np.uint8)
        
        # Z -> Blue (Often constant or reconstructed)
        # For visualization, we just use a high blue value or try to fix it.
        # Simple reconstruction: 
        nx = (a / 255.0) * 2.0 - 1.0
        ny = (g / 255.0) * 2.0 - 1.0
        nz = np.sqrt(np.clip(1.0 - nx*nx - ny*ny, 0, 1))
        
        new_img[:,:,2] = ((nz * 0.5 + 0.5) * 255).astype(np.uint8)
        
        return Image.fromarray(new_img)
    
    return image.convert("RGB")

# -----------------------------------------------------------------------------
# Main Dashboard UI
# -----------------------------------------------------------------------------
st.title("🦖 PolyForge Interactive DINO Dashboard")
st.markdown("---")

processor, model = load_dino_model()
base_data = load_analysis_data(INPUT_ANALYSIS_DIR)

sidebar = st.sidebar
sidebar.header("Scan & Upload")

if sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

uploaded_file = sidebar.file_uploader("Upload Texture/Image for Analysis", type=["png", "jpg", "jpeg", "tga"])

sidebar.markdown("---")
# Handle selection & Visibility Defaults using session_state
if "selected_asset_id" not in st.session_state:
    st.session_state.selected_asset_id = None
if "show_norm" not in st.session_state:
    st.session_state.show_norm = True
if "show_rough" not in st.session_state:
    st.session_state.show_rough = True
if "show_albedo" not in st.session_state:
    st.session_state.show_albedo = True
if "show_attention" not in st.session_state:
    st.session_state.show_attention = True

sidebar.header("Visibility Settings")
sidebar.checkbox("Show Albedo (Texture)", key="show_albedo")
sidebar.checkbox("Show DINO Attention Map", key="show_attention")
sidebar.checkbox("Show Normal Map", key="show_norm")
sidebar.checkbox("Show Roughness Map", key="show_rough")

# Sidebar dropdown as a secondary selection method
st.session_state.selected_asset_id = sidebar.selectbox(
    "Select Existing Asset to Inspect", 
    options=[None] + [d["asset_id"] for d in base_data],
    index=([None] + [d["asset_id"] for d in base_data]).index(st.session_state.get("selected_asset_id")) if st.session_state.get("selected_asset_id") in [d["asset_id"] for d in base_data] else 0
)

selected_asset = next((d for d in base_data if d["asset_id"] == st.session_state.selected_asset_id), None) if st.session_state.selected_asset_id else None

# -----------------------------------------------------------------------------
# Processing Logic
# -----------------------------------------------------------------------------
display_data = base_data.copy()
new_entry = None

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Analyzing uploaded image..."):
            new_vec = extract_features(img, processor, model)
            
            new_entry = {
                "asset_id": f"UPLOADED: {uploaded_file.name}",
                "category": "User Upload",
                "genre": "N/A",
                "style": "N/A",
                "dino_mode": "raw_upload",
                "vector": new_vec,
                "is_new": True
            }
            display_data.append(new_entry)
            st.success("Analysis complete!")
    except Exception as e:
        st.error(f"Failed to process image: {e}")

# -----------------------------------------------------------------------------
# Visualization & Layout
# -----------------------------------------------------------------------------
if display_data:
    # Prepare vectors for DR
    vectors = np.array([d["vector"] for d in display_data])
    
    # Perform Dimensionality Reduction
    # We refit everything for accurate "relative" positioning
    # PCA first
    n_pca = min(50, len(vectors))
    pca = PCA(n_components=n_pca)
    vectors_pca = pca.fit_transform(vectors)
    
    # t-SNE
    perp = min(30, max(1, len(vectors) - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(vectors_pca)
    
    df = pd.DataFrame(display_data)
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]
    df['Size'] = df.apply(lambda r: 25 if r.get('is_new') else 10, axis=1)
    df['Symbol'] = df.apply(lambda r: 'star' if r.get('is_new') else 'circle', axis=1)

    # Plot
    col_map, col_details = st.columns([2, 1])

    with col_map:
        st.subheader("Asset Distribution Map")
        fig = px.scatter(
            df, 
            x='x', y='y', 
            color='category', 
            text='index', # Show index number on map
            hover_data=['index', 'asset_id', 'category', 'dino_mode'],
            size='Size',
            symbol='Symbol',
            height=600,
            template="plotly_dark",
            title="DINO Texture Similarity Space (t-SNE)"
        )
        # Style the text labels
        fig.update_traces(textposition='top center')
        fig.update_layout(showlegend=True, clickmode='event+select')
        
        # Plotly Selection
        map_selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
        
        if map_selection and "selection" in map_selection and "points" in map_selection["selection"]:
            points = map_selection["selection"]["points"]
            if points:
                # Get asset_id from hover_data (customdata)
                selected_id_from_map = points[0].get("customdata", [None])[0]
                if selected_id_from_map and selected_id_from_map != st.session_state.selected_asset_id:
                    st.session_state.selected_asset_id = selected_id_from_map
                    st.rerun()

    with col_details:
        st.subheader("Selected / Uploaded Analysis")
        
        # If new file is uploaded, show its attention map
        if uploaded_file is not None:
            st.info(f"Analysis for: **{uploaded_file.name}**")
            attn_img = generate_attention_map(img, processor, model)
            
            with st.expander("View Analysis Maps & Raw Vector", expanded=False):
                imgs_to_show = []
                captions = []
                if st.session_state.show_albedo:
                    imgs_to_show.append(img)
                    captions.append("Original")
                if st.session_state.show_attention and attn_img:
                    imgs_to_show.append(attn_img)
                    captions.append("DINO Attention Map")
                
                if imgs_to_show:
                    st.image(imgs_to_show, caption=captions, use_container_width=True)
                elif not st.session_state.show_albedo and not st.session_state.show_attention:
                    st.info("Albedo and Attention maps are hidden.")

                st.markdown("---")
                st.write("**Raw DINO Vector:**")
                st.write(new_entry["vector"])
                st.write(f"Vector Shape: {new_entry['vector'].shape}")
                
        elif selected_asset:
            st.warning(f"Inspecting: **{selected_asset['asset_id']}**")
            st.write(f"**Category:** {selected_asset['category']}")
            st.write(f"**DINO Mode:** {selected_asset['dino_mode']}")
            
            mesh_info = selected_asset.get("mesh_info", {})
            textures = mesh_info.get("textures", [])
            
            with st.expander("View Analysis Maps & Raw Vector", expanded=False):
                # Put all image displays here as requested
                if textures:
                    tex_path = textures[0]
                    # Handle path resolving
                    if not os.path.exists(tex_path):
                        potential_relative = os.path.join(os.getcwd(), tex_path)
                        if os.path.exists(potential_relative):
                            tex_path = potential_relative
                    
                    if os.path.exists(tex_path):
                        try:
                            tex_img = Image.open(tex_path).convert("RGB")
                            st.write(f"Source Texture: `{os.path.basename(tex_path)}`")
                            
                            attn_img = None
                            if st.session_state.show_attention:
                                with st.spinner("Generating attention map..."):
                                    attn_img = generate_attention_map(tex_img, processor, model)
                            
                            # Show Albedo & Attention
                            imgs_to_show = []
                            captions = []
                            if st.session_state.show_albedo:
                                imgs_to_show.append(tex_img)
                                captions.append("Albedo")
                            if st.session_state.show_attention and attn_img:
                                imgs_to_show.append(attn_img)
                                captions.append("Attention Map")
                            
                            if imgs_to_show:
                                st.image(imgs_to_show, caption=captions, use_container_width=True)
                            
                            # Show additional maps
                            extra_cols = st.columns(2)
                            norm_path = mesh_info.get("normal_map")
                            rough_path = mesh_info.get("roughness_map")
                            
                            if st.session_state.show_norm and norm_path and os.path.exists(norm_path):
                                with extra_cols[0]:
                                    try:
                                        n_img = Image.open(norm_path)
                                        n_img_fixed = reconstruct_normal_map(n_img)
                                        st.image(n_img_fixed, caption="Normal (Corrected)", use_container_width=True)
                                    except:
                                        st.image(norm_path, caption="Normal Map", use_container_width=True)
                            
                            if st.session_state.show_rough and rough_path and os.path.exists(rough_path):
                                with extra_cols[1]:
                                    st.image(rough_path, caption="Roughness", use_container_width=True)
                                    
                        except Exception as e:
                            st.warning(f"Could not load textures in expander: {e}")
                    else:
                        st.warning(f"Texture file not found for inspection.")

                # Always show ISO view as another reference in expander if possible
                images = selected_asset.get("images", [])
                if images and len(images) > 3 and os.path.exists(images[3]):
                    st.image(images[3], caption="ISO View Reference", use_container_width=True)

                st.markdown("---")
                st.write("**Raw DINO Vector:**")
                st.write(selected_asset["vector"])
                st.write(f"Vector Shape: {selected_asset['vector'].shape}")
        else:
            st.write("Click on the sidebar to upload an image or select an existing asset.")

    st.markdown("---")
    st.subheader("📁 Asset Explorer")
    
    tab1, tab2 = st.tabs(["📊 Table Mode", "🖼️ Gallery Mode"])
    
    with tab1:
        # Table selection logic (Single Selection)
        table_df = df[['index', 'asset_id', 'category', 'genre', 'style', 'dino_mode']].copy()
        event = st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # Sync table selection with session state
        if event and "selection" in event and "rows" in event["selection"] and event["selection"]["rows"]:
            row_idx = event["selection"]["rows"][0]
            selected_id_from_table = table_df.iloc[row_idx]["asset_id"]
            if selected_id_from_table != st.session_state.selected_asset_id:
                st.session_state.selected_asset_id = selected_id_from_table
                st.rerun()

    with tab2:
        gallery_cols = st.columns(5)
        for i, row in df.iterrows():
            with gallery_cols[i % 5]:
                # Gallery Mode now ONLY shows ISO view as requested
                images = row.get("images", [])
                img_to_show = None
                
                if images and len(images) > 3 and os.path.exists(images[3]):
                    img_to_show = images[3]
                
                if img_to_show:
                    st.image(img_to_show, caption=f"#{row['index']} - {row['asset_id']}", use_container_width=True)
                else:
                    st.write(f"#{row['index']} (No ISO View)")
                
                if st.button(f"Select #{row['index']}", key=f"btn_{row['asset_id']}"):
                    st.session_state.selected_asset_id = row['asset_id']
                    st.rerun()

else:
    st.warning("No analysis data found in `output/iso_analysis`. Please run the pipeline first.")
