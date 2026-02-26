# PolyForge2: 3D Asset Semantic Classification & Style Explorer

This project provides a comprehensive pipeline for collecting, analyzing, and exploring large-scale 3D asset datasets (Objaverse & Sketchfab). It leverages **DINOv2** for visual style embeddings and **Qwen-VL (Vision-Language Model)** for deep semantic tagging.

## 🚀 Key Features

- **Mass Downloader**: Hybrid filtering (LVIS annotations + Keyword Metadata) for targeted Objaverse data collection.
- **Style Analysis**: Feature extraction using DINOv2 with Mean Patch Pooling for robust style representation.
- **Semantic Tagging**: Large-scale VLM pipeline using Qwen3-VL to extract categories, genres, styles, and surface conditions.
- **Interactive Explorer**: Streamlit-based dashboards for t-SNE visualization, similarity search, and asset inspection.

---

## 📊 Current Progress (as of 2026-02-26)

- **Objaverse Dataset**: **595 / 626** assets tagged (95.0%)
- **Sketchfab Dataset**: **359 / 3,703** assets tagged (9.7%)
- **Total Semantic Tags**: **954** JSON files generated.
- **Analysis Resolution**: Optimized at 960x540 (Half-HD) for a balance of visual fidelity and processing speed.

### 💡 Strategy: Remote-First Indexing
This project is designed to act as a **lightweight semantic index**. 
- You do **not** need to download full GLB/FBX files to build the style map.
- Analysis can be performed directly on thumbnail URLs.
- The Style Explorer provides direct links to official 3D viewers (Sketchfab/Objaverse), significantly saving disk space.

---

## 🛠️ Project Structure

### 1. Data Collection (`sketchfab_tools/`)
- `objaverse_mass_downloader.py`: Downloads Objaverse assets based on specific categories/genres.
- `sketchfab_batch_downloader.py`: Utility for batch downloading Sketchfab models and metadata.

### 2. Feature Extraction & Analysis
- `objaverse_mass_analyzer.py`: Generates 768-dim DINOv2 embeddings for style-based clustering.
- `mass_semantic_tagger.py`: **[Core]** Uses Qwen3-VL to generate detailed semantic JSON tags for every asset.
  - Supports resume functionality and optimized resolution (960x540) for speed/quality balance.

### 3. Visualization (`sketchfab_tools/` & `classification/`)
- `objaverse_style_explorer.py`: Interactive style map for Objaverse data.
- `dino_dashboard.py`: Advanced aesthetic/style analysis dashboard.

---

## 💻 Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (VRAM 12GB+ recommended for Qwen-VL)

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/polyforge2-classification.git
cd polyforge2-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Usage Guide

### 1. Mass Semantic Tagging (Qwen-VL)
This is the main pipeline for generating detailed metadata for thousands of assets.

```bash
python sketchfab_tools/mass_semantic_tagger.py \
    --input_dir objaverse_mass_data \
    --output_dir output/semantic_tags/objaverse \
    --filter thumbnail.jpg \
    --recursive
```
*Note: Uses an optimized 960x540 resolution to process assets in ~15s each while maintaining high semantic accuracy.*

### 3. Resuming Tagging on Another Computer (Remote Mode)
If you move this project to another computer and delete the local images, you can continue tagging by fetching images directly from Sketchfab's URLs.

```bash
# Resume Sketchfab tagging using metadata JSONs
python sketchfab_tools/mass_semantic_tagger.py \
    --metadata_dir sketchfab_metadata \
    --output_dir output/semantic_tags/sketchfab \
    --input_dir .
```
*Note: This mode requires an active internet connection as it fetches images on-the-fly to memory.*

---

## 🏷️ Semantic Taxonomy
The system classifies assets using the following taxonomies defined in `classification/classification_config.py`:
- **Genres**: Cyberpunk, Steampunk, Medieval, Sci-Fi, etc.
- **Styles**: Low-Poly, Voxel, Realistic, Stylized, Hand-painted, etc.
- **Surface Conditions**: Rusty, Shiny, Damaged, Clean, etc.

---

## 🤖 AI Model Stack & Versions

This project utilizes state-of-the-art vision and multimodal models. Below are the specific versions and checkpoints used.

### 🧠 Core Models
| Task | Model ID (HuggingFace) | Description |
| :--- | :--- | :--- |
| **Style Embedding** | `facebook/dinov2-base` | Vision Transformer (ViT) for high-fidelity style feature extraction (768-dim). |
| **Semantic Tagging** | `Qwen/Qwen3-VL-2B-Instruct` | Advanced Multimodal LLM (Qwen-VL) for natural language reasoning over images. |

### 🛠️ Key Dependencies
- **PyTorch**: `2.4.1+cu118`
- **Transformers**: `5.3.0.dev0` (Qwen3-VL optimized)
- **Qwen-VL Utils**: `0.0.14`
- **Pillow**: `12.1.1`

---

## 📄 License
[Insert License Information Here]
