import os
import json
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px

def visualize_dino_vectors(input_dir, output_html):
    print(f"[*] Reading JSON files from {input_dir}...")
    
    data = []
    
    # 순회하면서 JSON 파일 읽기
    for filename in os.listdir(input_dir):
        if filename.endswith("_analysis.json"):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                    
                    asset_id = entry.get("asset_id", filename)
                    category = entry.get("category", "Unknown")
                    genre = entry.get("genre", "Unknown")
                    style = entry.get("style", "Unknown")
                    dino_vec = entry.get("dino_vector", [])
                    dino_mode = entry.get("dino_mode", "unknown")
                    
                    if dino_vec and len(dino_vec) > 0:
                        data.append({
                            "asset_id": asset_id,
                            "category": category,
                            "genre": genre,
                            "style": style,
                            "dino_mode": dino_mode,
                            "vector": np.array(dino_vec)
                        })
            except Exception as e:
                print(f"    [!] Error reading {filename}: {e}")

    if not data:
        print("[!] No valid DINO vectors found.")
        return

    print(f"[*] Found {len(data)} vectors. Preparing dimensionality reduction...")
    
    # 벡터 추출
    vectors = np.array([d["vector"] for d in data])
    
    # 1. PCA로 차원 예비 축소 (t-SNE 성능 및 안정성 향상)
    # 최소 데이터 개수보다 작게 설정
    n_components_pca = min(50, len(data))
    if len(data) > 1:
        pca = PCA(n_components=n_components_pca)
        vectors_reduced = pca.fit_transform(vectors)
    else:
        vectors_reduced = vectors

    # 2. t-SNE 수행 (2D로 축소)
    print("[*] Running t-SNE...")
    # 데이터 개수가 너무 적으면 perplexity 조정 필요
    perp = min(30, max(1, len(data) - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    vis_coords = tsne.fit_transform(vectors_reduced)

    # DataFrame 생성
    df = pd.DataFrame(data)
    df['x'] = vis_coords[:, 0]
    df['y'] = vis_coords[:, 1]
    
    # 시각화 생성
    print("[*] Generating Plotly interactive map...")
    fig = px.scatter(
        df, 
        x='x', 
        y='y', 
        color='category',
        hover_data=['asset_id', 'genre', 'style', 'dino_mode'],
        title='DINO Texture Embedding Clustering (t-SNE)',
        labels={'x': 't-SNE component 1', 'y': 't-SNE component 2'},
        template='plotly_dark'
    )

    # 레이아웃 조정
    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='White')))
    fig.update_layout(
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # HTML 저장
    fig.write_html(output_html)
    print(f"[+] Visualization saved to: {output_html}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "output", "iso_analysis")
    output_path = os.path.join(current_dir, "output", "dino_visualization.html")
    
    # output 폴더 확인
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    visualize_dino_vectors(input_path, output_path)
