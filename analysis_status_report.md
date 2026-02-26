# Analysis Progress Report

| Dataset | Total Assets | DINO Embedded (Style) | Qwen Tagged (Semantic) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Objaverse Mass Data** | 626 | 626 (100%) | 595 (95.0%) | Almost Complete |
| **Sketchfab Data** | 3,703 | 3,703 (100%) | 468 (12.6%) | In Progress |
| **Total** | **4,329** | **4,329 (100%)** | **1,063 (24.5%)** | |

### 🔍 Analysis Details
- **DINO (Style Map)**: 모든 에셋에 대한 시각적 특징 추출이 완료되어 스타일 지도(t-SNE) 상에 매핑되었습니다.
- **Qwen-VL (Semantic)**: 각 에셋의 장르, 재질, 스타일명을 텍스트로 추출하는 과정입니다. 현재 1,063개의 JSON 파일이 생성되었습니다.

### 📂 File Reference
- **Embeddings**: `sketchfab_tools/*.pkl`
- **Semantic Tags**: `output/semantic_tags/`
- **Status Log**: `analysis_status_summary.json`
