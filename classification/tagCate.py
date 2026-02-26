import pandas as pd
import json

# JSON 데이터 로드
with open('integrated_assets.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 리스트 형태를 콤마로 구분된 문자열로 변환 (엑셀 보기 편함)
for item in data['asset_taxonomy']:
    item['multi_categories'] = ", ".join(item['multi_categories'])
    item['environments'] = ", ".join(item['environments'])
    item['recommended_themes'] = ", ".join(item['recommended_themes'])
    item['search_keywords'] = ", ".join(item['search_keywords'])

# 엑셀로 저장
df = pd.DataFrame(data['asset_taxonomy'])
df.to_excel('3D_Asset_Taxonomy_Master.xlsx', index=False)