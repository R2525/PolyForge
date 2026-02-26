# Classification Configuration for Asset Tagging

# 1. Genre (장르/분위기) - 9 Categories
GENRE_TAXONOMY = {
    "Fantasy": "Medieval, Magic, RPG, Castle, Dungeon, Knight (중세 유럽, 마법, 전설 속의 신비로운 분위기)",
    "Sci-Fi": "Futuristic, Space, High-tech, Robotics, Alien, Galaxy (우주, 첨단 과학 기술이 발달한 먼 미래)",
    "Cyberpunk": "Neon, Tech-Noir, Hackers, Cyborg, Night-City (네온사인, 기계 의수 등 화려하지만 암울한 근미래)",
    "Modern": "Contemporary, City, Office, Daily-life, Urban (현재 우리가 살고 있는 시대의 일상적인 모습)",
    "Post-Apocalyptic": "Survival, Wasteland, Rusty, Abandoned, Ruined, Zombie (멸망한 세계, 황폐화된 환경과 생존의 느낌)",
    "Historical": "Ancient, Victorian, Western, Antique, Vintage, Retro (특정 과거 시대서부극, 유물 등를 고증함)",
    "Horror": "Scary, Dark, Occult, Supernatural, Blood, Skeleton (공포, 기괴함, 어두운 분위기를 조성함)",
    "Military": "Tactical, Army, War, Special-Forces, Camo, Weaponry (실제 군대, 전술 장비, 현대전의 딱딱한 느낌)",
    "Steampunk": "Steam-powered, Gears, Clockwork, Victorian-SciFi (증기기관, 톱니바퀴 등 아날로그 기계 미학)"
}

# 2. Visual Style (시각적 스타일) - 7 Styles
STYLE_TAXONOMY = {
    "Photorealistic": "PBR, 8K, Scanned, Real-life, High-fidelity (사진이나 실물처럼 정밀하고 사실적인 질감)",
    "Stylized": "Hand-painted, Overwatch-style, Soft, Gentle, Casual (형태를 단순화하고 색감을 강조한 만화적 느낌)",
    "Low-Poly": "Flat-shaded, Minimal, Sharp-edge, Optimized (면을 적게 써서 각진 느낌이 살아있는 미니멀 스타일)",
    "Toon / Anime": "Cel-shaded, Outline, Manga, 2D-look, Flat-color (외곽선이 뚜렷한 2D 애니메이션 화풍)",
    "Voxel": "Blocky, Minecraft-style, Pixel-3D, Cubes (큐브블록를 쌓아서 만든 레고 같은 스타일)",
    "Retro / PS1": "Pixelated, Crunchy, Low-res, 90s-Gaming, Jittery (거친 텍스처와 투박한 폴리곤의 고전 게임 감성)",
    "Abstract": "Geometric, Artistic, Conceptual, Non-representative (현실에 없는 기하학적이고 예술적인 디자인)"
}

# 3. Surface Condition (표면 상태)
SURFACE_CONDITION_TAXONOMY = {
    "Rusty": "Oxidized metal, reddish-brown corrosion, weathered iron",
    "Dusty": "Covered in fine particles, powdery surface, long-term storage look",
    "Mossy": "Overgrown with green moss or lichen, organic damp aging",
    "Glow": "Emissive parts, neon lights, magical aura, internal light source",
    "Blood-stained": "Dried or fresh blood splatters, gore, violent aftermath",
    "Wet": "Shiny, water droplets, soaked appearance, rain effect",
    "Broken": "Cracked, shattered, fractured, damaged geometry",
    "Clean": "Pristine, brand new, polished, well-maintained"
}

# Legacy Condition (kept for DINO backward compatibility)
CONDITION_CONFIG = {
    "labels": ["weathered_old", "pristine_new"],
    "prompts": [
        "a worn-out 3D object with scratches and rust, weathered and old surface",
        "a brand new 3D object with a clean polished surface, pristine and unused"
    ]
}

# 4. Top-level Categories (LVIS Taxonomy 기반 18개 대분류)
# lvis_taxonomy.json의 키값이 라벨이 됩니다.
CATEGORY_PROMPTS = {
    "Animals & Creatures": "a 3D studio render of an animal, creature, or pet, organic creature model",
    "Architecture": "a 3D studio render of a building, house, structure, or architectural element",
    "Art & Abstract": "a 3D studio render of a sculpture, statue, or artistic decorative artifact",
    "Cars & Vehicles": "a 3D studio render of a vehicle, car, truck, or transportation machine",
    "Characters & Creatures": "a 3D studio render of a character, robot, or humanoid figure",
    "Cultural Heritage & History": "a 3D studio render of an ancient artifact, historical relic, or traditional heritage object",
    "Electronics & Gadgets": "a 3D studio render of an electronic device, gadget, computer, or digital hardware",
    "Fashion & Style": "a 3D studio render of clothing, apparel, jewelry, or fashion accessory",
    "Food & Drink": "a 3D studio render of edible food, delicious meal, or beverage item",
    "Furniture & Home": "a 3D studio render of indoor furniture, household decor, or home appliance",
    "Music": "a 3D studio render of a musical instrument or audio equipment",
    "Nature & Plants": "a 3D studio render of a natural plant, tree, flower, or environmental element",
    "News & Politics": "a 3D studio render of a document, flag, or sign related to information",
    "People": "a 3D studio render of a human being or human-related object",
    "Places & Locations": "a 3D studio render of a landmark or specific environment spot",
    "Science & Technology": "a 3D studio render of a scientific tool, laboratory equipment, or engineering component",
    "Sports & Fitness": "a 3D studio render of sports equipment, fitness tool, or athletic gear",
    "Weapons & Military": "a 3D studio render of a combat weapon, firearm, or military hardware"
}

# LVIS 태그 매핑용 프롬프트 템플릿
LVIS_PROMPT_TEMPLATE = "a 3D studio render of a {}, high-quality game asset"
