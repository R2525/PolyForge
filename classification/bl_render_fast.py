import bpy
import sys
import os
import mathutils
import math

def setup_fast_rendering():
    """EEVEE 엔진을 사용하여 PBR 재질이 보이도록 렌더링 설정"""
    scene = bpy.context.scene
    
    # 렌더 엔진을 EEVEE로 설정 (Blender 4.2+ 에서는 BLENDER_EEVEE_NEXT)
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except:
        scene.render.engine = 'BLENDER_EEVEE'
    
    # EEVEE 성능 최적화 (속도를 위해 불필요한 기능 끄기)
    if hasattr(scene, "eevee"):
        scene.eevee.use_gtao = False
        scene.eevee.use_bloom = False
        scene.eevee.use_ssr = False
        scene.eevee.shadow_method = 'NONE'
    
    # 배경 설정 (흰색)
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0) # White
        bg_node.inputs[1].default_value = 1.0 # Strength
    
    # 해상도 설정
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    
    # 불필요한 렌더러 기능 끄기
    scene.render.use_compositing = False
    scene.render.use_sequencer = False

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
try:
    from unity_texture_mapper import UnityTextureMapper
except Exception as e:
    print(f"Failed to load UnityTextureMapper: {e}")
    UnityTextureMapper = None

def fix_material_nodes(model_path):
    """모든 재질을 순회하며 UnityTextureMapper를 통해 정확한 텍스처를 연결"""
    if UnityTextureMapper is None:
        return
        
    base_dir = os.path.dirname(model_path)
    # Search up to 2 directories up for the root of the asset
    search_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))
    if "assets" not in search_dir.lower() and "model" not in search_dir.lower():
        search_dir = os.path.abspath(os.path.join(base_dir, ".."))
        
    try:
        mapper = UnityTextureMapper(search_dir)
    except Exception as e:
        print(f"Error initializing UnityTextureMapper: {e}")
        return

    for mat in bpy.data.materials:
        # Get all textures for this material
        tex_dict = mapper.get_textures_for_fbx_material(model_path, mat.name)
        if not tex_dict:
            continue
            
        print(f"[*] UnityTextureMapper matched {mat.name} with: {tex_dict}")
        
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not bsdf: continue
        
        # 1. Base Color
        if "albedo" in tex_dict:
            tex_path = tex_dict["albedo"]
            try:
                img = bpy.data.images.load(tex_path)
                node = nodes.new('ShaderNodeTexImage')
                node.image = img
                node.location = (bsdf.location.x - 600, bsdf.location.y + 300)
                links.new(node.outputs['Color'], bsdf.inputs['Base Color'])
                
                # 뷰포트/렌더링에서 이 텍스처가 보이도록 'active' 설정
                nodes.active = node
                mat.blend_method = 'HASHED' # 투명도 대응
            except: pass
            
        # 2. Normal Map
        if "normal" in tex_dict:
            tex_path = tex_dict["normal"]
            try:
                img = bpy.data.images.load(tex_path)
                img.colorspace_settings.name = 'Non-Color'
                tex_node = nodes.new('ShaderNodeTexImage')
                tex_node.image = img
                tex_node.location = (bsdf.location.x - 600, bsdf.location.y - 300)
                
                norm_node = nodes.new('ShaderNodeNormalMap')
                norm_node.location = (bsdf.location.x - 300, bsdf.location.y - 300)
                
                links.new(tex_node.outputs['Color'], norm_node.inputs['Color'])
                links.new(norm_node.outputs['Normal'], bsdf.inputs['Normal'])
            except: pass
            
        # 3. Roughness
        if "roughness" in tex_dict:
            tex_path = tex_dict["roughness"]
            try:
                img = bpy.data.images.load(tex_path)
                img.colorspace_settings.name = 'Non-Color'
                node = nodes.new('ShaderNodeTexImage')
                node.image = img
                node.location = (bsdf.location.x - 600, bsdf.location.y)
                links.new(node.outputs['Color'], bsdf.inputs['Roughness'])
            except: pass

def relink_textures(model_path):
    # 1. 100% 안전한 Unity 매핑 로직 실행 (우선순위)
    fix_material_nodes(model_path)
    
    # 2. Blender 기본 누락 탐색 (Fallback)
    base_dir = os.path.dirname(model_path)
    search_paths = [
        os.path.join(base_dir, "texture"),
        os.path.join(base_dir, "textures"),
        os.path.join(os.path.dirname(base_dir), "texture"),
        os.path.join(os.path.dirname(base_dir), "textures")
    ]
    for path in search_paths:
        if os.path.exists(path):
            bpy.ops.file.find_missing_files(directory=path)

def import_model(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=filepath)
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=filepath)
        else:
            return False
            
        # 텍스처 재연결 시도 (Workbench TEXTURE 모드에서 분홍색 방지)
        relink_textures(filepath)    
        return True
    except Exception as e:
        print(f"FAILED_TO_IMPORT: {e}")
        return False

def setup_camera_and_lighting():
    # 카메라 및 조명 초기화
    for obj in bpy.context.scene.objects:
        if obj.type in ['CAMERA', 'LIGHT']:
            bpy.data.objects.remove(obj, do_unlink=True)
            
    # 카메라 생성
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # 조명 생성 (EEVEE는 조명이 필요함)
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 4.0
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (0.785, 0, 0.785) # 45도 각도
    
    return cam_obj

def render_model(filepath, output_prefix):
    clear_scene()
    setup_fast_rendering()
    
    if not import_model(filepath):
        print("FAILED: Cannot import model.")
        return False

    # 객체들의 Bounding Box 계산
    min_coord = mathutils.Vector((float("inf"), float("inf"), float("inf")))
    max_coord = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))
    has_mesh = False

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            has_mesh = True
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ mathutils.Vector(corner)
                for i in range(3):
                    min_coord[i] = min(min_coord[i], world_corner[i])
                    max_coord[i] = max(max_coord[i], world_corner[i])

    if not has_mesh:
        print("FAILED: No mesh found.")
        return False

    center = (min_coord + max_coord) / 2.0
    dimensions = max_coord - min_coord
    max_dim = max(dimensions.x, dimensions.y, dimensions.z)

    # 폴리곤 및 정점 수 계산
    total_polygons = sum(len(obj.data.polygons) for obj in bpy.context.scene.objects if obj.type == 'MESH')
    total_vertices = sum(len(obj.data.vertices) for obj in bpy.context.scene.objects if obj.type == 'MESH')

    # MESH_DATA 출력 (파이프라인에서 추출용)
    print(f"MESH_DATA: faces={total_polygons}, vertices={total_vertices}, size_x={dimensions.x:.4f}, size_y={dimensions.y:.4f}, size_z={dimensions.z:.4f}")

    if max_dim == 0:
        max_dim = 1.0

    cam_obj = setup_camera_and_lighting()

    # 카메라 앵글 리스트 (직관적인 x, y, z 오프셋 설정)
    # 1: Front, 2: Back, 3: Top, 4: ISO
    distance = max_dim * 1.5
    angles_offsets = [
        (0, -distance, distance * 0.1),         # 1: Front (y축 뒤편에서 바라봄, 약간 위)
        (0, distance, distance * 0.1),          # 2: Back (y축 앞편에서 바라봄, 약간 위)
        (0, 0, distance),                       # 3: Top (z축 꼭대기에서 아래를 바라봄)
        (distance*0.7, -distance*0.7, distance*0.7) # 4: ISO (대각선 위)
    ]

    for i, offset in enumerate(angles_offsets, 1):
        # x, y, z 위치 계산
        cx = center.x + offset[0]
        cy = center.y + offset[1]
        cz = center.z + offset[2]
        
        cam_obj.location = (cx, cy, cz)
        
        # 카메라가 센터를 바라보도록 회전 설정
        direction = center - cam_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()

        # 출력 파일 경로 설정
        render_path = f"{output_prefix}_view{i}.png"
        bpy.context.scene.render.filepath = render_path
        
        # 렌더링
        bpy.ops.render.render(write_still=True)
        print(f"RENDERED: {render_path}")

    # [추가] 사용된 텍스처 파일명 수집 및 디버깅 로그
    used_textures = []
    print("\n--- MATERIAL DEBUG ---")
    has_missing_texture = False
    for mat in bpy.data.materials:
        print(f"Material: {mat.name}, use_nodes: {mat.use_nodes}")
        if not mat.use_nodes: continue
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE':
                img = node.image
                if img and getattr(img, 'filepath', ''):
                    tex_name = os.path.basename(img.filepath)
                    if not tex_name: tex_name = img.name
                    print(f"  - Node: {node.name}, Image: {tex_name} ({img.filepath})")
                    # Use absolute paths if possible
                    abs_path = bpy.path.abspath(img.filepath)
                    if hasattr(img, "filepath_from_user"):
                        abs_path = os.path.abspath(img.filepath_from_user())
                        
                    if abs_path not in used_textures:
                        used_textures.append(abs_path)
                else:
                    print(f"  - Node: {node.name}, Image: No Image Linked")
                    has_missing_texture = True
            elif node.type == 'BSDF_PRINCIPLED':
                print(f"  - Node: {node.name}")
    print("----------------------\n")
    
    if used_textures:
        print(f"TEXTURES: {','.join(used_textures)}")
        
        # Identify specific maps for pipeline
        for tex in used_textures:
            low = tex.lower()
            if any(x in low for x in ["norm", "nrm", "bump"]):
                print(f"NORMAL_MAP: {tex}")
            elif any(x in low for x in ["rough", "gloss", "spec", "rma"]):
                print(f"ROUGHNESS_MAP: {tex}")
            elif any(x in low for x in ["alb", "base", "diff", "color"]):
                print(f"ALBEDO_MAP: {tex}")
        
    if has_missing_texture:
        print("TEXTURE_STATUS: MISSING")
    else:
        print("TEXTURE_STATUS: OK")

    print("RENDER_SUCCESS")
    return True

if __name__ == "__main__":
    import argparse
    
    # Extract arguments after "--"
    if "--" not in sys.argv:
        print("Usage: blender --background --python bl_render_fast.py -- --input <file> --output <prefix>")
        sys.exit(1)
        
    argv = sys.argv[sys.argv.index("--") + 1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    
    # 백그라운드에서 강제 종료 방지용 로깅
    print(f"[*] Fast rendering started for: {args.input}")
    render_model(args.input, args.output)
