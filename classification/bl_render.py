import bpy
import sys
import os
import argparse
import mathutils

def setup_blender():
    # 모든 기존 개체 수동 삭제 (factory_settings 대신)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # World setup (White background)
    if not bpy.data.worlds:
        world = bpy.data.worlds.new("World")
    else:
        world = bpy.data.worlds[0]
    
    bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (1, 1, 1, 1)

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
        # FBX Material name is mat.name
        tex_path = mapper.get_texture_for_fbx_material(model_path, mat.name)
        if not tex_path or not os.path.exists(tex_path):
            continue
            
        print(f"[*] UnityTextureMapper matched: {mat.name} -> {tex_path}")
        
        # Ensure material has nodes
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Find Principled BSDF
        bsdf = None
        for n in nodes:
            if n.type == 'BSDF_PRINCIPLED':
                bsdf = n
                break
        
        if not bsdf: continue
        
        # Find or create TEX_IMAGE node
        tex_node = None
        for n in nodes:
            if n.type == 'TEX_IMAGE':
                tex_node = n
                break
                
        if not tex_node:
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (bsdf.location.x - 300, bsdf.location.y)
            
        # Try to load the image
        try:
            img = bpy.data.images.load(tex_path)
            tex_node.image = img
            links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
        except Exception as e:
            print(f"Failed to load image {tex_path}: {e}")

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
            
        relink_textures(filepath)    
        return True
    except Exception as e:
        print(f"FAILED_TO_IMPORT: {e}")
        return False

def setup_camera_and_lighting(obj):
    # 3-Point Lighting for better quality
    #Bounding Box를 기반으로 계산
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, mathutils.Vector((0,0,0))) / 8
    max_dim = max(obj.dimensions)
    
    # Key Light
    bpy.ops.object.light_add(type='SUN', location=(center.x + max_dim, center.y - max_dim, center.z + max_dim))
    key = bpy.context.active_object
    key.data.energy = 5.0
    key.rotation_euler = (0.785, 0, 0.785) # 45 degrees
    
    # Fill Light
    bpy.ops.object.light_add(type='AREA', location=(center.x - max_dim, center.y - max_dim, center.z + max_dim*0.5))
    fill = bpy.context.active_object
    fill.data.energy = 100.0
    fill.data.size = max_dim
    
    # Rim Light
    bpy.ops.object.light_add(type='SUN', location=(center.x, center.y + max_dim, center.z + max_dim*0.5))
    rim = bpy.context.active_object
    rim.data.energy = 2.0
    rim.rotation_euler = (-0.785, 0, 0)
    
    # Camera setup
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # Select the model
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

def render_views(obj, output_prefix):
    scene = bpy.context.scene
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.image_settings.file_format = 'PNG'
    
    camera = scene.camera
    cam_data = camera.data
    
    # [줌 로직] 바운딩 구체 계산
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, mathutils.Vector((0,0,0))) / 8
    radius = max([(v - center).length for v in bbox])
    
    # 10% 여백 추가 (90%만 채움)
    target_radius = radius / 0.9
    
    # 화각(FOV)에 따른 거리 계산: d = R / sin(fov/2)
    # Blender fov는 라디안 단위
    import math
    fov = cam_data.angle
    dist = target_radius / math.sin(fov / 2)
    
    # 1. Front: (0, -1, 0.1) - 약간 위에서 정면
    # 2. Back: (0, 1, 0.1) - 약간 위에서 후면
    # 3. Top: (0, 0, 1) - 정수리(위)
    # 4. Iso: (0.7, -0.7, 0.7) - 대각선 위
    
    view_dirs = [
        ("Front", mathutils.Vector((0, -1, 0.1)).normalized()),
        ("Back", mathutils.Vector((0, 1, 0.1)).normalized()),
        ("Top", mathutils.Vector((0, 0, 1)).normalized()),
        ("Iso", mathutils.Vector((0.7, -0.7, 0.7)).normalized())
    ]
    
    for i, (name, v_dir) in enumerate(view_dirs):
        # 카메라 위치 설정
        camera.location = center + v_dir * dist
        
        # 객체 중심을 바라보도록 회전
        direction = center - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
        
        scene.render.filepath = f"{output_prefix}_view{i+1}.png"
        bpy.ops.render.render(write_still=True)
        print(f"[*] Rendered {name} view → {scene.render.filepath}")

if __name__ == "__main__":
    # Custom arg parse for Blender
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    
    setup_blender()
    if import_model(args.input):
        # 첫 번째로 임포트된 메쉬 개체 찾기
        imported_objs = [o for o in bpy.data.objects if o.type == 'MESH']
        if imported_objs:
            obj = imported_objs[0]
            setup_camera_and_lighting(obj)
            render_views(obj, args.output)
            
            # [추가] 객체 크기(Dimensions) 출력
            dims = obj.dimensions
            print(f"DIMENSIONS: {dims.x:.4f},{dims.y:.4f},{dims.z:.4f}")
            
            # [추가] 사용된 텍스처 파일명 수집 및 디버깅 로그
            used_textures = []
            has_missing_texture = False
            for mat in bpy.data.materials:
                if not mat.use_nodes: continue
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE':
                        img = node.image
                        if img and getattr(img, 'filepath', ''):
                            tex_name = os.path.basename(img.filepath)
                            if not tex_name: tex_name = img.name
                            if tex_name not in used_textures:
                                used_textures.append(tex_name)
                        else:
                            has_missing_texture = True
            
            if used_textures:
                print(f"TEXTURES: {','.join(used_textures)}")
                
            if has_missing_texture:
                print("TEXTURE_STATUS: MISSING")
            else:
                print("TEXTURE_STATUS: OK")
            
            print(f"RENDER_SUCCESS: {args.output}")
        else:
            print(f"Error: No mesh found in {args.input}")
