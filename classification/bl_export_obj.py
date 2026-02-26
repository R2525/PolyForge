import bpy
import sys
import os

def export_to_obj(input_fbx, output_obj):
    # 기존 객체 모두 삭제
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # FBX 임포트
    try:
        bpy.ops.import_scene.fbx(filepath=input_fbx)
    except Exception as e:
        print(f"FAILED_TO_IMPORT: {e}")
        return False
        
    # 모든 객체 선택
    bpy.ops.object.select_all(action='SELECT')
    
    # OBJ로 익스포트 (재질 포함)
    try:
        bpy.ops.wm.obj_export(
            filepath=output_obj,
            export_materials=True,
            export_triangulated_mesh=True,
            up_axis='Z',
            forward_axis='Y'
        )
        print("EXPORT_SUCCESS")
        return True
    except Exception as e:
        print(f"FAILED_TO_EXPORT: {e}")
        return False

if __name__ == "__main__":
    # 커맨드라인 인자 파싱 (-- 이후의 인자들)
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender --background --python bl_export_obj.py -- <input.fbx> <output.obj>")
        sys.exit(1)
        
    args = argv[argv.index("--") + 1:]
    if len(args) != 2:
        print("Usage: blender --background --python bl_export_obj.py -- <input.fbx> <output.obj>")
        sys.exit(1)
        
    input_file = args[0]
    output_file = args[1]
    
    export_to_obj(input_file, output_file)
