import bpy
import sys

def check_fbx_materials(filepath):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=filepath)
    with open("check_mats.log", "a", encoding='utf-8') as f:
        f.write(f"\n--- Materials in {filepath} ---\n")
        for mat in bpy.data.materials:
            f.write(f"Material: {mat.name}\n")
            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE':
                        img = node.image
                        if img:
                            f.write(f"  -> Linked texture: {img.filepath}\n")
                        else:
                            f.write(f"  -> Unlinked texture node: {node.name}\n")

if __name__ == "__main__":
    with open("check_mats.log", "w", encoding='utf-8') as f:
        f.write("Log Start\n")
    check_fbx_materials("assets/food/model/sausages2.fbx")
    check_fbx_materials("assets/food/model/SM_chair_metalring_01.fbx")
