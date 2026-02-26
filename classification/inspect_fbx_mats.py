import bpy
import os
import sys

def inspect_fbx(fbx_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if not os.path.exists(fbx_path):
        print(f"ERROR: {fbx_path} not found")
        return
    
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    
    print("\n--- MATERIAL INSPECTION ---")
    for mat in bpy.data.materials:
        print(f"Material: {mat.name}")
        if not mat.use_nodes:
            print("  (No Nodes)")
            continue
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE':
                img = node.image
                if img:
                    print(f"  - Node {node.name}: Image {img.name} ({img.filepath})")
                else:
                    print(f"  - Node {node.name}: No Image")
    print("--- END ---")

if __name__ == "__main__":
    inspect_fbx(sys.argv[-1])
