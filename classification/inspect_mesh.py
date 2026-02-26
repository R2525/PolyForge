import open3d as o3d
import numpy as np
import sys

def inspect_mesh(path):
    print(f"Inspecting {path}...")
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_triangles():
        print("No triangles found.")
        return
    
    bounds = mesh.get_axis_aligned_bounding_box()
    min_bound = bounds.get_min_bound()
    max_bound = bounds.get_max_bound()
    center = bounds.get_center()
    extent = bounds.get_max_extent()
    
    print(f"Min bound: {min_bound}")
    print(f"Max bound: {max_bound}")
    print(f"Center: {center}")
    print(f"Max extent: {extent}")
    
    vertices = np.asarray(mesh.vertices)
    print(f"Vertex count: {len(vertices)}")
    
    if len(vertices) > 0:
        std = np.std(vertices, axis=0)
        print(f"Vertex STD: {std}")
        
if __name__ == "__main__":
    inspect_mesh(sys.argv[1])
