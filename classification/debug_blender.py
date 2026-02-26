import subprocess
import os

blender_exe = "C:/Program Files/Blender Foundation/Blender 4.3/blender.exe"
input_file = "assets/food/model/sausages1.fbx"
output_prefix = "debug_output/image/sausages1"

cmd = [
    blender_exe, "--background", "--python", "bl_render.py", "--",
    "--input", input_file, "--output", output_prefix
]

print(f"Running: {' '.join(cmd)}")
res = subprocess.run(cmd, capture_output=True, text=True)

with open("blender_debug_log.txt", "w", encoding="utf-8") as f:
    f.write("STDOUT:\n")
    f.write(res.stdout)
    f.write("\n\nSTDERR:\n")
    f.write(res.stderr)

print("Log saved to blender_debug_log.txt")
