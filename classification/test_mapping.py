import os
import re

def test_mapping(base_dir):
    print(f"Testing in {base_dir}")
    guid_to_tex = {}
    tex_dirs = [os.path.join(base_dir, d) for d in ['texture', 'textures']]
    for tdir in tex_dirs:
        if os.path.exists(tdir):
            for root, dirs, files in os.walk(tdir):
                for f in files:
                    if f.endswith('.meta'):
                        meta_path = os.path.join(root, f)
                        with open(meta_path, 'r', encoding='utf-8') as mf:
                            content = mf.read()
                            match = re.search(r"guid:\s*([a-fA-F0-9]+)", content)
                            if match:
                                guid = match.group(1)
                                tex_file = f[:-5] # remove .meta
                                guid_to_tex[guid] = os.path.join(root, tex_file)
                                
    print(f"Found {len(guid_to_tex)} texture GUIDs")
    for g, t in list(guid_to_tex.items())[:3]:
        print(f"  {g} -> {t}")
        
    mat_to_tex = {}
    mat_dirs = [os.path.join(base_dir, d) for d in ['mat', 'mats', 'material', 'materials']]
    for mdir in mat_dirs:
        if os.path.exists(mdir):
            for root, dirs, files in os.walk(mdir):
                for f in files:
                    if f.endswith('.mat'):
                        mat_name = os.path.splitext(f)[0]
                        mat_path = os.path.join(root, f)
                        with open(mat_path, 'r', encoding='utf-8') as mf:
                            content = mf.read()
                            
                            # Try to find any texture GUID
                            guids = re.findall(r"m_Texture:\s*\{.*?guid:\s*([a-fA-F0-9]+)", content)
                            if guids:
                                matched = False
                                for g in guids:
                                    if g in guid_to_tex:
                                        mat_to_tex[mat_name.lower()] = guid_to_tex[g]
                                        print(f"Matched: {mat_name} -> {os.path.basename(guid_to_tex[g])}")
                                        matched = True
                                        break
                                if not matched:
                                    print(f"Failed to match {mat_name}: Found GUIDs {guids} but none exist in texture folder.")
                            else:
                                print(f"Failed to map {mat_name}: No texture GUIDs found in .mat file.")

    print(f"\nTotal exact mappings found: {len(mat_to_tex)}")
    
if __name__ == '__main__':
    test_mapping('assets/food')
