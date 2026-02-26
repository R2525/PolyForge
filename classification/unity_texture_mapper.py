import os
import re

class UnityTextureMapper:
    def __init__(self, base_dir):
        self.base_dir = os.path.abspath(base_dir)
        self.tex_guid_to_path = {}
        self.mat_guid_to_path = {}
        self.mat_path_to_tex_guids = {} # Dictionary of GUIDs per material
        self.mat_name_to_path = {}
        
        self._scan_directories()

    def _scan_directories(self):
        # Scan for textures
        tex_dirs = [os.path.join(self.base_dir, d) for d in ['texture', 'textures', '../texture', '../textures']]
        # Additional search across entire base_dir just in case
        tex_dirs.append(self.base_dir)
        
        scanned_dirs = set()
        
        # 1. Build Texture Mappings
        for tdir in tex_dirs:
            tdir = os.path.abspath(tdir)
            if not os.path.exists(tdir) or tdir in scanned_dirs: continue
            scanned_dirs.add(tdir)
            
            for root, _, files in os.walk(tdir):
                for f in files:
                    if f.endswith('.meta') and not f.endswith('.mat.meta') and not f.endswith('.fbx.meta'):
                        meta_path = os.path.join(root, f)
                        try:
                            with open(meta_path, 'r', encoding='utf-8') as mf:
                                match = re.search(r"guid:\s*([a-fA-F0-9]+)", mf.read())
                                if match:
                                    tex_path = os.path.abspath(os.path.join(root, f[:-5])) # remove .meta
                                    if os.path.exists(tex_path):
                                        self.tex_guid_to_path[match.group(1)] = tex_path
                        except Exception:
                            pass

        # 2. Build Material Mappings
        mat_dirs = [os.path.join(self.base_dir, d) for d in ['mat', 'mats', 'material', 'materials', '../mat', '../materials', '']]
        scanned_dirs.clear()
        for mdir in mat_dirs:
            mdir = os.path.abspath(mdir)
            if not os.path.exists(mdir) or mdir in scanned_dirs: continue
            scanned_dirs.add(mdir)
            
            for root, _, files in os.walk(mdir):
                for f in files:
                    if f.endswith('.mat.meta'):
                        meta_path = os.path.join(root, f)
                        try:
                            with open(meta_path, 'r', encoding='utf-8') as mf:
                                match = re.search(r"guid:\s*([a-fA-F0-9]+)", mf.read())
                                if match:
                                    mat_path = os.path.abspath(os.path.join(root, f[:-5]))
                                    self.mat_guid_to_path[match.group(1)] = mat_path
                        except Exception:
                            pass
                            
                    elif f.endswith('.mat'):
                        mat_path = os.path.abspath(os.path.join(root, f))
                        mat_name = os.path.splitext(f)[0]
                        self.mat_name_to_path[mat_name] = mat_path
                        try:
                            with open(mat_path, 'r', encoding='utf-8') as mf:
                                content = mf.read()
                                guids_dict = {}
                                
                                # 1. Albedo / MainTex / BaseMap
                                albedo_match = re.search(r"-\s*_(?:Albedo|MainTex|BaseColor|BaseMap|Color)\s*:\s*\n\s*m_Texture:\s*\{.*?guid:\s*([a-fA-F0-9]+)", content, re.IGNORECASE)
                                if albedo_match:
                                    guids_dict["albedo"] = albedo_match.group(1)
                                
                                # 2. Normal / Bump
                                normal_match = re.search(r"-\s*_(?:BumpMap|NormalMap|DetailNormalMap)\s*:\s*\n\s*m_Texture:\s*\{.*?guid:\s*([a-fA-F0-9]+)", content, re.IGNORECASE)
                                if normal_match:
                                    guids_dict["normal"] = normal_match.group(1)
                                    
                                # 3. Metallic / Roughness / Specular / RMA
                                rough_match = re.search(r"-\s*_(?:MetallicGlossMap|SpecGlossMap|RoughnessMap|GlossMap|MetallicMap|OcclusionMap)\s*:\s*\n\s*m_Texture:\s*\{.*?guid:\s*([a-fA-F0-9]+)", content, re.IGNORECASE)
                                if rough_match:
                                    guids_dict["roughness"] = rough_match.group(1)
                                
                                if guids_dict:
                                    self.mat_path_to_tex_guids[mat_path] = guids_dict
                                else:
                                    # Very broad fallback: Find all textures and categorize by filename
                                    all_guids = re.findall(r"m_Texture:\s*\{.*?guid:\s*([a-fA-F0-9]+)", content)
                                    for g in all_guids:
                                        t_path = self.tex_guid_to_path.get(g)
                                        if not t_path: continue
                                        
                                        lower_path = t_path.lower()
                                        if any(x in lower_path for x in ["alb", "diff", "base", "color"]) and "albedo" not in guids_dict:
                                            guids_dict["albedo"] = g
                                        elif any(x in lower_path for x in ["nrm", "norm", "bump"]) and "normal" not in guids_dict:
                                            guids_dict["normal"] = g
                                        elif any(x in lower_path for x in ["rough", "gloss", "spec", "met", "rma"]) and "roughness" not in guids_dict:
                                            guids_dict["roughness"] = g
                                            
                                    if guids_dict:
                                        self.mat_path_to_tex_guids[mat_path] = guids_dict
                        except Exception:
                            pass

    def get_textures_for_fbx_material(self, fbx_path, fbx_mat_name):
        fbx_path = os.path.abspath(fbx_path)
        fbx_meta = fbx_path + '.meta'
        
        guids_dict = None
        
        # 1. Parse fbx.meta for explicit externalObjects mapping
        if os.path.exists(fbx_meta):
            try:
                with open(fbx_meta, 'r', encoding='utf-8') as mf:
                    content = mf.read()
                
                blocks = content.split("name: ")
                for block in blocks[1:]:
                    if block.strip().startswith(fbx_mat_name):
                        match = re.search(r"guid:\s*([a-fA-F0-9]+)", block)
                        if match:
                            mat_guid = match.group(1)
                            if mat_guid in self.mat_guid_to_path:
                                mat_path = self.mat_guid_to_path[mat_guid]
                                guids_dict = self.mat_path_to_tex_guids.get(mat_path)
                                if guids_dict: break
            except Exception as e:
                print(f"Error parsing FBX meta: {e}")
                
        # 2. Fallback to direct material name matching
        if not guids_dict and fbx_mat_name in self.mat_name_to_path:
            mat_path = self.mat_name_to_path[fbx_mat_name]
            guids_dict = self.mat_path_to_tex_guids.get(mat_path)
        
        if guids_dict:
            res_paths = {}
            for key, guid in guids_dict.items():
                p = self.tex_guid_to_path.get(guid)
                if p:
                    res_paths[key] = p
            return res_paths
                
        return {}

    # Legacy method for backward compatibility
    def get_texture_for_fbx_material(self, fbx_path, fbx_mat_name):
        res = self.get_textures_for_fbx_material(fbx_path, fbx_mat_name)
        return res.get("albedo") or next(iter(res.values()), None) if res else None

if __name__ == "__main__":
    import sys
    mapper = UnityTextureMapper("assets/food/model")
    print("Texture GUIDs mapped:", len(mapper.tex_guid_to_path))
    print("Material GUIDs mapped:", len(mapper.mat_guid_to_path))
    print("Material Paths to Textures:", len(mapper.mat_path_to_tex_guid))
    
    test_mats = ["food6", "MI_trim_01", "MI_ext_parts_01"]
    for m in test_mats:
        res = mapper.get_texture_for_fbx_material("assets/food/model/sausages2.fbx", m)
        print(f"Mapping '{m}' -> {res}")
