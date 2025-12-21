from utils.downsample import process_and_downsample_off, process_segmentation_map
from utils.mesh import mesh2off, seg2txt, mesh2ply
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

ROOT = Path("/Users/bilalabbasi/Projects/shape-analysis/data/COSEG")
CATEGORIES = ['chairs', 'vases', 'tele_aliens']
TARGET_VERTICES = 1000

DIR_NAME = ROOT.name + f"_{TARGET_VERTICES}"
TARGET_ROOT = ROOT.parent / DIR_NAME
TARGET_ROOT.mkdir(parents=True, exist_ok=True)

metadata = {}
for mode in ['train', 'test']:
    metadata[mode] = {}
    for category in CATEGORIES:
        metadata[mode][category] = {}
        category_gt_path = ROOT / mode / category / "vert_gt" 
        category_shape_path = ROOT / mode / category / "shapes" 

        target_category_gt_path = TARGET_ROOT / mode / category / "vert_gt"
        target_category_gt_path.mkdir(parents=True, exist_ok=True)
        target_category_shape_path = TARGET_ROOT / mode / category / "shapes"
        target_category_shape_path.mkdir(parents=True, exist_ok=True)
        target_category_ply_path = TARGET_ROOT / mode / category / "ply"
        target_category_ply_path.mkdir(parents=True, exist_ok=True)

        for shape_path in tqdm(sorted(category_shape_path.glob("*.off"))):
            shape = shape_path.name
            gt = shape.replace("off", "seg")
            gt_path = category_gt_path / gt
            ply = shape.replace("off", "ply")
            
            target_shape_path = target_category_shape_path / shape
            target_gt_path = target_category_gt_path / gt
            target_ply_path = target_category_ply_path / ply

            if target_ply_path.exists():
                print(f"SKIPPING {shape}")
                continue

            new_vertices, new_faces, vertex_map = process_and_downsample_off(shape_path, TARGET_VERTICES)
            _, new_gt = process_segmentation_map(gt_path, vertex_map)

            seg2txt(new_gt, fname=target_gt_path)
            mesh2off(new_vertices, new_faces, fname=target_shape_path)
            mesh2ply(new_vertices, new_faces, np.array(new_gt), fname=target_ply_path)

            metadata[mode][category][shape] = {
                "vertices": len(new_vertices),
                "faces": len(new_faces),
            }
            with open(TARGET_ROOT / "metadata.json", 'w') as j:
                json.dump(metadata, fp=j, indent=4)
