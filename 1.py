# %%
from pathlib import Path
from deep_image_matching.utils.image import ImageList

p = 'data/lab4d/cat0_small/'
image_dir = Path(p, 'images')
mask_dir = Path(p, 'masks')

ImageList(image_dir, mask_dir)
# %%
!python main.py --dir /workspace/2024_05_08/nerfstudio/data/colmap/cat0 --masks /workspace/2024_05_08/nerfstudio/data/colmap/cat0/masks_bg --mask_type bg --pipeline superpoint+lightglue --force -s bruteforce
# %%
!python main.py --dir /workspace/2024_05_08/nerfstudio/data/colmap/cat0_small --masks /workspace/2024_05_08/nerfstudio/data/colmap/cat0_small/masks_bg --mask_type bg --pipeline superpoint+lightglue --force -s bruteforce

# %%
!python main.py --dir /workspace/2024_05_08/nerfstudio/data/colmap/cat0 --masks /workspace/2024_05_08/nerfstudio/data/colmap/cat0/masks_bg --mask_type bg --pipeline loftr -s bruteforce --force

# %%
!python main.py --dir /workspace/2024_05_08/nerfstudio/data/colmap/cat0_sparse --masks /workspace/2024_05_08/nerfstudio/data/colmap/cat0_sparse/masks_bg --mask_type bg -p loftr -c config/loftr.yaml -s bruteforce --force
# %%
!python main.py --dir /workspace/2024_05_08/nerfstudio/data/colmap/cat0_sparse --masks /workspace/2024_05_08/nerfstudio/data/colmap/cat0_sparse/masks_bg --mask_type bg -p roma -c config/roma.yaml -s bruteforce --force
# %%
