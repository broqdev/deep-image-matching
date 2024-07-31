# %%
import cv2
import numpy as np

im = np.ones((3, 4), dtype=np.uint8) * 255
im
# %%
im2 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
im2.shape
# %%
im_ = np.expand_dims(im, axis=2)
im3 = cv2.cvtColor(np.repeat(im_.astype(np.float32), 3, axis=2), cv2.COLOR_RGB2GRAY)
im3

# %%
mask_path = 'data/lab4d/cat0_small/masks/00000.jpg'
mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask_img
# %%
import numpy as np

a = np.random.rand(3, 4)
b = np.random.rand(3, 4)
c = np.vstack((a, b))
c.shape
# %%
