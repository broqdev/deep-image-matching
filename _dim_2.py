# %%
from deep_image_matching.hloc.utils.read_write_model import read_model

p = "data/lab4d/cat0_t1/results_aliked+lightglue_bruteforce_quality_high/reconstruction"
#p = "/workspace/2024_07_03/deep-image-matching/data/lab4d/cat0_small/results_superpoint+lightglue_bruteforce_quality_high/reconstruction"

cameras, poses, points3D = read_model(p, ".bin")


# %%
from pathlib import Path
import cv2
from PIL import Image

p = "/workspace/2024_07_03/deep-image-matching/data/lab4d/cat0_t1/images"
#p = "/workspace/2024_07_03/deep-image-matching/data/lab4d/cat0_small/images"


image_paths = sorted(list(Path(p).glob("*")))

images = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    images.append(im)

# %%
import rerun as rr
import numpy as np

def vis_rerun(image_list, extrinsics, intrinsics, points, colors):

    T = len(image_list)
    H, W, _ = image_list[0].shape

    rr.init("t2")
    rr.serve(open_browser=True, web_port=8888, ws_port=6006)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)

    for t in range(T):

        rr.set_time_sequence("frame", t)

        K = np.eye(3)
        K[0, 0] = intrinsics[t][0]
        K[1, 1] = intrinsics[t][1]
        K[0, 2] = intrinsics[t][2]
        K[1, 2] = intrinsics[t][3]

        img_rgb = image_list[t]
        cams_T_world = extrinsics[t]
        rr.log(f"world/camera/image/rgb", rr.Image(img_rgb))

        rr.log(
            f"world/camera",
            rr.Transform3D(
                translation=cams_T_world[:3, 3], mat3x3=cams_T_world[:3, :3]
            ),
        )
        rr.log(
            f"world/camera",
            rr.Pinhole(
                resolution=[W, H],
                image_from_camera=K,
                camera_xyz=rr.ViewCoordinates.RDF,  # FIXME LUF -> RDF
            ),
        )

        points_=points[t]
        colors_=colors[t]
        rr.log(f"world/points", rr.Points3D(points_, colors=colors_))

# %%
import pypose as pp
#import scipy
#import pytorch3d
import torch

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])

def qvec_tvec_to_pypose_SE3(qvec, tvec):
    r = pp.SO3([qvec[1], qvec[2], qvec[3], qvec[0]])
    #print(1, r)
    # r = r.matrix().T

    # TODO: double check
    r = r.matrix().T
    #print(2, r)
    #ttt = qvec2rotmat(qvec)
    #print(3, ttt.T)

    rr = torch.eye(4)
    rr[:3, :3] = r
    rr[:3, 3] = torch.tensor([tvec[0], -tvec[1], tvec[2]])
    print(4, rr, rr.shape)

    # init = np.concatenate([tvec, qvec[3:], qvec[:3]], axis=-1)
    # print(init)
    # tmp = pp.SE3(rr)
    tmp = pp.mat2SE3(rr)
    #return tmp.matrix()
    print(5, tmp)
    return tmp

extrinsics = []
intrinsics = []
points = []
colors = []

poses_ = list(poses.values())

for image_path in image_paths:
    image_name = image_path.name

    pose = list(filter(lambda p: p.name == str(image_name), poses_))
    assert len(pose) == 1, f"{image_name} has {len(pose)} poses"
    pose_ = pose[0]
    print(111, pose_)
    print(112, pose_.qvec)
    print(113, pose_.tvec)

    pose_lie = qvec_tvec_to_pypose_SE3(pose_.qvec, pose_.tvec)
    print(123, pose_lie.matrix())
    extrinsics.append(pose_lie.matrix())

    camera_ = cameras[pose_.camera_id]
    intrinsics.append(camera_.params)

    ids = pose_.point3D_ids[pose_.point3D_ids != -1]
    points_ = []
    colors_ = []
    for id in ids:
        point3D = points3D[id]
        points_.append(point3D.xyz)
        colors_.append(point3D.rgb)

    points.append(np.array(points_))
    colors.append(np.array(colors_))
    #break

# %%
vis_rerun(images, extrinsics, intrinsics, points, colors)

# %%
import pypose as pp
qvec = poses[51].qvec
tvec = poses[51].tvec
pp.SO3([qvec[1], qvec[2], qvec[3], qvec[0]]).matrix()
# %%
