from torch.utils.data import Dataset
from utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
from data.ray_utils import *
import math
import trimesh
from pathlib import Path
import imageio
from tqdm import tqdm

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
scale_factor = 1.0 / 200
downsample=1.0
img_wh = (int(640*downsample),int(512*downsample))
w, h = img_wh
def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0]) * scale_factor
    depth_max = depth_min + float(lines[11].split()[1]) * 192 * scale_factor
    depth_interval = float(lines[11].split()[1])

    # scaling
    extrinsics[:3, 3] *= scale_factor
    intrinsics[0:2] *= downsample

    return intrinsics, extrinsics, [depth_min, depth_max]


def read_depth(filename):
    depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)
    # print(depth_h.shape)
    depth_h = cv2.resize(depth_h, (640, 512), interpolation=cv2.INTER_NEAREST)  # (600, 800)
    # depth_h = depth_h[44:556, 80:720]  # (512, 640)
    
    depth_h = cv2.resize(depth_h, None, fx=downsample, fy=downsample,
                            interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!

    return depth_h

def read_mask(filename):
    mask = imageio.imread(filename) / 255.
    mask = cv2.resize(mask, (640, 512), interpolation=cv2.INTER_NEAREST) 
    
    mask = mask > 0.5

    return mask



root_dir = '/rds/project/rds-JDeuXlFW9KE/data/dtu'
with open(f'configs/lists/dtu_train_all.txt') as f:
    scanlist = [line.rstrip() for line in f.readlines()]
with open(f'configs/lists/dtu_test_all.txt') as f:
    scanlist = scanlist + [line.rstrip() for line in f.readlines()]


MASK_TYPE = 'photo' # no_mask, final, photo
PT_DOWNSAMPLE = 5

# scanlist=['scan1']
print(len(scanlist))
for scan in tqdm(scanlist):
    pointclouds = []
    # for idx in range(49):
    for idx in range(49):
        proj_mat_filename = os.path.join(root_dir, f'Cameras/train/{idx:08d}_cam.txt')
        intrinsic, w2c, near_far = read_cam_file(proj_mat_filename)
        c2w = np.linalg.inv(w2c)
        c2w = torch.FloatTensor(c2w)
        intrinsic[:2] *= 4 # * the provided intrinsics is 4x downsampled, now keep the same scale with image
        center = [intrinsic[0,2], intrinsic[1,2]]
        focal = [intrinsic[0,0], intrinsic[1,1]]
        directions = get_ray_directions(h, w, focal, center)  # (h, w, 3)
        rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
        depth_filename = f"/rds/project/rds-JDeuXlFW9KE/tw554/TransMVSNet/outputs/dtu_testing/{scan}/depth_est/{idx:08d}.pfm"
        image_path = os.path.join(root_dir,
                                    f'Rectified/{scan}_train/rect_{idx + 1:03d}_3_r5000.png')
        img = Image.open(image_path)
        rgb_numpy = (np.array(img) / 255.0).reshape(-1,3)
        # print(depth_filename)
        if os.path.exists(depth_filename):
            depth = read_depth(depth_filename)
            depth *= scale_factor
            depth = depth.reshape(-1)

            # read mask
            if MASK_TYPE == 'no_mask':
                mask = np.ones_like(depth).astype(bool)
            else:
                if MASK_TYPE == 'final':
                    mask_path = f"/rds/project/rds-JDeuXlFW9KE/tw554/TransMVSNet/outputs/dtu_testing/{scan}/mask/{idx:08d}_final.png"
                elif MASK_TYPE == 'photo':
                    mask_path = f"/rds/project/rds-JDeuXlFW9KE/tw554/TransMVSNet/outputs/dtu_testing/{scan}/mask/{idx:08d}_photo.png"
                mask = read_mask(filename=mask_path).reshape(-1)


            point_samples = rays_o.numpy()[mask] + np.expand_dims(depth[mask],-1) * rays_d.numpy()[mask]
            pointclouds.append(np.concatenate([point_samples,rgb_numpy[mask]],axis=-1))
    if len(pointclouds)!=0:
        pointclouds = np.vstack(pointclouds)
        pointclouds = pointclouds[::5]

        out_path = Path(root_dir) / f'pointclouds{PT_DOWNSAMPLE}_transmvs_{MASK_TYPE}_ply'
        out_path.mkdir(exist_ok=True)

        trimesh.PointCloud(pointclouds[:,:3], colors=pointclouds[:,3:6]).export(str(out_path/f'{scan}_pointclouds.ply'))
        init_pointclouds = torch.tensor(pointclouds).float()
        print('init_pointclouds shape',init_pointclouds.shape) #114 [220212, 6]