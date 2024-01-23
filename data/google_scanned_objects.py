import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import sys
import cv2
from scipy.spatial.transform import Rotation as R

import math
from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms as T
from .ray_utils import *
from pathlib import Path
import trimesh



rng = np.random.RandomState(234)
_EPS = np.finfo(float).eps * 4.0
TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision


def PILtoTorch(pil_image, resolution=None):
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def vector_norm(data, axis=None, out=None):
    """Return length, i.e. eucledian norm, of ndarray along axis."""
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis."""
    quaternion = np.zeros((4,), dtype=np.float64)
    quaternion[:3] = axis[:3]
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle / 2.0) / qlen
    quaternion[3] = math.cos(angle / 2.0)
    return quaternion


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion."""
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(
        (
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def rectify_inplane_rotation(src_pose, tar_pose, src_img, th=40):
    relative = np.linalg.inv(tar_pose).dot(src_pose)
    relative_rot = relative[:3, :3]
    r = R.from_matrix(relative_rot)
    euler = r.as_euler("zxy", degrees=True)
    euler_z = euler[0]
    if np.abs(euler_z) < th:
        return src_pose, src_img

    R_rectify = R.from_euler("z", -euler_z, degrees=True).as_matrix()
    src_R_rectified = src_pose[:3, :3].dot(R_rectify)
    out_pose = np.eye(4)
    out_pose[:3, :3] = src_R_rectified
    out_pose[:3, 3:4] = src_pose[:3, 3:4]
    h, w = src_img.shape[:2]
    center = ((w - 1.0) / 2.0, (h - 1.0) / 2.0)
    M = cv2.getRotationMatrix2D(center, -euler_z, 1)
    src_img = np.clip((255 * src_img).astype(np.uint8), a_max=255, a_min=0)
    rotated = cv2.warpAffine(
        src_img, M, (w, h), borderValue=(255, 255, 255), flags=cv2.INTER_LANCZOS4
    )
    rotated = rotated.astype(np.float32) / 255.0
    return out_pose, rotated


def random_crop(rgb, camera, src_rgbs, src_cameras, size=(400, 600), center=None):
    h, w = rgb.shape[:2]
    out_h, out_w = size[0], size[1]
    if out_w >= w or out_h >= h:
        return rgb, camera, src_rgbs, src_cameras

    if center is not None:
        center_h, center_w = center
    else:
        center_h = np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)
        center_w = np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)

    rgb_out = rgb[
        center_h - out_h // 2 : center_h + out_h // 2,
        center_w - out_w // 2 : center_w + out_w // 2,
        :,
    ]
    src_rgbs = np.array(src_rgbs)
    src_rgbs = src_rgbs[
        :,
        center_h - out_h // 2 : center_h + out_h // 2,
        center_w - out_w // 2 : center_w + out_w // 2,
        :,
    ]
    camera[0] = out_h
    camera[1] = out_w
    camera[4] -= center_w - out_w // 2
    camera[8] -= center_h - out_h // 2
    src_cameras[:, 4] -= center_w - out_w // 2
    src_cameras[:, 8] -= center_h - out_h // 2
    src_cameras[:, 0] = out_h
    src_cameras[:, 1] = out_w
    return rgb_out, camera, src_rgbs, src_cameras


def random_flip(rgb, camera, src_rgbs, src_cameras):
    h, w = rgb.shape[:2]
    h_r, w_r = src_rgbs.shape[1:3]
    rgb_out = np.flip(rgb, axis=1).copy()
    src_rgbs = np.flip(src_rgbs, axis=-2).copy()
    camera[2] *= -1
    camera[4] = w - 1.0 - camera[4]
    src_cameras[:, 2] *= -1
    src_cameras[:, 4] = w_r - 1.0 - src_cameras[:, 4]
    return rgb_out, camera, src_rgbs, src_cameras


def get_color_jitter_params(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    color_jitter = transforms.ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )
    transform = transforms.ColorJitter.get_params(
        color_jitter.brightness, color_jitter.contrast, color_jitter.saturation, color_jitter.hue
    )
    return transform


def color_jitter(img, transform):
    """
    Args:
        img: np.float32 [h, w, 3]
        transform:
    Returns: transformed np.float32
    """
    img = Image.fromarray((255.0 * img).astype(np.uint8))
    img_trans = transform(img)
    img_trans = np.array(img_trans).astype(np.float32) / 255.0
    return img_trans


def color_jitter_all_rgbs(rgb, ref_rgbs, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    transform = get_color_jitter_params(brightness, contrast, saturation, hue)
    rgb_trans = color_jitter(rgb, transform)
    ref_rgbs_trans = []
    for ref_rgb in ref_rgbs:
        ref_rgbs_trans.append(color_jitter(ref_rgb, transform))

    ref_rgbs_trans = np.array(ref_rgbs_trans)
    return rgb_trans, ref_rgbs_trans


def deepvoxels_parse_intrinsics(filepath, trgt_sidelength, invert_y=False):
    # Get camera intrinsics
    with open(filepath, "r") as file:
        f, cx, cy = list(map(float, file.readline().split()))[:3]
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        near_plane = float(file.readline())
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    cx = cx / width * trgt_sidelength
    cy = cy / height * trgt_sidelength
    f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0], [0.0, 0, 1, 0], [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses


def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit * vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    """
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    """
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(
        np.clip(
            (np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.0,
            a_min=-1 + TINY_NUMBER,
            a_max=1 - TINY_NUMBER,
        )
    )


def get_nearest_pose_ids(
    tar_pose,
    ref_poses,
    num_select,
    tar_id=-1,
    angular_dist_method="vector",
    scene_center=(0, 0, 0),
):
    """
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    """
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams - 1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == "matrix":
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == "vector":
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == "dist":
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception("unknown angular distance calculation method!")

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids

# only for training
class GoogleScannedDataset(Dataset):
    def __init__(self, args=None, split='train', n_views=3, max_len=-1, **kwargs):
        self.folder_path = args.datadir
        self.pointcloud_dir = self.folder_path+'_pts'
        self.num_source_views = n_views
        # self.rectify_inplane_rotation = args.rectify_inplane_rotation
        self.rectify_inplane_rotation = False
        self.scene_path_list = glob.glob(os.path.join(self.folder_path, "*"))
        self.split = split
        assert self.split in ['train', 'val', 'test'], \
            'split must be either "train", "val" or "test"!'

        all_rgb_files = []
        all_pose_files = []
        all_intrinsics_files = []
        num_files = 250
        for i, scene_path in enumerate(self.scene_path_list):
            rgb_files = [
                os.path.join(scene_path, "rgb", f)
                for f in sorted(os.listdir(os.path.join(scene_path, "rgb")))
            ]
            pose_files = [f.replace("rgb", "pose").replace("png", "txt") for f in rgb_files]
            intrinsics_files = [
                f.replace("rgb", "intrinsics").replace("png", "txt") for f in rgb_files
            ]

            if np.min([len(rgb_files), len(pose_files), len(intrinsics_files)]) < num_files:
                print(scene_path)
                continue

            all_rgb_files.append(rgb_files)
            all_pose_files.append(pose_files)
            all_intrinsics_files.append(intrinsics_files)

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files)[index]
        self.all_pose_files = np.array(all_pose_files)[index]
        self.all_intrinsics_files = np.array(all_intrinsics_files)[index]

        self.transform = T.ToTensor()
        self.src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.n_views = n_views


    def __len__(self):
        return len(self.all_rgb_files)
    
    def focal2fov(self,focal, pixels):
        return 2*math.atan(pixels/(2*focal))
    

    def get_rays_single_image(self, H, W, intrinsics, c2w):
        """
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        """
        u, v = np.meshgrid(
            np.arange(W), np.arange(H)
        )
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels)
        batched_pixels = pixels.unsqueeze(0).repeat(1, 1, 1)

        c2w = torch.from_numpy(c2w).type_as(batched_pixels)

        rays_d = (
            c2w[:, :3, :3].bmm(torch.inverse(torch.from_numpy(intrinsics[:, :3, :3]).type_as(batched_pixels))).bmm(batched_pixels)
        ).transpose(1, 2)
        rays_d = rays_d.reshape(-1, 3)
        rays_d = rays_d / rays_d.norm(dim=-1)[:, None]
        rays_o = (
            c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)
        )  # B x HW x 3
        return rays_o, rays_d
    

    def read_source_views(self, obj_idx=None, pair_idx=None, device=torch.device("cpu"), **kwargs):
        rgb_files = self.all_rgb_files[obj_idx]
        pose_files = self.all_pose_files[obj_idx]
        intrinsics_files = self.all_intrinsics_files[obj_idx]

        src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        for i,idx in enumerate(pair_idx[0]):
            idx = idx.detach().cpu().numpy()

            c2w = np.loadtxt(pose_files[idx]).reshape([4,4])
            w2c = np.linalg.inv(c2w)
            intrinsic = np.loadtxt(intrinsics_files[obj_idx]).reshape([4,4])[:3,:3]


            c2ws.append(c2w)
            w2cs.append(w2c)

            # build proj mat from source views to ref view
            proj_mat_l = np.eye(4)
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_l)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]
            intrinsics.append(intrinsic.copy())

            # image_path = os.path.join(self.folder_path,
            #                             f'{scan}/rgb/{idx:06d}.png')
            image_path = rgb_files[idx]

            img = Image.open(image_path)
            img = self.transform(img)
            imgs.append(src_transform(img))

        pose_source = {}
        pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
        pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
        pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).float().to(device)

        imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)

        return imgs, proj_mats, None, pose_source
    
    def read_source_views_mvs(self, obj_idx=None, pair_idx=None, device=torch.device("cpu"), **kwargs):
        # very nasty work around for train_mvs_nerf_pl.py
        # which requires different data loader format
        rgb_files = self.all_rgb_files[obj_idx]
        pose_files = self.all_pose_files[obj_idx]
        intrinsics_files = self.all_intrinsics_files[obj_idx]

        src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        near_fars = []
        for i,idx in enumerate(pair_idx[0]):
            idx = idx.detach().cpu().numpy()

            c2w = np.loadtxt(pose_files[idx]).reshape([4,4])
            w2c = np.linalg.inv(c2w)
            intrinsic = np.loadtxt(intrinsics_files[obj_idx]).reshape([4,4])[:3,:3]


            c2ws.append(c2w)
            w2cs.append(w2c)

            # build proj mat from source views to ref view
            proj_mat_l = np.eye(4)
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_l)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]
            intrinsics.append(intrinsic.copy())

            # image_path = os.path.join(self.folder_path,
            #                             f'{scan}/rgb/{idx:06d}.png')
            image_path = rgb_files[idx]

            img = Image.open(image_path)
            img = self.transform(img)
            imgs.append(src_transform(img))

            # get depth range
            min_ratio = 0.1
            origin_depth = np.linalg.inv(c2w)[2, 3]
            max_radius = 0.5 * np.sqrt(2) * 1.1
            near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
            far_depth = origin_depth + max_radius
            near_fars.append([near_depth, far_depth])

        pose_source = {}
        pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
        pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
        pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).float().to(device)


        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)

        imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        proj_mats = np.stack(proj_mats)[:,:3]

        # very nasty work around for train_mvs_nerf_pl.py
        # which requires different data loader format
        sample = {
            'images': imgs,  # (V, H, W, 3) V=4
            'w2cs': w2cs.astype(np.float32),  # (V, 4, 4)
            'c2ws': c2ws.astype(np.float32),  # (V, 4, 4)
            'near_fars': near_fars.astype(np.float32),
            'proj_mats': proj_mats.astype(np.float32),
            'intrinsics': intrinsics.astype(np.float32),  # (V, 3, 3)
        }


        return sample

    def __getitem__(self, idx):
        rgb_files = self.all_rgb_files[idx]
        pose_files = self.all_pose_files[idx]
        intrinsics_files = self.all_intrinsics_files[idx]

        id_render = np.random.choice(np.arange(len(rgb_files)))
        train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
        render_pose = train_poses[id_render]
        # subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])
        subsample_factor = 1 # for now, only select N nearest view

        id_feat_pool = get_nearest_pose_ids(
            render_pose,
            train_poses,
            self.num_source_views * subsample_factor,
            tar_id=id_render,
            angular_dist_method="vector",
        )
        id_feat = np.random.choice(id_feat_pool, self.num_source_views, replace=False)

        assert id_render not in id_feat
        # # occasionally include input image
        # if np.random.choice([0, 1], p=[0.995, 0.005]):
        #     id_feat[np.random.choice(len(id_feat))] = id_render


        image_path = rgb_files[id_render]
        depth_filename = image_path.replace('/rgb/', '/depth/')
        img = Image.open(image_path)
        img = PILtoTorch(img) #[3,H,W]
        _, h,w = img.shape
        target_rgbs = img

        obj_name = Path(image_path).parent.parent.name

        depth = imageio.imread(depth_filename)
        target_depth = torch.from_numpy(depth).float()
        target_mask = torch.tensor(depth>0)

        # get depth range
        min_ratio = 0.1
        origin_depth = np.linalg.inv(render_pose)[2, 3]
        max_radius = 0.5 * np.sqrt(2) * 1.1
        near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
        far_depth = origin_depth + max_radius
        near_far = torch.tensor([near_depth, far_depth])

        # get R and T
        intrinsic = np.loadtxt(intrinsics_files[id_render]).reshape([4,4])
        c2w = render_pose
        w2c = np.linalg.inv(render_pose)

        center = [intrinsic[0,2], intrinsic[1,2]]
        focal = [intrinsic[0,0], intrinsic[1,1]]

        # directions = get_ray_directions(h, w, focal, center)
        # rays_o, rays_d = get_rays(directions, c2w)


        rays_o, rays_d = self.get_rays_single_image(h, w, intrinsic[None,...], c2w[None,...])



        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]
        focal_length_x = intrinsic[0,0]
        focal_length_y = intrinsic[1,1]
        FovY = self.focal2fov(focal_length_y, h)
        FovX = self.focal2fov(focal_length_x, w)
        target_R=torch.tensor(R).float()
        target_T=torch.tensor(T).float()
        target_FovX=torch.tensor(FovX).float()
        target_FovY=torch.tensor(FovY).float()
        target_rays = torch.cat([rays_o, rays_d,
                                    near_far[0] * torch.ones_like(rays_o[:, :1]),
                                    near_far[1] * torch.ones_like(rays_o[:, :1])],
                                1)
    

        self.extract_points(idx)

        sample = {'rays': target_rays,
                  'rgbs': target_rgbs,
                  'R':target_R,
                  'T':target_T,
                  'mask':target_mask,
                  'FovX':target_FovX,
                  'FovY':target_FovY,
                  'src_views':id_feat_pool,
                  'obj_name':obj_name,
                  'obj_idx':idx,
                  'near_far': near_far,
                #   'light_idx':light_idx,
                #   'scan':scan,
                  }
        

        return sample
    

    # def extract_points(self, obj_idx):
    #     obj_name = Path(self.all_rgb_files[obj_idx][0]).parent.parent.name
    #     pt_path = Path(self.pointcloud_dir) / 'Pointclouds10' / f'{obj_name}_pointclouds.npy' 

    #     # if pt_path.exists():
    #     #     return
        
    #     print(f'Creating pt clouds at {str(pt_path)}')

    #     pt_path.parent.mkdir(exist_ok=True, parents=True)

    #     rgb_files = self.all_rgb_files[obj_idx]
    #     pose_files = self.all_pose_files[obj_idx]
    #     intrinsics_files = self.all_intrinsics_files[obj_idx]

    #     all_pts = []
    #     for i in range(len(rgb_files)):
    #         rgb_file = rgb_files[i]
    #         depth_file = rgb_file.replace('/rgb/', '/depth/')

    #         depth = imageio.imread(depth_file).reshape([-1,1]) / 1024.
    #         h,w = depth.shape

    #         intrinsic = np.loadtxt(intrinsics_files[i]).reshape([4,4])
    #         c2w = np.loadtxt(pose_files[i]).reshape([4,4])
    #         w2c = np.linalg.inv(c2w)

    #         center = [intrinsic[0,2], intrinsic[1,2]]
    #         focal = [intrinsic[0,0], intrinsic[1,1]]
    #         rays_o, rays_d = self.get_rays_single_image(h, w, intrinsic[None,...], c2w[None,...])

    #         depth_mask = depth[...,0] > 0

    #         pts = rays_o[depth_mask] + rays_d[depth_mask] * depth[depth_mask]

    #         all_pts.append(pts.numpy()[::10])

    #     all_pts = np.concatenate(all_pts,axis=0)

    #     np.save(str(pt_path), all_pts)
    #     trimesh.PointCloud(all_pts).export(str(pt_path).replace('.npy','.ply'))


    def extract_points(self, obj_idx):
        obj_name = Path(self.all_rgb_files[obj_idx][0]).parent.parent.name
        pt_path = Path(self.pointcloud_dir) / 'Pointclouds10' / f'{obj_name}_pointclouds.npy' 

        if pt_path.exists():
            return
        
        print(f'Creating pt clouds at {str(pt_path)}')

        mesh_path = Path(f'/rds/project/rds-qxpdOeYWi78/dataset/google_scanned_obj/obj/{obj_name}/meshes/model.obj')

        mesh = trimesh.load_mesh(str(mesh_path))

        all_pts = np.array(mesh.vertices)

        np.save(str(pt_path), all_pts)
        # trimesh.PointCloud(all_pts).export(str(pt_path).replace('.npy','.ply'))


        # rgb_files = self.all_rgb_files[obj_idx]
        # pose_files = self.all_pose_files[obj_idx]
        # intrinsics_files = self.all_intrinsics_files[obj_idx]

        # all_pts = []
        # for i in range(len(rgb_files)):
        #     rgb_file = rgb_files[i]
        #     depth_file = rgb_file.replace('/rgb/', '/depth/')

        #     depth = imageio.imread(depth_file).reshape([-1,1]) / 1024.
        #     h,w = depth.shape

        #     intrinsic = np.loadtxt(intrinsics_files[i]).reshape([4,4])
        #     c2w = np.loadtxt(pose_files[i]).reshape([4,4])
        #     w2c = np.linalg.inv(c2w)

        #     rays_o, rays_d = self.get_rays_single_image(h, w, intrinsic[None,...], c2w[None,...])

        #     all_pts.append(rays_o.numpy()[0])

        # all_pts = np.stack(all_pts,axis=0)

        # trimesh.PointCloud(all_pts).export(str(pt_path.parent / 'camera.ply'))







