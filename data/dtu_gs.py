
from torch.utils.data import Dataset
from utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
from .ray_utils import *
import math
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
class DTU_gs(Dataset):
    def __init__(self, args, split='train', load_ref=False,n_views=3,max_len=-1):
        self.args = args
        self.root_dir = args.datadir
        self.split = split
        downsample = args.imgScale_train if split=='train' else args.imgScale_test
        assert self.split in ['train', 'val', 'test'], \
            'split must be either "train", "val" or "test"!'
        self.img_wh = (int(640*downsample),int(512*downsample))
        print(f'==> image down scale: {downsample}')
        self.downsample = downsample
        self.scale_factor = 1.0 / 200
        self.max_len = max_len
        assert int(640*downsample)%32 == 0, \
            f'image width is {int(640*downsample)}, it should be divisible by 32, you may need to modify the imgScale'
        self.define_transforms()
        # self.pair_idx = torch.load('configs/pairs.th')  
        # dtu_train = [item for item in range(49) if item not in self.pair_idx['dtu_test'] and item not in [25,0,48]]
        # dtu_train = [25,0,48]+dtu_train
        # self.pair_idx = [dtu_train,self.pair_idx['dtu_test']]

        self.build_metas()
        self.n_views = n_views
        self.build_proj_mats()
        
        print(f'==> image down scale: {self.downsample}')

    def build_metas(self):
        self.metas = []
        # self.scans = ['scan114']
        with open(f'configs/lists/dtu_{self.split}_all.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()] ##hanxue

        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
        light_idxs = [3]
        # light_idxs = [3] if 'train' != self.split else range(7) ##hanxue

        ##hanxue
        self.id_list = []
        for scan in self.scans:
            with open(f'configs/dtu_pairs.txt') as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for ref_view in range(num_viewpoint):
                    src_views = [25,0,48] 
                    for light_idx in light_idxs:
                        self.metas += [(scan, light_idx, [ref_view], src_views)]
                        self.id_list.append([ref_view] + src_views)
        print('len(self.metas),self.metas[0]',len(self.metas),self.metas[0])
        print('len(self.metas),self.metas[5]',len(self.metas),self.metas[5])

        # self.id_list = []
        # for scan in self.scans:
        #     with open(f'configs/dtu_pairs.txt') as f:
        #         num_viewpoint = int(f.readline())
        #         # viewpoints (49)
        #         for _ in range(num_viewpoint):
        #             ref_view = int(f.readline().rstrip())
        #             src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
        #             for light_idx in light_idxs:
        #                 self.metas += [(scan, light_idx, ref_view, src_views)]
        #                 self.id_list.append([ref_view] + src_views)
        #             # metas[0]
        #             # ('scan3', 0, 0, [10, 1, 9, 12, 11, 13, 2, 8, 14, 27])
        #             # metas[1]
        #             # ('scan3', 1, 0, [10, 1, 9, 12, 11, 13, 2, 8, 14, 27])
        #             # id_list[0]
        #             # [0, 10, 1, 9, 12, 11, 13, 2, 8, 14, 27]
        #             # id_list[1]
        #             # [0, 10, 1, 9, 12, 11, 13, 2, 8, 14, 27]
        self.id_list = np.unique(self.id_list) # range(49)
        self.build_remap()

    def build_remap(self):
        self.remap = np.zeros(np.max(self.id_list) + 1).astype('int') # 0*range(49)
        for i, item in enumerate(self.id_list):
            self.remap[item] = i
        print('self.remap',self.remap)

    def define_transforms(self):
        self.transform = T.ToTensor()
        self.src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) * self.scale_factor
        depth_max = depth_min + float(lines[11].split()[1]) * 192 * self.scale_factor
        self.depth_interval = float(lines[11].split()[1])

        # scaling
        extrinsics[:3, 3] *= self.scale_factor
        intrinsics[0:2] *= self.downsample

        return intrinsics, extrinsics, [depth_min, depth_max]

    def build_proj_mats(self):
        proj_mats, intrinsics, world2cams, cam2worlds = [], [], [], []
        for vid in self.id_list:
            proj_mat_filename = os.path.join(self.root_dir,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(proj_mat_filename)
            proj_mat_l = np.eye(4)
            proj_mat_l[:3, :4] = intrinsic @ extrinsic[:3, :4]
            intrinsic[:2] *= 4
            intrinsics += [intrinsic.copy()]
            
            proj_mats += [(proj_mat_l, near_far)]
            world2cams += [extrinsic]
            cam2worlds += [np.linalg.inv(extrinsic)]
        # print(proj_mats)
        # print(intrinsics)
        self.proj_mats, self.intrinsics = np.stack(proj_mats), np.stack(intrinsics)
        self.world2cams, self.cam2worlds = np.stack(world2cams), np.stack(cam2worlds)

    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        depth_h = cv2.resize(depth_h, None, fx=self.downsample, fy=self.downsample,
                             interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!

        return depth_h

    def read_source_views(self, scan, light_idx=3, pair_idx=None, device=torch.device("cpu")):

        src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        light_idx = int(light_idx.detach().cpu().numpy())
        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        for i,idx in enumerate(pair_idx):
            idx = int(idx.detach().cpu().numpy())
            proj_mat_filename = os.path.join(self.root_dir, f'Cameras/train/{idx:08d}_cam.txt')
            intrinsic, w2c, near_far_source = self.read_cam_file(proj_mat_filename)
            c2w = np.linalg.inv(w2c)
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
            intrinsic[:2] = intrinsic[:2]*4  # 4 times downscale in the feature space
            intrinsics.append(intrinsic.copy())

            image_path = os.path.join(self.root_dir,
                                        f'Rectified/{scan}_train/rect_{idx + 1:03d}_{light_idx}_r5000.png')

            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)
            imgs.append(src_transform(img))

        pose_source = {}
        pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
        pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
        pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).float().to(device)

        imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)
        return imgs, proj_mats, near_far_source, pose_source

    def load_poses_all(self):
        c2ws = []
        List = sorted(os.listdir(os.path.join(self.root_dir, f'Cameras/train/')))
        for item in List:
            proj_mat_filename = os.path.join(self.root_dir, f'Cameras/train/{item}')
            intrinsic, w2c, near_far = self.read_cam_file(proj_mat_filename)
            intrinsic[:2] *= 4
            c2ws.append(np.linalg.inv(w2c))
        self.focal = [intrinsic[0, 0], intrinsic[1, 1]]
        return np.stack(c2ws)

    def focal2fov(self,focal, pixels):
        return 2*math.atan(pixels/(2*focal))
    
    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len
    
    def __getitem__(self, idx):
        w, h = self.img_wh
        sample = {}
        scan, light_idx, target_view, src_views = self.metas[idx]

        target_rays = []
        target_rgbs = []
        target_depth = []
        target_R = []
        target_T = []
        target_FovX = []
        target_FovY = []
        target_mask = []
        
        for i, vid in enumerate(target_view):
            image_path = os.path.join(self.root_dir,
                                            f'Rectified/{scan}_train/rect_{vid + 1:03d}_{light_idx}_r5000.png')
            depth_filename = os.path.join(self.root_dir,
                                            f'Depths/{scan}/depth_map_{vid:04d}.pfm')
            img = Image.open(image_path)
            resolution = w,h   
            img = PILtoTorch(img,resolution) #[3,H,W]
            target_rgbs += [img]
            if os.path.exists(depth_filename):# and self.split!='train':
                depth = self.read_depth(depth_filename)
                depth *= self.scale_factor
                # print('depth_near_far',np.min(depth),np.max(depth))
                target_depth += [torch.from_numpy(depth).float().view(-1,1)]
                target_mask += [torch.tensor(depth>0)]

            index_mat = self.remap[vid]
            proj_mat_ls, near_far = self.proj_mats[index_mat]
            intrinsic = self.intrinsics[index_mat]
            c2w = self.cam2worlds[index_mat]
            w2c = self.world2cams[index_mat]
 
            center = [intrinsic[0,2], intrinsic[1,2]]
            focal = [intrinsic[0,0], intrinsic[1,1]]
            directions = get_ray_directions(h, w, focal, center)
            c2w = torch.FloatTensor(c2w)
            rays_o, rays_d = get_rays(directions, c2w)
            R = np.transpose(w2c[:3,:3])
            T = w2c[:3, 3]
            focal_length_x = intrinsic[0,0]
            focal_length_y = intrinsic[1,1]
            FovY = self.focal2fov(focal_length_y, h)
            FovX = self.focal2fov(focal_length_x, w)
            target_R+=[torch.tensor(R).float()]
            target_T+=[torch.tensor(T).float()]
            target_FovX+=[torch.tensor(FovX).float()]
            target_FovY+=[torch.tensor(FovY).float()]
            target_rays += [torch.cat([rays_o, rays_d,
                                        near_far[0] * torch.ones_like(rays_o[:, :1]),
                                        near_far[1] * torch.ones_like(rays_o[:, :1])],
                                    1)] 
        
        if 'train' == self.split:
            target_rays = torch.stack(target_rays, 0)  # (len(meta['frames]),h*w, 8)
            target_rgbs = torch.stack(target_rgbs, 0)  # (len(meta['frames]),3,h,w)
            target_R = torch.stack(target_R,0)
            target_T = torch.stack(target_T,0)
            target_FovY = torch.stack(target_FovY,0)
            target_FovX = torch.stack(target_FovX,0)
            target_mask = torch.stack(target_mask,0)
            # target_rays = torch.cat(target_rays, 0)  # (len(meta['frames])*h*w, 3)
            # target_rgbs = torch.cat(target_rgbs, 0)  # (len(meta['frames])*h*w, 3)
        else:
            target_rays = torch.stack(target_rays, 0)  # (len(meta['frames]),h*w, 3)
            target_rgbs = torch.stack(target_rgbs, 0)#.reshape(-1,*img_wh[::-1], 3)  # (len(meta['frames]),3,h,w)
            target_depth = torch.stack(target_depth, 0).reshape(-1,*self.img_wh[::-1])  # (len(meta['frames]),h,w,3)
            target_R = torch.stack(target_R,0)
            target_T = torch.stack(target_T,0)
            target_FovY = torch.stack(target_FovY,0)
            target_FovX = torch.stack(target_FovX,0)
            target_mask = torch.stack(target_mask,0)

        sample = {'rays': target_rays[-1],
                  'rgbs': target_rgbs[-1],
                  'R':target_R[-1],
                  'T':target_T[-1],
                  'mask':target_mask[-1],
                  'FovX':target_FovX[-1],
                  'FovY':target_FovY[-1],
                  'src_views':src_views,
                  'light_idx':light_idx,
                  'scan':scan}
        sample['idx'] = idx
        return sample
    # def read_meta(self):

    #     # sub select training views from pairing file
    #     if os.path.exists('configs/pairs.th'):
    #         self.img_idx = self.pair_idx[0] if 'train'==self.split else self.pair_idx[1]
    #         print(f'===> {self.split}ing index: {self.img_idx}')

    #     # name = os.path.basename(self.root_dir)
    #     # test_idx = torch.load('configs/pairs.th')[f'{name}_test']
    #     # self.img_idx = test_idx if self.split!='train' else np.delete(np.arange(0,49),test_idx)

    #     w, h = self.img_wh

    #     self.image_paths = []
    #     self.poses = []
    #     self.all_rays = []
    #     self.all_rgbs = []
    #     self.all_depth = []
    #     self.all_R = []
    #     self.all_T = []
    #     self.all_FovX = []
    #     self.all_FovY = []
    #     self.all_mask = []

    #     for idx in self.img_idx:
    #         proj_mat_filename = os.path.join(self.root_dir, f'Cameras/train/{idx:08d}_cam.txt')
    #         intrinsic, w2c, near_far = self.read_cam_file(proj_mat_filename)
    #         c2w = np.linalg.inv(w2c)
    #         self.poses += [c2w]
    #         c2w = torch.FloatTensor(c2w)

    #         image_path = os.path.join(self.root_dir,
    #                                     f'Rectified/{self.scan}_train/rect_{idx + 1:03d}_3_r5000.png')
    #         depth_filename = os.path.join(self.root_dir,
    #                                       f'Depths/{self.scan}/depth_map_{idx:04d}.pfm')
    #         self.image_paths += [image_path]
    #         img = Image.open(image_path)
    #         resolution = w,h
    #         # import imageio
    #         # imageio.imwrite('gs-gt1.png',(np.array(img)).astype('uint8'))    
    #         img = PILtoTorch(img,resolution)
    #         # import imageio
    #         # vis=img.reshape(512, 640,3).numpy()
    #         # imageio.imwrite('gs-gt2.png',(vis*255).astype('uint8'))

    #         # img = img[:3, ...]
            
    #         # img = img.resize(self.img_wh, Image.LANCZOS)
    #         # img = self.transform(img)  # (3, h, w)
    #         # img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGBA
    #         self.all_rgbs += [img]

    #         if os.path.exists(depth_filename):# and self.split!='train':
    #             depth = self.read_depth(depth_filename)
    #             depth *= self.scale_factor
    #             # print('depth_near_far',np.min(depth),np.max(depth))
    #             self.all_depth += [torch.from_numpy(depth).float().view(-1,1)]
    #             self.all_mask+=[torch.tensor(depth>0)]

    #         # ray directions for all pixels, same for all images (same H, W, focal)
    #         intrinsic[:2] *= 4 # * the provided intrinsics is 4x downsampled, now keep the same scale with image
    #         center = [intrinsic[0,2], intrinsic[1,2]]
    #         self.focal = [intrinsic[0,0], intrinsic[1,1]]
    #         self.directions = get_ray_directions(h, w, self.focal, center)  # (h, w, 3)
    #         rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
    #         R = np.transpose(w2c[:3,:3])
    #         T = w2c[:3, 3]
    #         focal_length_x = intrinsic[0,0]
    #         focal_length_y = intrinsic[1,1]
    #         FovY = self.focal2fov(focal_length_y, h)
    #         FovX = self.focal2fov(focal_length_x, w)
    #         self.all_R+=[torch.tensor(R).float()]
    #         self.all_T+=[torch.tensor(T).float()]
    #         self.all_FovX+=[torch.tensor(FovX).float()]
    #         self.all_FovY+=[torch.tensor(FovY).float()]
    #         self.all_rays += [torch.cat([rays_o, rays_d,
    #                                      near_far[0] * torch.ones_like(rays_o[:, :1]),
    #                                      near_far[1] * torch.ones_like(rays_o[:, :1])],
    #                                     1)]  # (h*w, 8)
    #         # print('near_far',near_far)
    #     pointclouds = []
    #     for idx in range(49):
    #         proj_mat_filename = os.path.join(self.root_dir, f'Cameras/train/{idx:08d}_cam.txt')
    #         intrinsic, w2c, near_far = self.read_cam_file(proj_mat_filename)
    #         c2w = np.linalg.inv(w2c)
    #         c2w = torch.FloatTensor(c2w)
    #         intrinsic[:2] *= 4 # * the provided intrinsics is 4x downsampled, now keep the same scale with image
    #         center = [intrinsic[0,2], intrinsic[1,2]]
    #         focal = [intrinsic[0,0], intrinsic[1,1]]
    #         directions = get_ray_directions(h, w, focal, center)  # (h, w, 3)
    #         rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
    #         depth_filename = os.path.join(self.root_dir,
    #                                       f'Depths/{self.scan}/depth_map_{idx:04d}.pfm')
    #         image_path = os.path.join(self.root_dir,
    #                                     f'Rectified/{self.scan}_train/rect_{idx + 1:03d}_3_r5000.png')
    #         img = Image.open(image_path)
    #         rgb_numpy = (np.array(img) / 255.0).reshape(-1,3)
    #         if os.path.exists(depth_filename):
    #             depth = self.read_depth(depth_filename)
    #             depth *= self.scale_factor
    #             depth = depth.reshape(-1)
    #             mask = depth>0
    #             point_samples = rays_o.numpy()[mask] + np.expand_dims(depth[mask],-1) * rays_d.numpy()[mask]
    #             pointclouds.append(np.concatenate([point_samples,rgb_numpy[mask]],axis=-1))
    #     if len(pointclouds)!=0:
    #         pointclouds = np.vstack(pointclouds)
    #         self.init_pointclouds = torch.tensor(pointclouds[::50]).float()
    #         print('self.init_pointclouds shape',self.init_pointclouds.shape)
    #     self.poses = np.stack(self.poses)
    #     if 'train' == self.split:
    #         self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 8)
    #         self.all_rgbs = torch.stack(self.all_rgbs, 0)  # (len(self.meta['frames]),3,h,w)
    #         self.all_R = torch.stack(self.all_R,0)
    #         self.all_T = torch.stack(self.all_T,0)
    #         self.all_FovY = torch.stack(self.all_FovY,0)
    #         self.all_FovX = torch.stack(self.all_FovX,0)
    #         self.all_mask = torch.stack(self.all_mask,0)
    #         # self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
    #         # self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
    #     else:
    #         self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
    #         self.all_rgbs = torch.stack(self.all_rgbs, 0)#.reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),3,h,w)
    #         self.all_depth = torch.stack(self.all_depth, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)
    #         self.all_R = torch.stack(self.all_R,0)
    #         self.all_T = torch.stack(self.all_T,0)
    #         self.all_FovY = torch.stack(self.all_FovY,0)
    #         self.all_FovX = torch.stack(self.all_FovX,0)
    #         self.all_mask = torch.stack(self.all_mask,0)


    # def __len__(self):
    #     if self.split == 'train':
    #         return len(self.all_rays)
    #     return len(self.all_rgbs)

    # def __getitem__(self, idx):
    #     idx = idx%self.__len__()
    #     if self.split == 'train':  # use data in the buffers
    #         # view, ray_idx = torch.randint(0,len(self.all_rays),(1,)), torch.randperm(self.all_rays.shape[1])[:self.args.batch_size]
    #         # sample = {'rays': self.all_rays[view,ray_idx],
    #         #           'rgbs': self.all_rgbs[view,ray_idx]}


    #         sample = {'rays': self.all_rays[idx],
    #                   'rgbs': self.all_rgbs[idx],
    #                   'R':self.all_R[idx],
    #                   'T':self.all_T[idx],
    #                   'mask':self.all_mask[idx],
    #                   'FovX':self.all_FovX[idx],
    #                   'FovY':self.all_FovY[idx]}

    #     else:  # create data for each image separately

    #         img = self.all_rgbs[idx]
    #         rays = self.all_rays[idx]
    #         depth = self.all_depth[idx] # for quantity evaluation

    #         sample = {'rays': rays,
    #                   'rgbs': img,
    #                   'depth': depth,
    #                   'R':self.all_R[idx],
    #                   'T':self.all_T[idx],
    #                   'mask':self.all_mask[idx],
    #                   'FovX':self.all_FovX[idx],
    #                   'FovY':self.all_FovY[idx]}
    #     sample['idx'] = idx
    #     return sample

