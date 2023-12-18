#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import einsum
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation,sphere_points
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize



    def __init__(self, sh_degree : int, grow_dir = False, num_dirs = 128, continous_dir=False,grow_distance=False,modelcg=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.grow_dir = grow_dir
        self.continous_dir = continous_dir
        self.grow_distance = grow_distance
        self.num_dirs = num_dirs
        self.modelcg = modelcg
        self.learn_split_distance = self.modelcg.learn_split_distance
        self.learn_split_scale = self.modelcg.learn_split_scale
        if self.grow_dir:
            self._dirs_prob = torch.empty(0)
            dirs = sphere_points(self.num_dirs)
            self.dirs = torch.tensor(dirs,device="cuda").to(torch.float32)
        elif self.continous_dir:
            self._conti_dirs = torch.empty(0)
        
        if self.grow_distance:
            self._grow_dist = torch.empty(0)
        if self.learn_split_distance:
            self._split_distance = torch.empty(0)
        if self.learn_split_scale:
            self._split_scale = torch.empty(0)
        self.setup_functions()


    def capture(self):
        if self.grow_dir:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self._dirs_prob,
                self.xyz_gradient_accum,
                self.denom,
                # self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        elif self.continous_dir:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self._conti_dirs,
                self.xyz_gradient_accum,
                self.denom,
                # self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        else:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_grow_dist(self):
        return 2*torch.sigmoid(self._grow_dist)

    @property
    def get_split_distance(self):
        return 2.2*torch.sigmoid(self._split_distance)
    
    @property
    def get_split_scale(self):
        return 0.6*torch.sigmoid(self._split_scale)+0.5
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_dirs_prob(self):
        return self._dirs_prob
    
    @property
    def get_conti_dirs(self):
        return self._conti_dirs
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        if self.grow_dir:
            probs = torch.ones((fused_point_cloud.shape[0],self.num_dirs),device="cuda")/self.num_dirs
            self._dirs_prob = nn.Parameter(probs.requires_grad_(True))
        if self.continous_dir:
            probs = torch.nn.functional.normalize(torch.randn((fused_point_cloud.shape[0],3),device="cuda"), p=2.0, dim=-1)
            self._conti_dirs = nn.Parameter(probs.requires_grad_(True))
        if self.grow_distance:
            dists = torch.zeros((fused_point_cloud.shape[0],1),device="cuda")
            self._grow_dist = nn.Parameter(dists.requires_grad_(True))
        if self.learn_split_distance:
            split_distances = torch.zeros((fused_point_cloud.shape[0],3),device='cuda')
            self._split_distance = nn.Parameter(split_distances.requires_grad_(True))
        if self.learn_split_scale:
            split_scale = torch.zeros((fused_point_cloud.shape[0],1),device='cuda')
            self._split_scale = nn.Parameter(split_scale.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        if self.grow_dir:
            l.append({'params': [self._dirs_prob], 'lr': training_args.growdirs_lr, "name": "dirs_prob"})
        if self.continous_dir:
            l.append({'params': [self._conti_dirs], 'lr': training_args.growdirs_lr, "name": "conti_dirs"})
        if self.grow_distance:
            l.append({'params': [self._grow_dist], 'lr': training_args.growdistance_lr, "name": "grow_dist"})
        if self.learn_split_distance:
            l.append({'params': [self._split_distance], 'lr': training_args.splitdistance_lr, "name": "split_distance"})
        if self.learn_split_scale:
            l.append({'params': [self._split_scale], 'lr': training_args.splitscale_lr, "name": "split_scale"})
        print('len(l)',len(l))
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def argmax_softmax(self, logits, tau=1, dim=-1):
        logits = logits / tau
        y_soft = logits.softmax(dim)
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret
    
    def fake_grow(self):
        soft_one_hot = self.argmax_softmax(self._dirs_prob, tau=1.0, dim=-1)
        grow_dirs = einsum('b n, n d -> b d', soft_one_hot, self.dirs)
        # shift_distance = torch.mean(self.get_scaling,-1,keepdim=True)
        shift_distance = torch.max(self.get_scaling, dim=1,keepdim=True).values
        new_xyz = self._xyz.data+grow_dirs*shift_distance
        print('whether new xyz optimizable',new_xyz.requires_grad,new_xyz.detach().requires_grad)
        self._xyz = torch.cat((self._xyz,new_xyz.detach()),0)
        self._features_dc = torch.cat((self._features_dc,self._features_dc.detach()),0)
        self._features_rest = torch.cat((self._features_rest,self._features_rest.detach()),0)
        self._opacity = torch.cat((self._opacity,self._opacity.detach()),0)
        self._scaling = torch.cat((self._scaling,self._scaling.detach()),0)
        self._rotation = torch.cat((self._rotation,self._rotation.detach()),0)
        self._dirs_prob = torch.cat((self._dirs_prob.softmax(-1),self._dirs_prob.softmax(-1).detach()),0)
    


        
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.grow_dir:
            self._dirs_prob = optimizable_tensors["dirs_prob"]
        if self.continous_dir:
            self._conti_dirs = optimizable_tensors["conti_dirs"]
        if self.grow_distance:
            self._grow_dist = optimizable_tensors["grow_dist"]
        if self.learn_split_distance:
            self._split_distance = optimizable_tensors["split_distance"]
        if self.learn_split_scale:
            self._split_scale = optimizable_tensors["split_scale"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            # print(group['name'],stored_state)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_probs=None,new_continous_dir=None,new_grow_dist = None,new_split_distance=None,new_split_scale=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if self.grow_dir:
            # probs = torch.ones((new_xyz.shape[0],self.num_dirs),device="cuda")/self.num_dirs
            d['dirs_prob'] = new_probs
        if self.continous_dir:
            d['conti_dirs'] = new_continous_dir
        if self.grow_distance:
            d['grow_dist'] = new_grow_dist
        if self.learn_split_distance:
            d['split_distance'] = new_split_distance
        if self.learn_split_scale:
            d["split_scale"] = new_split_scale
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.grow_dir:
            self._dirs_prob = optimizable_tensors["dirs_prob"]
        if self.continous_dir:
            self._conti_dirs = optimizable_tensors["conti_dirs"]
        if self.grow_distance:
            self._grow_dist =  optimizable_tensors["grow_dist"]
        if self.learn_split_distance:
            self._split_distance = optimizable_tensors["split_distance"]
        if self.learn_split_scale:
            self._split_scale = optimizable_tensors["split_scale"]


        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        new_probs=None
        new_continous_dir = None
        new_grow_dist = None
        new_split_distance = None
        new_split_scale = None
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if self.learn_split_distance:
            stds = self.get_scaling[selected_pts_mask].repeat(int(N/2),1)
            splitdistance = self.get_split_distance[selected_pts_mask].repeat(int(N/2),1)
            if stds.size(0)>0:
                print('splitdistance',splitdistance.shape,torch.max(splitdistance),torch.min(splitdistance))
            samples = stds*splitdistance
            samples = torch.cat((samples,-samples),dim=0)
        else:
            if self.modelcg.symmetric_split and N%2==0:
                print("will use symmetric_split",self.get_scaling[selected_pts_mask].shape,self.get_scaling[selected_pts_mask].repeat(int(N/2),1).shape)
                stds = self.get_scaling[selected_pts_mask].repeat(int(N/2),1)
                means =torch.zeros((stds.size(0), 3),device="cuda")
                samples = torch.normal(mean=means, std=stds)
                samples = torch.cat((samples,-samples),dim=0)
            else:
                stds = self.get_scaling[selected_pts_mask].repeat(N,1)
                means =torch.zeros((stds.size(0), 3),device="cuda")
                samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        if self.learn_split_scale:
            splitscale = self.get_split_scale[selected_pts_mask].repeat(N,3)
            if stds.size(0)>0:
                print('splitscale',splitscale.shape,torch.max(splitscale),torch.min(splitscale))
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (splitscale*N))
        else:
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        if self.grow_dir:
            new_probs = self._dirs_prob[selected_pts_mask].repeat(N,1)
        if self.continous_dir:
            new_continous_dir = self._conti_dirs[selected_pts_mask].repeat(N,1)
        if self.grow_distance:
            new_grow_dist = self._grow_dist[selected_pts_mask].repeat(N,1)
        if self.learn_split_distance:
            if not self.modelcg.split_notreinit:
                splitdistance = torch.zeros((torch.sum(selected_pts_mask),3),device="cuda")
                # print('self._split_distance[selected_pts_mask]',self._split_distance[selected_pts_mask])
                self._split_distance[selected_pts_mask]= splitdistance
                new_split_distance = self._split_distance[selected_pts_mask].repeat(N,1)
            else:
                new_split_distance = self._split_distance[selected_pts_mask].repeat(N,1)
        if self.learn_split_scale:
            # new_split_scale = self._split_scale[selected_pts_mask].repeat(N,1)
            if not self.modelcg.split_notreinit:
                splitscale = torch.zeros((torch.sum(selected_pts_mask),1),device="cuda")
                # print('self._split_scale[selected_pts_mask]',self._split_scale[selected_pts_mask])
                self._split_scale[selected_pts_mask]= splitscale
                new_split_scale = self._split_scale[selected_pts_mask].repeat(N,1)
            else:
                new_split_scale = self._split_scale[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,new_probs,new_continous_dir,new_grow_dist,new_split_distance,new_split_scale)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        new_probs=None
        new_continous_dir=None
        new_grow_dist = None
        new_split_distance = None
        new_split_scale = None
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if self.grow_dir:
            new_probs = self._dirs_prob[selected_pts_mask] 
        if self.continous_dir:
            new_continous_dir = self._conti_dirs[selected_pts_mask]
        if self.grow_distance:
            new_grow_dist = self._grow_dist[selected_pts_mask]
        if self.learn_split_distance:
            new_split_distance = self._split_distance[selected_pts_mask]
        if self.learn_split_scale:
            new_split_scale = self._split_scale[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,new_probs,new_continous_dir,new_grow_dist,new_split_distance,new_split_scale)
    
    def densify_and_grow(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if self.grow_dir:
            soft_one_hot = self.argmax_softmax(self._dirs_prob[selected_pts_mask], tau=1.0, dim=-1)
            print('soft_one_hot',soft_one_hot)
            # print(soft_one_hot.dtype,pc.dirs.dtype)
            grow_dirs = einsum('b n, n d -> b d', soft_one_hot, self.dirs)
        elif self.continous_dir:
            grow_dirs = torch.nn.functional.normalize(self._conti_dirs[selected_pts_mask], p=2.0, dim=-1)
            print('conti dirs',grow_dirs)

        if self.grow_distance:
            _grow_dist = self.get_grow_dist[selected_pts_mask]
            print('_grow_dist',_grow_dist)
        else:
            _grow_dist = 1
        

        shift_distance = torch.max(self.get_scaling[selected_pts_mask], dim=1,keepdim=True).values
        # shift_distance = torch.mean(self.get_scaling[selected_pts_mask], dim=1,keepdim=True)
        new_xyz = self._xyz[selected_pts_mask]+(grow_dirs*shift_distance*_grow_dist)
        print('num of points will grow',torch.sum(selected_pts_mask))
        # new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        # new_scaling = torch.mean(self._scaling[selected_pts_mask],dim=1,keepdim=True).repeat(1,3)
        new_rotation = self._rotation[selected_pts_mask]
        # prob_init = True
        if not self.modelcg.prob_notreinit:
            if self.grow_dir:
                probs = torch.ones((torch.sum(selected_pts_mask),self.num_dirs),device="cuda")/self.num_dirs
                self._dirs_prob[selected_pts_mask] = probs
            elif self.continous_dir:
                probs = torch.randn((torch.sum(selected_pts_mask),3),device="cuda")
                self._conti_dirs[selected_pts_mask] = torch.nn.functional.normalize(probs, p=2.0, dim=-1)
            if self.grow_distance:
                dists = torch.zeros((torch.sum(selected_pts_mask),1),device="cuda")
                self._grow_dist[selected_pts_mask]= dists
            
        new_split_distance = None
        new_grow_dist = None
        new_split_scale = None
        if self.learn_split_distance:
            new_split_distance = self._split_distance[selected_pts_mask]
        if self.learn_split_scale:
            new_split_scale = self._split_scale[selected_pts_mask]
        if self.grow_distance:
            new_grow_dist = self._grow_dist[selected_pts_mask]
        if self.grow_dir:
            new_probs = self._dirs_prob[selected_pts_mask] 
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,new_probs,None,new_grow_dist,new_split_distance,new_split_scale)
        elif self.continous_dir:
            new_continous_dir = self._conti_dirs[selected_pts_mask] 
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,None,new_continous_dir,new_grow_dist,new_split_distance,new_split_scale)
        

        # for group in self.optimizer.param_groups:
        #     if group["name"]=="dirs_prob":
        #         group["params"][0][selected_pts_mask] = probs
        #         group["params"][0] = nn.Parameter((group["params"][0].requires_grad_(True)))
        #         self._dirs_prob = group["params"][0]
        
    def densify_and_growsplit(self, grads, grad_threshold, scene_extent, N=2):
        new_probs=None
        new_continous_dir = None
        new_grow_dist = None
        new_split_distance = None
        new_split_scale = None
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask[grads.shape[0]:] = True
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if self.learn_split_distance:
            stds = self.get_scaling[selected_pts_mask].repeat(int(N/2),1)
            splitdistance = self.get_split_distance[selected_pts_mask].repeat(int(N/2),1)
            if stds.size(0)>0:
                print('splitdistance',splitdistance.shape,torch.max(splitdistance),torch.min(splitdistance))
            samples = stds*splitdistance
            samples = torch.cat((samples,-samples),dim=0)
        else:
            if self.modelcg.symmetric_split and N%2==0:
                print("will use symmetric_split",self.get_scaling[selected_pts_mask].shape,self.get_scaling[selected_pts_mask].repeat(int(N/2),1).shape)
                stds = self.get_scaling[selected_pts_mask].repeat(int(N/2),1)
                means = torch.zeros((stds.size(0), 3),device="cuda")
                samples = torch.normal(mean=means, std=stds)
                samples = torch.cat((samples,-samples),dim=0)
            else:
                stds = self.get_scaling[selected_pts_mask].repeat(N,1)
                means =torch.zeros((stds.size(0), 3),device="cuda")
                samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        if self.learn_split_scale:
            splitscale = self.get_split_scale[selected_pts_mask].repeat(N,3)
            if stds.size(0)>0:
                print('splitscale',splitscale.shape,torch.max(splitscale),torch.min(splitscale))
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (splitscale*N))
        else:
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        if self.grow_dir:
            new_probs = self._dirs_prob[selected_pts_mask].repeat(N,1)
        if self.continous_dir:
            new_continous_dir = self._conti_dirs[selected_pts_mask].repeat(N,1)
        if self.grow_distance:
            new_grow_dist = self._grow_dist[selected_pts_mask].repeat(N,1)
        if self.learn_split_distance:
            if not self.modelcg.split_notreinit:
                splitdistance = torch.zeros((torch.sum(selected_pts_mask),3),device="cuda")
                # print('self._split_distance[selected_pts_mask]',self._split_distance[selected_pts_mask])
                self._split_distance[selected_pts_mask]= splitdistance
                new_split_distance = self._split_distance[selected_pts_mask].repeat(N,1)
            else:
                new_split_distance = self._split_distance[selected_pts_mask].repeat(N,1)
        if self.learn_split_scale:
            if not self.modelcg.split_notreinit:
                splitscale = torch.zeros((torch.sum(selected_pts_mask),1),device="cuda")
                # print('self._split_scale[selected_pts_mask]',self._split_scale[selected_pts_mask])
                self._split_scale[selected_pts_mask]= splitscale
                new_split_scale = self._split_scale[selected_pts_mask].repeat(N,1)
            else:
                new_split_scale = self._split_scale[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,new_probs,new_continous_dir,new_grow_dist,new_split_distance,new_split_scale)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size,opt=None,iteration=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if (self.grow_dir or self.continous_dir) and (iteration > opt.opacity_reset_interval):# and (iteration % opt.opacity_reset_interval > (2*opt.densify_from_iter+opt.densification_interval-1))):
            self.densify_and_grow(grads,max_grad,extent)
            self.densify_and_growsplit(grads, max_grad, extent)
        else:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            # print('iteration,big_points_vs, big_points_ws,prune_mask',iteration,torch.sum(big_points_vs),torch.sum(big_points_ws),torch.sum(prune_mask))
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # if self.grow_dir and (iteration > opt.opacity_reset_interval) and (iteration % opt.opacity_reset_interval == (2*opt.densify_from_iter+opt.densification_interval)):
        #     print('at this grow iteration, we will not prune points', iteration)
        # else:
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1