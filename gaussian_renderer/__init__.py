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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, grow_dir =False, densify_grad_threshold = 0, iteration=None, opt=None,continous_dir=False,grow_distance=False,modelcg=None,cameras_extent=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    learn_split_distance = False
    learn_split_scale = False
    if modelcg is not None:
        learn_split_distance = modelcg.learn_split_distance
        learn_split_scale = modelcg.learn_split_scale
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    origin_num = means3D.shape[0]
    selected_pts_mask = None
    if iteration is not None and opt is not None:
        if (grow_dir or continous_dir) and (iteration > (opt.densify_from_iter-opt.densification_interval-1)) and iteration<opt.densify_until_iter:
            if (iteration > opt.opacity_reset_interval): #and (iteration % opt.opacity_reset_interval > 2*opt.densify_from_iter):
                grads = pc.xyz_gradient_accum / pc.denom
                grads[grads.isnan()] = 0.0
                selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= densify_grad_threshold, True, False)
                # selected_pts_mask = torch.logical_and(selected_pts_mask,
                #                                     pc.get_opacity[:,0]>= 0.2)
                if grow_dir:
                    soft_one_hot = pc.argmax_softmax(pc.get_dirs_prob[selected_pts_mask], tau=1.0, dim=-1)
                    grow_dirs = einsum('b n, n d -> b d', soft_one_hot, pc.dirs)
                elif continous_dir:
                    grow_dirs = torch.nn.functional.normalize(pc.get_conti_dirs[selected_pts_mask], p=2.0, dim=-1)
                if grow_distance:
                    _grow_dist = pc.get_grow_dist[selected_pts_mask]
                    # print('_grow_dist',_grow_dist)
                else:
                    _grow_dist = 1
                # shift_distance = torch.mean(pc.get_scaling[selected_pts_mask], dim=1,keepdim=True)
                shift_distance = torch.max(pc.get_scaling[selected_pts_mask], dim=1,keepdim=True).values
                new_xyz = means3D[selected_pts_mask]+grow_dirs*shift_distance*_grow_dist
                means3D = torch.cat((means3D,new_xyz),0)
                shs = torch.cat((shs,shs[selected_pts_mask]),0)
                opacity = torch.cat((opacity,opacity[selected_pts_mask]),0)
                scales = torch.cat((scales,scales[selected_pts_mask]),0)
                # new_scales = torch.mean(scales[selected_pts_mask],dim=1,keepdim=True).repeat(1,3)
                # scales = torch.cat((scales,new_scales),0)
                rotations = torch.cat((rotations,rotations[selected_pts_mask]),0)
                means2D = torch.cat((means2D,means2D[selected_pts_mask]),0)

                n_points = means3D.shape[0]
                padded_grad = torch.zeros((n_points), device="cuda")
                # print('padded_grad',padded_grad.shape,grads.shape)
                padded_grad[:grads.shape[0]] = grads.squeeze()
                selected_split_pts_mask = torch.where(padded_grad >= densify_grad_threshold, True, False)
                selected_split_pts_mask[grads.shape[0]:] = True
                selected_split_pts_mask = torch.logical_and(selected_split_pts_mask,
                                                    torch.max(scales, dim=1).values > pc.percent_dense*cameras_extent)
                num_point_to_split = torch.sum(selected_split_pts_mask)
                if (learn_split_distance or learn_split_scale) and num_point_to_split>0:
                    raw_rotation = pc._rotation
                    raw_rotation = torch.cat((raw_rotation,raw_rotation[selected_pts_mask]),0)
                    if learn_split_distance:
                        split_dist = pc.get_split_distance
                        split_dist = torch.cat((split_dist,split_dist[selected_pts_mask]),0)
                        # print('split_dist.requires_grad',split_dist.requires_grad)
                    if learn_split_scale:
                        split_scale = pc.get_split_scale
                        split_scale = torch.cat((split_scale,split_scale[selected_pts_mask]),0)
                        # print('split_scale.requies_grad',split_scale.requires_grad)
                    assert cameras_extent is not None
                    N=2
                    if learn_split_distance:
                        stds = scales[selected_split_pts_mask].repeat(int(N/2),1)
                        splitdistance = split_dist[selected_split_pts_mask].repeat(int(N/2),1)
                        # if stds.size(0)>0:
                            # print('splitdistance',splitdistance.shape,torch.unique(splitdistance),torch.max(splitdistance),torch.min(splitdistance))
                        samples = stds*splitdistance
                        samples = torch.cat((samples,-samples),dim=0)
                    else:
                        stds = scales[selected_split_pts_mask].repeat(int(N/2),1)
                        means = torch.zeros((stds.size(0), 3),device="cuda")
                        samples = torch.normal(mean=means, std=stds)
                        samples = torch.cat((samples,-samples),dim=0) 
                    rots = build_rotation(raw_rotation[selected_split_pts_mask]).repeat(N,1,1)
                    new_split_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + means3D[selected_split_pts_mask].repeat(N, 1)
                    assert N*num_point_to_split == new_split_xyz.size(0)
                    if learn_split_scale:
                        splitscale = split_scale[selected_split_pts_mask].repeat(N,3)
                        # if stds.size(0)>0:
                            # print('splitscale',splitscale.shape,torch.unique(splitscale),torch.max(splitscale),torch.min(splitscale))
                        new_split_scales = scales[selected_split_pts_mask].repeat(N,1) / (splitscale*N)
                    else:
                        new_split_scales = scales[selected_split_pts_mask].repeat(N,1) / (0.8*N)
                    assert N*num_point_to_split == new_split_scales.size(0)
                    prune_split_points = ~selected_split_pts_mask
                    means3D = torch.cat((means3D[prune_split_points],new_split_xyz),0)
                    shs = torch.cat((shs[prune_split_points],shs[selected_split_pts_mask].repeat(N,1,1)),0)
                    opacity = torch.cat((opacity[prune_split_points],opacity[selected_split_pts_mask].repeat(N,1)),0)
                    scales = torch.cat((scales[prune_split_points],new_split_scales),0)
                    rotations = torch.cat((rotations[prune_split_points],rotations[selected_split_pts_mask].repeat(N,1)),0)
                    means2D = torch.cat((means2D[prune_split_points],means2D[selected_split_pts_mask].repeat(N,1)),0)
                    assert means3D.size(0)-n_points==(N-1)*num_point_to_split
                    assert scales.size(0) == means3D.size(0)
        elif learn_split_distance or learn_split_scale:
            n_points = means3D.shape[0]
            # print('learn_split_scale,learn_split_distance',learn_split_scale,learn_split_distance)
            assert cameras_extent is not None
            grads = pc.xyz_gradient_accum / pc.denom
            grads[grads.isnan()] = 0.0
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= densify_grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(pc.get_scaling, dim=1).values > pc.percent_dense*cameras_extent)
            num_point_to_split = torch.sum(selected_pts_mask)
            if (learn_split_distance or learn_split_scale) and num_point_to_split>0:
                raw_rotation = pc._rotation
                # raw_rotation = torch.cat((raw_rotation,raw_rotation[selected_pts_mask]),0)
                if learn_split_distance:
                    split_dist = pc.get_split_distance
                    # split_dist = torch.cat((split_dist,split_dist[selected_pts_mask]),0)
                    # print('split_dist.requires_grad',split_dist.requires_grad)
                if learn_split_scale:
                    split_scale = pc.get_split_scale
                    # split_scale = torch.cat((split_scale,split_scale[selected_pts_mask]),0)
                    # print('split_scale.requies_grad',split_scale.requires_grad)
                N=2
                if learn_split_distance:
                    stds = scales[selected_pts_mask]
                    splitdistance = split_dist[selected_pts_mask]
                    # if stds.size(0)>0:
                        # print('splitdistance',splitdistance.shape,torch.unique(splitdistance),torch.max(splitdistance),torch.min(splitdistance))
                    samples = stds*splitdistance
                    # samples = torch.cat((samples,-samples),dim=0)
                else:
                    stds = scales[selected_pts_mask]
                    means = torch.zeros((stds.size(0), 3),device="cuda")
                    samples = torch.normal(mean=means, std=stds)
                    # samples = torch.cat((samples,-samples),dim=0) 
                rots = build_rotation(raw_rotation[selected_pts_mask])#.repeat(N,1,1)
                new_split_xyz = torch.bmm(rots, -samples.unsqueeze(-1)).squeeze(-1) + means3D[selected_pts_mask]
                delta_xyz = torch.zeros_like(means3D,device='cuda')
                delta_xyz[selected_pts_mask]  = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)#[:num_point_to_split]

                # means3D[selected_pts_mask] = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)[:num_point_to_split] + means3D[selected_pts_mask]
                assert num_point_to_split == new_split_xyz.size(0)
                if learn_split_scale:
                    # splitscale = split_scale[selected_pts_mask].repeat(N,3)
                    # if stds.size(0)>0:
                        # print('splitscale',splitscale.shape,torch.unique(splitscale),torch.max(splitscale),torch.min(splitscale))
                    new_split_scales = scales[selected_pts_mask] / (split_scale[selected_pts_mask]*N)
                    delta_scales = torch.ones_like(scales,device='cuda')
                    delta_scales[selected_pts_mask] = split_scale[selected_pts_mask]*N
                    # print(delta_scales)
                    # scales[selected_pts_mask] = scales[selected_pts_mask]/(split_scale[selected_pts_mask]*N)
                    scales = torch.cat((scales/delta_scales,new_split_scales),0)
                else:
                    new_split_scales = scales[selected_pts_mask] / (0.8*N)
                    delta_scales = torch.ones_like(scales,device='cuda')
                    delta_scales[selected_pts_mask] = 0.8*N*delta_scales[selected_pts_mask]
                    # print(delta_scales)
                    # scales[selected_pts_mask] = scales[selected_pts_mask]/(0.8*N)
                    scales = torch.cat((scales/delta_scales,new_split_scales),0)
                assert num_point_to_split == new_split_scales.size(0)
                prune_split_points = ~selected_pts_mask
                means3D = torch.cat((means3D+delta_xyz,new_split_xyz),0)
                shs = torch.cat((shs,shs[selected_pts_mask]),0)
                opacity = torch.cat((opacity,opacity[selected_pts_mask]),0)
                
                rotations = torch.cat((rotations,rotations[selected_pts_mask]),0)
                means2D = torch.cat((means2D,means2D[selected_pts_mask]),0)
                assert means3D.size(0)-n_points==(N-1)*num_point_to_split
                assert scales.size(0) == means3D.size(0)


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    if grow_dir or continous_dir:
        radii = radii[:origin_num]
    if  learn_split_distance or learn_split_scale:
        radii = radii[:origin_num]
    # if iteration is not None and opt is not None:
    #     if grow_dir and (iteration > (opt.densify_from_iter-opt.densification_interval-1)) and iteration<opt.densify_until_iter:
    #         if (iteration > opt.opacity_reset_interval) and (iteration % opt.opacity_reset_interval > 2*opt.densify_from_iter):
    #             # print('===================as usual')
    #             grads = pc.xyz_gradient_accum / pc.denom
    #             grads[grads.isnan()] = 0.0
    #             selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= densify_grad_threshold, True, False)
    #             # selected_pts_mask = torch.logical_and(selected_pts_mask,
    #             #                                     pc.get_opacity[:,0]>= 0.2)
                
    #             # print(pc.get_dirs_prob[selected_pts_mask].shape)
    #             soft_one_hot = pc.argmax_softmax(pc.get_dirs_prob[selected_pts_mask], tau=1.0, dim=-1)
    #             # print(soft_one_hot.dtype,pc.dirs.dtype)
    #             grow_dirs = einsum('b n, n d -> b d', soft_one_hot, pc.dirs)
    #             # shift_distance = torch.mean(pc.get_scaling,-1,keepdim=True)
    #             shift_distance = torch.max(pc.get_scaling[selected_pts_mask], dim=1,keepdim=True).values
    #             # print(means3D.shape,grow_dirs.shape,shift_distance.shape)
    #             # new_xyz = means3D.data[selected_pts_mask]+(grow_dirs*shift_distance)[selected_pts_mask]
    #             new_xyz = means3D[selected_pts_mask] +grow_dirs*shift_distance 
    #             # print('whether new xyz optimizable',new_xyz.requires_grad,new_xyz .requires_grad)
    #             means3D = torch.cat((means3D,new_xyz),0)
    #             shs = torch.cat((shs,shs[selected_pts_mask] ),0)
    #             opacity = torch.cat((opacity,opacity[selected_pts_mask] ),0)
    #             scales = torch.cat((scales,scales[selected_pts_mask] ),0)
    #             rotations = torch.cat((rotations,rotations[selected_pts_mask] ),0)
    #             means2D = torch.cat((means2D,means2D[selected_pts_mask] ),0)
    #             rendered_image_, radii_ = rasterizer(
    #                 means3D = means3D,
    #                 means2D = means2D,
    #                 shs = shs,
    #                 colors_precomp = colors_precomp,
    #                 opacities = opacity,
    #                 scales = scales,
    #                 rotations = rotations,
    #                 cov3D_precomp = cov3D_precomp)
    #             rendered_image = torch.cat((rendered_image,rendered_image_),0)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "selected_pts_mask":selected_pts_mask}
