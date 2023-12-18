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
import torchvision
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# def l1_loss(network_output, gt):
#     return torch.abs((network_output - gt)).mean()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, grow_dir = dataset.grow_dir, num_dirs = dataset.num_dirs,continous_dir=dataset.continous_dir,grow_distance=dataset.grow_distance,modelcg=dataset)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # if dataset.grow_dir:
        #     gaussians.fake_grow()
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, grow_dir=dataset.grow_dir,densify_grad_threshold = opt.densify_grad_threshold,iteration=iteration,opt=opt,continous_dir=dataset.continous_dir,grow_distance=dataset.grow_distance,modelcg=dataset,cameras_extent=scene.cameras_extent)
        image, viewspace_point_tensor, visibility_filter, radii, selected_pts_mask = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["selected_pts_mask"]
        # print('shape of image',image.shape)
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # if iteration % 50 == 1:
        #     to_save = torch.cat([image,gt_image],axis=1)
        #     torchvision.utils.save_image(to_save, os.path.join(scene.model_path, f'{iteration:05d}' + ".png"))
        for k in range(int(image.shape[0]/3)):
            Ll1 = l1_loss(image[k*3:(k+1)*3], gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image[k*3:(k+1)*3], gt_image))
        prune_mask = (gaussians.get_opacity < 0.005).squeeze()
        # print('to prune shape',gaussians.get_opacity[prune_mask].shape)
        if opt.opacitysparse>0 and torch.sum(prune_mask)>0:
            opacity_Ll1 = opt.opacitysparse*l1_loss(gaussians.get_opacity[prune_mask],1)
            loss = loss+opacity_Ll1
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),grow_dir=dataset.grow_dir,densify_grad_threshold = opt.densify_grad_threshold,opt=opt,continous_dir=dataset.continous_dir,grow_distance=dataset.grow_distance,modelcg=dataset,cameras_extent=scene.cameras_extent)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # print(gaussians.max_radii2D.shape,visibility_filter.shape,radii.shape)
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold,opt,iteration)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                        
            # if dataset.grow_dir:
            #     gaussians.remove_fake_grow()

            if iteration in checkpoint_iterations:
                # print('group',gaussians.optimizer.param_groups)
                # for group in gaussians.optimizer.param_groups:
                    # print(group["name"])
                    # stored_state = gaussians.optimizer.state.get(group['params'][0], None)
                    # print(group["params"][0].shape)
                    # print(gaussians.optimizer.state)
                    # print(stored_state["exp_avg"].shape,stored_state["exp_avg_sq"].shape)
                # print('state',gaussians.optimizer.state.items())
                # for k,v in gaussians.optimizer.state.items():
                    # print('---------------k--------------',k)
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        # print('self.optimizer state',gaussians.optimizer.state)
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, grow_dir=False,densify_grad_threshold = 0, opt=None,continous_dir=False,grow_distance=False,modelcg=None,cameras_extent=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # if iteration % 50 == 1 or iteration%100<2 or iteration%100==99:
    if iteration%5000<2:
        viewpoint = scene.getTestCameras()[0]
        image_withoutgrow = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, grow_dir = False,continous_dir=False,grow_distance=False,modelcg=None,cameras_extent=None)["render"], 0.0, 1.0)
        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, grow_dir = grow_dir, densify_grad_threshold = densify_grad_threshold, iteration=iteration, opt=opt,continous_dir=continous_dir,grow_distance=grow_distance,modelcg=modelcg,cameras_extent=cameras_extent)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        if image.shape[0]==3:
            to_save = torch.cat((image_withoutgrow,image,gt_image),axis=2)
        elif image.shape[0]==6:
            to_save = torch.cat((image[:3,...],image[3:,...],gt_image),axis=2)
        torchvision.utils.save_image(to_save, os.path.join(scene.model_path, f'val_{iteration:05d}' + ".png"))

        viewpoint = scene.getTrainCameras()[0]
        image_withoutgrow = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, grow_dir = False,continous_dir=False,grow_distance=False,modelcg=None,cameras_extent=None)["render"], 0.0, 1.0)
        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, grow_dir = grow_dir, densify_grad_threshold = densify_grad_threshold, iteration=iteration, opt=opt,continous_dir=continous_dir,grow_distance=grow_distance,modelcg=modelcg,cameras_extent=cameras_extent)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        if image.shape[0]==3:
            to_save = torch.cat((image_withoutgrow,image,gt_image),axis=2)
        elif image.shape[0]==6:
            to_save = torch.cat((image[:3,...],image[3:,...],gt_image),axis=2)
        torchvision.utils.save_image(to_save, os.path.join(scene.model_path, f'train_{iteration:05d}' + ".png"))

    testing_iterations_new = [50*i for i in range(1000)]
    # Report test and samples of training set
    if iteration in testing_iterations_new:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if iteration == testing_iterations[0]:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        torch.cuda.empty_cache()

    if iteration in testing_iterations_new:
        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    opacity_histogram_iterations = sorted([3000*i+100 for i in range(6)]+[3000*i+200 for i in range(6)]+[3000*i+300 for i in range(6)])
    if iteration in opacity_histogram_iterations:
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
    # if iteration % opt.densification_interval == 0:
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000,30_000,40000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000,30_000,40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000, 30_000,40000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
