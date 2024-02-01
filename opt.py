import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--savedir", type=str,default='exp',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--with_depth_loss', action='store_true')
    parser.add_argument('--with_rgb_loss', action='store_true')
    parser.add_argument('--imgScale_train', type=float, default=1.0)
    parser.add_argument('--imgScale_test', type=float, default=1.0)
    parser.add_argument('--img_downscale', type=float, default=1.0)
    parser.add_argument('--pad', type=int, default=24)

    # loader options
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--pts_dim", type=int, default=3)
    parser.add_argument("--dir_dim", type=int, default=3)
    parser.add_argument("--alpha_feat_dim", type=int, default=8)
    parser.add_argument('--net_type', type=str, default='v0')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['dtu', 'blender', 'llff', 'dtu_ft','scannet','dtu_ft_gs','dtu_gs', 'google_scan_gs'])
    parser.add_argument('--use_color_volume', default=False, action="store_true",
                        help='project colors into a volume without indexing from image everytime')
    parser.add_argument('--use_density_volume', default=False, action="store_true",
                        help='point sampling with density')

    # training options
    parser.add_argument("--netdepth", type=int, default=6,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=6,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=128,
                        help='channels per layer in fine network')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--model_lr", type=float, default=5e-4,
                        help='model learning rate')
    parser.add_argument("--volume_lr", type=float, default=5e-4,
                        help='model learning rate')
    parser.add_argument("--opacity_lr", type=float, default=0.05,
                        help='model learning rate')
    parser.add_argument("--scaling_lr", type=float, default=0.01,
                        help='model learning rate')
    parser.add_argument("--feat_ds_lr", type=float, default=0.01,
                        help='model learning rate')
    parser.add_argument("--multi_volume", action='store_true',
                        help='whether use multi_volume')
    parser.add_argument("--withpointrgbloss", action='store_true',
                        help='whether use multi_volume')
    parser.add_argument("--increaseactivation_step",  type=int, default=300000,
                        help='whether use multi_volume')
    parser.add_argument('--decay_step', nargs='+', type=int, default=[5000, 8000, 9000],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.5,
                        help='learning rate decay amount')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')

    parser.add_argument("--chunk", type=int, default=1024,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--depth_res", type=int, default=128)
    parser.add_argument("--num_gpus", type=int, default=-1)
    parser.add_argument("--lambda_dssim", type=float, default=0)
    parser.add_argument("--lambda_lpips", type=float, default=0)
    parser.add_argument("--lambda_mrf", type=float, default=0)
    parser.add_argument("--pt_folder", type=str, default='Pointclouds50',
                        help='where to find point cloud for querying')
    parser.add_argument("--pt_downsample", type=int, default=1,
                        help='further downsample the loaded pts')


    # rendering options
    parser.add_argument("--N_samples", type=int, default=128,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--singlescale", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument('--decay_scale', type=float, default=0.1,
                        help='learning rate decay amount')
    parser.add_argument("--n_views", type=int, default=3,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--featurenet_outputdim", type=int, default=32,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--volume_feat_outputdim", type=int, default=8,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--nosave", action='store_true',
                        help='whether save images during validation')




    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')


    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=20,
                        help='frequency of visualize the depth')
    parser.add_argument("--val_only", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()