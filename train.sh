# python train_gs_mvs_nerf_finetuning_pl.py --dataset_name dtu_ft_gs --datadir  /anfs/gfxdisp/hanxue_nerf_data/dtu_example/scan114 --savedir /anfs/gfxdisp/hanxue_nerf_data/mvsnerf_exp --expname single_volume-simgleimage-lr5e-4 --with_rgb_loss  --batch_size 1 --num_epochs 30000  --imgScale_test 1.0   --pad 24 --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 10 --net_type v3 --netchunk 100000

# python train_gs_mvs_nerf_pl.py --dataset_name dtu_gs --datadir /anfs/gfxdisp/hanxue_nerf_data/data/dtu --savedir /anfs/gfxdisp/hanxue_nerf_data/mvsnerf_exp --expname dtu_gs_114_nopointloss_grow1w_multi_volume_lr1e-4 --increaseactivation_step 10000 --num_epochs 10000  --multi_volume --lrate 0.0001  --imgScale_test 1.0   --with_rgb_loss  --batch_size 1 --pad 24 --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 50 --net_type v3 --netchunk 100000 

python train_gs_mvs_nerf_pl.py --dataset_name dtu_gs --datadir /anfs/gfxdisp/hanxue_nerf_data/data/dtu --savedir /anfs/gfxdisp/hanxue_nerf_data/mvsnerf_exp --expname dtu_gs_withmask_onescale_point10_nopointloss_grow1w_multi_volume_lr1e-4 --singlescale --increaseactivation_step 10000 --num_epochs 10000  --multi_volume --lrate 0.0001  --imgScale_test 1.0   --with_rgb_loss  --batch_size 1 --pad 24 --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 5000 --net_type v3 --netchunk 100000 
