#!/bin/bash
#SBATCH --account OZTIRELI-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time=4:30:00
#SBATCH --mail-type=NONE

. /etc/profile.d/modules.sh
module unload cuda/8.0
module unload rhel8/default-gpu
module unload cuda/11.4
module unload ucx/1.11.1/gcc-9.4.0-lktqyl4
module unload cuda/11.4.0/gcc-9.4.0-3hnxhjt
module load cuda/11.8
module load cudnn/8.9_cuda-11.8

module unload gcc-5.4.0-gcc-4.8.5-fis24gg
module load gcc-7.2.0-gcc-4.8.5-pqn7o2k
source /usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh
conda activate mvsnerf
# python train.py -s /rds/user/hl589/hpc-work/data/nerf_llff_data/360_v2/garden -r 4 --port 6021 --eval --model_path /rds/user/hl589/hpc-work/gs_exp/garden_4/baseline --densify_grad_threshold 0.0002 --min_opacity 0.005 
# python train.py -s /rds/user/hl589/hpc-work/data/nerf_llff_data/360_v2/garden -r 4 --port 6021 --eval --model_path /rds/user/hl589/hpc-work/gs_exp/garden_4/reinit_notdetach_interval100_grad0.0002_min0.005_lr0.01   --grow_dir --growdirs_lr 0.01 --densification_interval 100 --densify_grad_threshold 0.0002 --min_opacity 0.005 
# python train.py -s /rds/user/hl589/hpc-work/data/nerf_llff_data/360_v2/garden -r 4 --port 6021 --eval --model_path /rds/user/hl589/hpc-work/gs_exp/garden_4/reinit_notdetach_interval100_grad0.0002_min0.005_lr0.02   --grow_dir --growdirs_lr 0.02 --densification_interval 100 --densify_grad_threshold 0.0002 --min_opacity 0.005
python train.py -s /rds/user/hl589/hpc-work/data/nerf_llff_data/360_v2/garden -r 4 --port 6021 --eval --model_path /rds/user/hl589/hpc-work/gs_exp/garden_4/reinit_notdetach_interval100_grad0.0002_min0.005_lr0.05   --grow_dir --growdirs_lr 0.05 --densification_interval 100 --densify_grad_threshold 0.0002 --min_opacity 0.005 
python train.py -s /rds/user/hl589/hpc-work/data/nerf_llff_data/360_v2/garden -r 4 --port 6021 --eval --model_path /rds/user/hl589/hpc-work/gs_exp/garden_4/reinit_notdetach_interval100_grad0.0002_min0.005_lr0.005   --grow_dir --growdirs_lr 0.005 --densification_interval 100 --densify_grad_threshold 0.0002 --min_opacity 0.005 