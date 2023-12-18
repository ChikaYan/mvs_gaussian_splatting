import imageio
import glob
import os
import numpy as np
import cv2
# imgfiles = sorted(glob.glob(os.path.join('./','train_*.png')))
# frames=[]
# for file in imgfiles:
#     im=imageio.imread(file)
#     frames.append(im)
# frames=np.array(frames)
# imageio.mimwrite(os.path.join('./','sanitycheck.mp4'), frames, fps=10, quality=8)

imids = [3001,3026,4201,4226,6001,6026,7201,7226,9001,9026,10201,10226]
for id in imids:
    # im=imageio.imread(os.path.join('/anfs/gfxdisp/hanxue_nerf_data/gs_exp/fern_4/resetstopgrow1000_notdetach_growinterval200_grad0.0002_lr0.1',f'train_{id:05d}' + ".png"))
    im=imageio.imread(os.path.join('/rds/user/hl589/hpc-work/gs_exp/fern_4/resetstopgrow1000_notprune_notdetach_growinterval200_grad0.0002_lr0.1',f'train_{id:05d}' + ".png"))
    image_withoutgrow = im[:,:1008,:]
    image = im[:,1008:1008*2,:]
    gt = im[:,1008*2:,:]
    diff_withoutgrow = cv2.applyColorMap(np.absolute(gt-image_withoutgrow)*10, cv2.COLORMAP_JET)
    # diff_withoutgrow = np.absolute(gt-image_withoutgrow)*10
    diff_grow = cv2.applyColorMap(np.absolute(gt-image)*10, cv2.COLORMAP_JET)
    # diff_grow = np.absolute(gt-image)*10
    im_vis = np.concatenate((im,diff_withoutgrow,diff_grow),axis=1)
    print(im_vis.shape)
    # imageio.imwrite(os.path.join('/anfs/gfxdisp/hanxue_nerf_data/gs_exp/fern_4/resetstopgrow1000_notdetach_growinterval200_grad0.0002_lr0.1',f'compare_train_{id:05d}' + ".png"),im_vis)
    imageio.imwrite(os.path.join('/rds/user/hl589/hpc-work/gs_exp/fern_4/resetstopgrow1000_notprune_notdetach_growinterval200_grad0.0002_lr0.1',f'compare_train_{id:05d}' + ".png"),im_vis)