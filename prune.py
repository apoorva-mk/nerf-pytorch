import os, sys
import numpy as np
import imageio
import json
import random
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *
from run_nerf import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from torch.nn.utils import prune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

class ActivationAwareDNNPruning():
    def layer_wise_pruning(self, model, layers, activations, percent=0.3):
        pruned_weights = []
        if len(layers) != len(activations):
            print("Error: Unable to prune, due to different size of activations and layers")
        for i in range(len(layers)):
            cumulative_activations = torch.sum(activations[i], dim=0)
            num_to_remove = int(cumulative_activations.numel() * percent)
            sorted_indices = torch.argsort(cumulative_activations.flatten())
            mask = torch.ones_like(cumulative_activations, dtype=torch.uint8)
            mask.flatten()[sorted_indices[:num_to_remove]] = 0
            # print("Masked array:", mask.shape)
            layers[i].weight.requires_grad = False
            # print(layers[i].weight.requires_grad)
            # layers[i] = prune.custom_from_mask(layers[i], name="weight", mask=mask.unsqueeze(1))
            layers[i].weight = nn.Parameter(layers[i].weight * mask.unsqueeze(1))
            pruned_weights.append(torch.count_nonzero(layers[i].weight))

            # print(layers[i].weight)

        param_size = 0.0
        pruned_param_size = 0.0
        pruned_index = 0
        pattern = re.compile(r"pts_linears\..*\.weight")

        for name, param in model.named_parameters():
            if pattern.match(name):
                # print("PRUNED", name, param.nelement(), pruned_weights[pruned_index])
                pruned_param_size += (pruned_weights[pruned_index]) * param.element_size()
                pruned_index += 1
                param_size += param.nelement() * param.element_size()
            else:
                param_size += param.nelement() * param.element_size()
                pruned_param_size += param.nelement() * param.element_size()

        # print(param_size, pruned_param_size)
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        pruned_size_all_mb = (pruned_param_size + buffer_size) / 1024**2

        print('Model size: {:.3f}MB'.format(size_all_mb))
        print('Pruned model size: {:.3f}MB'.format(pruned_size_all_mb))

        return layers


# Pass arguments: 
# sample_size: Number of samples to get calibration set for the pruning process
# 

def prune(percent_to_prune=0.3):
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    pruned_model = render_kwargs_train['network_fn']

     # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)


    N_rand = args.N_rand
    use_batching = not args.no_batching
    render_kwargs_train['network_fn'].save_activations = True
    for i in range(args.sample_size):
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3,:4]

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        model = (render_kwargs_train['network_fn'])
        print("Activations: ", len(model.activations)," ", len(model.pts_linears), "\n", model.activations)
    render_kwargs_train['network_fn'].save_activations = False
    # # Pruning the model
    dnn_pruning = ActivationAwareDNNPruning()
    model.pts_linears = dnn_pruning.layer_wise_pruning(model, model.pts_linears, model.activations, percent_to_prune)

    print("MOVIE\n")
    with torch.no_grad():
        # render_kwargs_test['network_fn'].save_activations = render_kwargs_train['network_fn'].save_activations = False
        print(render_kwargs_test['network_fn'].save_activations)
        render_kwargs_test['network_fn'] = render_kwargs_train['network_fn']
        rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_train)
    print('Done, saving', rgbs.shape, disps.shape)
    moviebase = os.path.join(args.basedir, args.expname, '{}_spiral_pruned_{}_'.format(args.expname, int(percent_to_prune*100)))
    imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
    imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

    # # Calculating loss and PSNR over test set
    # print('TEST views are', i_test)

    with torch.no_grad():
        loss = 0.0
        psnr = 0.0
        for i in i_train:
            print(i)
            img_i = np.random.choice(i_train)
            target = images[i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)

        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss += img_loss
        psnr += mse2psnr(img_loss)
    loss = loss / len(i_test)
    psnr = psnr / len(i_test)
    print(f"Over test set, \n loss: {loss} \n psnr : {psnr}")


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    prune(0.50)
