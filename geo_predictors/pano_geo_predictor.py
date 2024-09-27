import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

from .geo_predictor import GeoPredictor
from .omnidata_predictor import OmnidataPredictor

from fields.networks import VanillaMLP
import tinycudann as tcnn

from utils.geo_utils import panorama_to_pers_directions
from utils.camera_utils import *


def scale_unit(x):
    return (x - x.min()) / (x.max() - x.min())


class GeometricField(nn.Module):
    def __init__(self,
                 n_levels=16,
                 log2_hashmap_size=19,
                 base_res=16,
                 fine_res=2048):
        super().__init__()
        per_level_scale = np.exp(np.log(fine_res / base_res) / (n_levels - 1))
        self.hash_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",
            }
        )

        self.geo_mlp = VanillaMLP(dim_in=n_levels * 2 + 3,
                                  dim_out=1,
                                  n_neurons=64,
                                  n_hidden_layers=2,
                                  sphere_init=True,
                                  weight_norm=False)

    def forward(self, directions, requires_grad=False):
        if requires_grad:
            if not self.training:
                directions = directions.clone()  # get a copy to enable grad
            directions.requires_grad_(True)

        dir_scaled = directions * 0.49 + 0.49
        selector = ((dir_scaled > 0.0) & (dir_scaled < 1.0)).all(dim=-1).to(torch.float32)
        scene_feat = self.hash_grid(dir_scaled)

        distance = F.softplus(self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0] + 1.)

        if requires_grad:
            grad = torch.autograd.grad(
                distance, directions, grad_outputs=torch.ones_like(distance),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            return distance, grad
        else:
            return distance


class PanoGeoPredictor(GeoPredictor):
    def __init__(self):
        super().__init__()
        self.depth_predictor = OmnidataPredictor()

    def grads_to_normal(self, grads):
        grads = grads.cpu()
        height, width, _ = grads.shape
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        ortho_a = torch.randn([height, width, 3])
        ortho_b = torch.linalg.cross(pano_dirs, ortho_a)
        ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
        ortho_a = torch.linalg.cross(ortho_b, pano_dirs)
        ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

        val_a = (grads * ortho_a).sum(-1, True) * pano_dirs + ortho_a
        val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
        val_b = (grads * ortho_b).sum(-1, True) * pano_dirs + ortho_b
        val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)

        normals = torch.cross(val_a, val_b)
        normals = normals / torch.linalg.norm(normals, 2, -1, True)
        is_inside = ((normals * pano_dirs).sum(-1, True) < 0.).float()
        normals = normals * is_inside + -normals * (1. - is_inside)
        return normals.cuda()

    def __call__(self, img, gen_res=512, reg_loss_weight=1e-1,):
        '''
        :param img: [H, W, 3]
        :param ref_distance: [H, W] or [H, W, 1]
        :param mask: [H, W] or [H, W, 1], value 1 if need to be inpainted otherwise 0
        :return: inpainted distance [H, W]
        '''

        height, width, _ = img.shape
        device = img.device
        img = img.clone().squeeze().permute(2, 0, 1)    

        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = [], [], [], [], []
        for i in range(12):
            ratio = 1.1
            cur_pers_dirs, cur_pers_ratios, cur_to_vecs, cur_down_vecs, cur_right_vecs = panorama_to_pers_directions(gen_res=gen_res, ratio=ratio, ex_rot='rand')
            pers_dirs.append(cur_pers_dirs)
            pers_ratios.append(cur_pers_ratios)
            to_vecs.append(cur_to_vecs)
            down_vecs.append(cur_down_vecs)
            right_vecs.append(cur_right_vecs)

        pers_dirs = torch.cat(pers_dirs, 0)
        pers_ratios = torch.cat(pers_ratios, 0)
        to_vecs = torch.cat(to_vecs, 0)
        down_vecs = torch.cat(down_vecs, 0)
        right_vecs = torch.cat(right_vecs, 0)

        fx = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(right_vecs, 2, -1, True) * gen_res * .5
        fy = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(down_vecs, 2, -1, True) * gen_res * .5
        cx = torch.ones_like(fx) * gen_res * .5
        cy = torch.ones_like(fy) * gen_res * .5

        pers_dirs = pers_dirs.to(device)
        pers_ratios = pers_ratios.to(device)
        to_vecs = to_vecs.to(device)
        down_vecs = down_vecs.to(device)
        right_vecs = right_vecs.to(device)

        rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                               down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                               to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                              dim=1)

        rot_c2w = torch.linalg.inv(rot_w2c)

        n_pers = len(pers_dirs)
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)

        pers_imgs = F.grid_sample(img[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]



        ####################################### Start Optimization #####################################################

        pers_imgs_pc = pers_imgs[:20]
        pers_dirs_pc = pers_dirs[:20]
        n_pers_pc = len(pers_dirs_pc)

        pred_distances_raw = []

        for i in range(n_pers_pc):
            with torch.no_grad():
                intri = {
                    'fx': fx[i].item(),
                    'fy': fy[i].item(),
                    'cx': cx[i].item(),
                    'cy': cy[i].item()
                }

                pred_depth = self.depth_predictor.predict_depth(pers_imgs_pc[i: i+1], intri=intri).cuda().clip(0., None)  # [1, 1, res, res]
                pred_depth = pred_depth / (pred_depth.mean() + 1e-5)
                pred_depth_ratio = pred_depth * pers_ratios[i].permute(2, 0, 1)[None]
                pred_distances_raw.append(pred_depth_ratio)

        pred_distances_raw = torch.cat(pred_distances_raw, dim=0)  # [n_pers, 1, res, res]
        pers_dirs_pc = pers_dirs_pc.permute(0, 3, 1, 2)

        sup_infos = torch.cat([pers_dirs_pc, pred_distances_raw], dim=1)

        scale_params = torch.zeros([n_pers_pc], requires_grad=True)
        bias_params_local_distance  = torch.zeros([n_pers_pc, 1, gen_res, gen_res], requires_grad=True)

        geo_field = GeometricField(fine_res = 2048).cuda()

        # Stage 1: Optimize global parameters
        all_iter_steps = 1500
        lr_alpha = 1e-2
        init_lr = 1e-1
        init_lr_sp = 1e-2
        init_lr_local = 1e-1
        local_batch_size = 256

        optimizer_sp = torch.optim.Adam(geo_field.parameters(), lr=init_lr_sp)
        optimizer_global = torch.optim.Adam([scale_params], lr=init_lr)
        optimizer_local = torch.optim.Adam([bias_params_local_distance], lr=init_lr_local)

        for phase in ['global', 'hybrid']:
            ###
            ema_loss_for_log = 0.0
            progress_bar = tqdm(range(1, all_iter_steps + 1), desc="Training progress")
            loss_vis = []
            ###
            #for iter_step in tqdm(range(all_iter_steps)):
            for iter_step in range(1, all_iter_steps + 1):
                progress = iter_step / all_iter_steps
                if phase == 'global':
                    progress = progress * .5
                else:
                    progress = progress * .5 + .5

                lr_ratio = (np.cos(progress * np.pi) + 1.) * (1. - lr_alpha) + lr_alpha
                for g in optimizer_global.param_groups:
                    g['lr'] = init_lr * lr_ratio
                for g in optimizer_local.param_groups:
                    g['lr'] = init_lr_local * lr_ratio
                for g in optimizer_sp.param_groups:
                    g['lr'] = init_lr_sp * lr_ratio

                #idx = np.random.randint(low=0, high=n_pers)
                sample_coords = torch.rand(n_pers_pc, local_batch_size, 1, 2).cuda() * 2. - 1           # [n_pers, local_batch_size, 1, 2] range (-1, +1)
                cur_sup_info = F.grid_sample(sup_infos, sample_coords, padding_mode='border')        # [n_pers, 7, local_batch_size, 1]
                distance_bias = F.grid_sample(bias_params_local_distance.cuda(), sample_coords, padding_mode='border')  # [n_pers, 1, local_batch_size, 1]
                distance_bias = distance_bias[:, :, :, 0].permute(0, 2, 1)                              # [n_pers, local_batch_size, 1]

                dirs = cur_sup_info[:, :3, :, 0].permute(0, 2, 1)                                    # [n_pers, local_batch_size, 3]
                dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)

                ref_pred_distances = cur_sup_info[:, 3: 4, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                ref_pred_distances = ref_pred_distances * F.softplus(scale_params[:, None, None].cuda())  # [n_pers, local_batch_size, 1]
                ref_pred_distances = ref_pred_distances + distance_bias

                pred_distances = geo_field(dirs.reshape(-1, 3), requires_grad=False)
                pred_distances = pred_distances.reshape(n_pers_pc, local_batch_size, 1)

                distance_loss = F.smooth_l1_loss(ref_pred_distances, pred_distances, beta=5e-1, reduction='mean')
                reg_loss = ((F.softplus(scale_params).mean() - 1.)**2).mean()

                if phase == 'hybrid':
                    distance_bias_local = bias_params_local_distance
                    distance_bias_tv_loss = F.smooth_l1_loss(distance_bias_local[:, :, 1:, :], distance_bias_local[:, :, :-1, :], beta=1e-2) + \
                                            F.smooth_l1_loss(distance_bias_local[:, :, :, 1:], distance_bias_local[:, :, :, :-1], beta=1e-2)

                else:
                    distance_bias_tv_loss = 0.
                loss = distance_loss +\
                       reg_loss * reg_loss_weight +\
                       distance_bias_tv_loss 
              

                optimizer_global.zero_grad()
                optimizer_sp.zero_grad()
                if phase == 'hybrid':
                    optimizer_local.zero_grad()

                loss.backward()
                optimizer_global.step()
                optimizer_sp.step()
                if phase == 'hybrid':
                    optimizer_local.step()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    loss_vis.append(ema_loss_for_log) ###
                    if iter_step % 1 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                        progress_bar.update(1)
                    if iter_step == all_iter_steps:
                        progress_bar.close()


        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width)).cuda()
        new_distances = geo_field(pano_dirs.reshape(-1, 3), requires_grad=False)
        new_distances = new_distances.detach().reshape(height, width, 1)

        return new_distances, rot_w2c, fx, fy, cx, cy, pers_imgs



