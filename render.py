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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
###
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
###

def render_set(model_path, name, iteration, views, p_views_1, p_views_2, p_views_3, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    pr_path_1 = os.path.join(model_path, name, "ours_{}".format(iteration), "perturbation_render_1")
    pr_path_2 = os.path.join(model_path, name, "ours_{}".format(iteration), "perturbation_render_2")
    pr_path_3 = os.path.join(model_path, name, "ours_{}".format(iteration), "perturbation_render_3")
    p_path_1 = os.path.join(model_path, name, "ours_{}".format(iteration), "perturbation_depth_1")
    p_path_2 = os.path.join(model_path, name, "ours_{}".format(iteration), "perturbation_depth_2")
    p_path_3 = os.path.join(model_path, name, "ours_{}".format(iteration), "perturbation_depth_3")


    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(pr_path_1, exist_ok=True)
    makedirs(pr_path_2, exist_ok=True)
    makedirs(pr_path_3, exist_ok=True)
    makedirs(p_path_1, exist_ok=True)
    makedirs(p_path_2, exist_ok=True)
    makedirs(p_path_3, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        depth = render_pkg["depth"]
        gt = view.original_image[0:3, :, :]
        ##########
        scale_nor = depth.max().item()
        depth_nor = depth / scale_nor
        depth_tensor_squeezed = depth_nor.squeeze()  # Remove the channel dimension
        colormap = plt.get_cmap('jet')
        depth_colored = colormap(depth_tensor_squeezed.cpu().numpy())
        depth_colored_rgb = depth_colored[:, :, :3]
        depth_image = Image.fromarray((depth_colored_rgb * 255).astype(np.uint8))
        output_path = os.path.join(depth_path, '{0:05d}'.format(idx) + ".png")
        depth_image.save(output_path)
        ##########
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    if name == 'train':
        for idx, view in enumerate(tqdm(p_views_1, desc="Rendering progress")):
            render_pkg = render(view, gaussians, pipeline, background)
            p_render_1 = render_pkg["render"]
            p_depth_1 = render_pkg["depth"]
            torchvision.utils.save_image(p_depth_1, os.path.join(p_path_1, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(p_render_1, os.path.join(pr_path_1, '{0:05d}'.format(idx) + ".png"))
        for idx, view in enumerate(tqdm(p_views_2, desc="Rendering progress")):
            render_pkg = render(view, gaussians, pipeline, background)
            p_render_2 = render_pkg["render"]
            p_depth_2 = render_pkg["depth"]
            torchvision.utils.save_image(p_depth_2, os.path.join(p_path_2, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(p_render_2, os.path.join(pr_path_2, '{0:05d}'.format(idx) + ".png"))
        for idx, view in enumerate(tqdm(p_views_3, desc="Rendering progress")):
            render_pkg = render(view, gaussians, pipeline, background)
            p_render_3 = render_pkg["render"]
            p_depth_3 = render_pkg["depth"]
            torchvision.utils.save_image(p_depth_3, os.path.join(p_path_3, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(p_render_3, os.path.join(pr_path_3, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, api_key=None, self_refinement=None, num_prompt=None, max_rounds=None)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
             scene.getPerturbationCameras(stage=1), scene.getPerturbationCameras(stage=2), scene.getPerturbationCameras(stage=3), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
             scene.getPerturbationCameras(stage=1), scene.getPerturbationCameras(stage=2), scene.getPerturbationCameras(stage=3), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)