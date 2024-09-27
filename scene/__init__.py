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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, CameraInfo, SceneInfo ###
from scene.gaussian_model import GaussianModel, BasicPointCloud ###
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, img_coord_to_pano_direction ###
###
from geo_predictors.pano_geo_predictor import *
from utils.utils import read_image
from utils.sh_utils import SH2RGB
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from PIL import Image
import trimesh
import cv2 as cv
from utils.save_data import save_data
import sys
import importlib
sys.path.append('stitch_diffusion/kohya_trainer')

def pcd_from_depths(pano_img, distances, height, width, source_path):
    # ### save pano depth map
    # import matplotlib.pyplot as plt
    # scale_nor = distances.max().item()
    # distances_nor = distances / scale_nor
    # depth_tensor_squeezed = distances_nor.squeeze()  # Remove the channel dimension
    # colormap = plt.get_cmap('jet')
    # depth_colored = colormap(depth_tensor_squeezed.cpu().numpy())
    # depth_colored_rgb = depth_colored[:, :, :3]
    # depth_image = Image.fromarray((depth_colored_rgb * 255).astype(np.uint8))
    # output_path = "./pano_depth_map.png"
    # depth_image.save(output_path)
    # ###
    pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width)).cuda()
    scale = distances.max().item() * 0.7 #* 0.8 #* 1.05
    distances /= scale
    pts = pano_dirs * distances.squeeze()[..., None]
    pts = pts.cpu().numpy().reshape(-1, 3)
    # pcd = trimesh.PointCloud(pts, pano_img.reshape(-1, 3).cpu().numpy())
    # pcd_path = os.path.join(source_path, 'point_cloud.ply')
    # pcd.export(pcd_path)
    return pts

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def get_info_from_params(source_path, pano_img, distances, rot_w2c, fx, fy, cx, cy, pers_imgs, pts):
        H, W, _ = pano_img.shape
        n_pers, _, h, w = pers_imgs.shape
        cam_infos_unsorted = []
        cam_perturbation_infos_unsorted = [] ###
        cam_perturbation_infos_unsorted_stage2 = [] ###
        cam_perturbation_infos_unsorted_stage3 = [] ###
        for i in range(n_pers):
            with torch.no_grad():
                img = pers_imgs[i].cpu().numpy()
                img = img.transpose(1, 2, 0)
                img = (img*255).astype('uint8')
                img = Image.fromarray(img)
                intri = {
                    'fx': fx[i].item(),
                    'fy': fy[i].item(),
                    'cx': cx[i].item(),
                    'cy': cy[i].item()
                }
                fovx = focal2fov(intri['fx'], w)
                fovy = focal2fov(intri['fy'], h)
                R = np.transpose(np.asarray(rot_w2c[i].cpu()))
                T = np.transpose(np.array( [0, 0, 0]))
                T_perturbation = T + np.random.uniform(-0.05, 0.05, size=(1, 3)) ###
                uid = i
                image_name = 'image' + str(i)
                try:
                    os.mkdir( os.path.join ( source_path, 'images'))
                except Exception as e:
                    pass
                image_path = os.path.join(source_path, 'images', image_name)

                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=fovy, FovX=fovx, image=img,
                              image_path=image_path, image_name=image_name, width=w, height=h)      
                cam_infos_unsorted.append(cam_info)
                ### stage 1 perturbation
                cam_perturbation_info = CameraInfo(uid=uid, R=R, T=T_perturbation, FovY=fovy, FovX=fovx, image=img,
                              image_path=image_path, image_name=image_name, width=w, height=h)
                cam_perturbation_infos_unsorted.append(cam_perturbation_info)
                ### stage 2 perturbation
                T_perturbation_stage2 = T + np.random.uniform(-0.05 * 2, 0.05 * 2, size=(1, 3))
                cam_perturbation_info_stage2  = CameraInfo(uid=uid, R=R, T=T_perturbation_stage2, FovY=fovy, FovX=fovx, image=img,
                              image_path=image_path, image_name=image_name, width=w, height=h)
                cam_perturbation_infos_unsorted_stage2 .append(cam_perturbation_info_stage2)
                ### stage 3 perturbation
                T_perturbation_stage3 = T + np.random.uniform(-0.05 * 4, 0.05 * 4, size=(1, 3))
                cam_perturbation_info_stage3  = CameraInfo(uid=uid, R=R, T=T_perturbation_stage3, FovY=fovy, FovX=fovx, image=img,
                              image_path=image_path, image_name=image_name, width=w, height=h)
                cam_perturbation_infos_unsorted_stage3 .append(cam_perturbation_info_stage3)



        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        cam_perturbation_infos = sorted(cam_perturbation_infos_unsorted.copy(), key = lambda x : x.image_name) ###
        cam_perturbation_infos_stage2 = sorted(cam_perturbation_infos_unsorted_stage2.copy(), key = lambda x : x.image_name) ###
        cam_perturbation_infos_stage3 = sorted(cam_perturbation_infos_unsorted_stage3.copy(), key = lambda x : x.image_name) ###
        llffhold = 8
        if eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            perturbation_cam_infos = cam_perturbation_infos ###
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []
            perturbation_cam_infos = cam_perturbation_infos ###
        nerf_normalization = getNerfppNorm(train_cam_infos)
        # #random initialization (comment for using the input pcd)
        # num_pts = 100000
        # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        # shs = np.random.random((num_pts, 3)) / 255
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        xyz = pts #pcd_from_depths(pano_img, distances, H, W, source_path)
        vertex_colors = pano_img.reshape(-1, 3).cpu().numpy()
        ply_path = os.path.join(source_path, 'sparse/0/points3D.ply')
        pcd = BasicPointCloud(points = xyz, colors=vertex_colors, normals=np.zeros_like(xyz))
        scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           perturbation_cameras_stage1=perturbation_cam_infos, ###
                           perturbation_cameras_stage2=cam_perturbation_infos_stage2, ###
                           perturbation_cameras_stage3=cam_perturbation_infos_stage3, ###
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

        return scene_info


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel,  api_key, self_refinement, num_prompt, max_rounds, load_iteration=None, shuffle=True, resolution_scales=[1.0],):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.perturbation_cameras_stage1 = {} ###
        self.perturbation_cameras_stage2 = {} ###
        self.perturbation_cameras_stage3 = {} ###

        ## Change loading multi views data to pano ###
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval) 
        ###############################################
        elif any(filename.endswith('.png') for filename in os.listdir(args.source_path)) or any(filename.endswith('.txt') for filename in os.listdir(args.source_path)):
            #img = None
            if (any(filename.endswith('.png') for filename in os.listdir(args.source_path))):
                files = [f for f in os.listdir(args.source_path) if f.endswith('.png')]
                img_path = os.path.join(args.source_path, files[0]) ### only 1 pano image in the folder
                img = read_image(img_path, to_torch=True, squeeze=True).cuda()
            elif (any(filename.endswith('.txt') for filename in os.listdir(args.source_path))):
                sdk = importlib.import_module('stitch_diffusion.kohya_trainer.StitchDiffusionPipeline')
                imgrun = importlib.import_module('Text2PanoRunner')
                if (self_refinement):
                    assert api_key, "You must enter an api key to access prompt engineered diffusion output"
                    txtfile = [f for f in os.listdir(args.source_path) if f.endswith('.txt')][0]
                    runner = imgrun.Text2PanoRunner(api_key = api_key, testfile = os.path.join(args.source_path, txtfile) , num_prompt = num_prompt, max_rounds = max_rounds, foldername = txtfile.rstrip(".txt"))  
                    runner.run_command()
                    img_name = "self_refinement/" + txtfile.rstrip(".txt") + "/iter_best/image.png"
                    os.system("cp " + img_name + " " + os.path.join(args.source_path, "image.png"))
                else:
                    sd = sdk.StitchDiffusion(
                        sdk.my_args
                        )
                    txtfile = [f for f in os.listdir(args.source_path) if f.endswith('.txt')][0]
                    txtfile = os.path.join(args.source_path, txtfile)
                    with open(txtfile) as f:
                        prompt = f.read()
                    sd.inference(prompt, savename=os.path.join(args.source_path, "diffusion_img.png"))
                    img_name = os.path.join(args.source_path, "diffusion_img.png")
                img = read_image(img_name, to_torch=True, squeeze=True).cuda()
            
            # if img.shape[:2] != (512, 1024):
            #     img = cv.resize(img.cpu().numpy(), (1024, 512), cv.INTER_AREA)
            #     img = torch.from_numpy(img).cuda()
            img = cv.resize(img.cpu().numpy(), (2048, 1024), cv.INTER_AREA)
            img = torch.from_numpy(img).cuda()
            
            geo_predictor = PanoGeoPredictor()
            height, width, _ = img.shape
            distances, rot_w2c, fx, fy, cx, cy, pers_imgs = geo_predictor(img)
            pts = pcd_from_depths(img, distances, height, width, args.source_path)
            print('Saving data for future use...')
            save_data(args.source_path, img, distances, rot_w2c, fx, fy, cx, cy, pers_imgs, pts)
            scene_info = get_info_from_params(args.source_path, img, distances, rot_w2c, fx, fy, cx, cy, pers_imgs, pts)

        else:
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            #with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.perturbation_cameras_stage1)  ###
            random.shuffle(scene_info.perturbation_cameras_stage2)  ###
            random.shuffle(scene_info.perturbation_cameras_stage3)  ###

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Loading Perturbation Cameras") ###
            self.perturbation_cameras_stage1[resolution_scale] = cameraList_from_camInfos(scene_info.perturbation_cameras_stage1, resolution_scale, args)
            self.perturbation_cameras_stage2[resolution_scale] = cameraList_from_camInfos(scene_info.perturbation_cameras_stage2, resolution_scale, args)
            self.perturbation_cameras_stage3[resolution_scale] = cameraList_from_camInfos(scene_info.perturbation_cameras_stage3, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPerturbationCameras(self, stage, scale=1.0): ###
        if stage == 1:
            return self.perturbation_cameras_stage1[scale]
        elif stage == 2:
            return self.perturbation_cameras_stage2[scale]
        elif stage == 3:
            return self.perturbation_cameras_stage3[scale]