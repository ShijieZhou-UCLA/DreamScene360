import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import shutil
import os
import trimesh
import torch


def save_data(source_path, pano_img, distances, R, fx, fy, cx, cy, pers_imgs, pts):
    image_folder = os.path.join(source_path, 'images')
    #depth_folder = os.path.join(source_path, 'depths')
    pose_folder = os.path.join(source_path, 'sparse/0')

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # if not os.path.exists(depth_folder):
    #     os.makedirs(depth_folder)
    
    if not os.path.exists(pose_folder):
        os.makedirs(pose_folder)

    # Loop over each image in the array
    for i in range(pers_imgs.shape[0]):
        img = pers_imgs[i].permute(1, 2, 0).cpu().numpy()  # Change shape to (384, 384, 3)
        # Update the file path to include the folder name
        plt.imsave(os.path.join(image_folder, f'image_{i+1}.png'), img)
        # depth = pers_depths[i].transpose(1, 2, 0)  # Change shape to (384, 384, 1)
        # depth = np.squeeze(depth)
        # # Update the file path to include the folder name
        # plt.imsave(os.path.join(depth_folder, f'image_{i+1}.png'), depth)

    

    # Placeholder values for image width and height
    width, height = pers_imgs.shape[2], pers_imgs.shape[3] #384, 384

    # Write cameras.txt
    cameras_file_path = os.path.join(pose_folder, 'cameras.txt')
    with open(cameras_file_path, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write(f'# Number of cameras: {len(cx)}\n')
        for i in range(len(cx)):
            f.write(f'{i + 1} PINHOLE {width} {height} {fx[i][0]} {fy[i][0]} {cx[i][0]} {cy[i][0]}\n')

    # Write images.txt
    images_file_path = os.path.join(pose_folder, 'images.txt')
    with open(images_file_path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write(f'# Number of images: {len(R)}, mean observations per image: TBD\n')
        for i, R_mat in enumerate(R):
            # Convert rotation matrix to quaternion
            quaternion = Rotation.from_matrix(R_mat.cpu().numpy()).as_quat()  # [x, y, z, w]
            # Assuming placeholders for TX, TY, TZ
            tx, ty, tz = 0, 0, 0  # Replace these with your actual data if available
            # Write to images.txt
            f.write(f'{i + 1} {quaternion[3]} {quaternion[0]} {quaternion[1]} {quaternion[2]} {tx} {ty} {tz} {i + 1} image_{i+1}.png\n')
            f.write('\n')  # Assuming no keypoints are provided, leave this line empty


    # Define the path for points3D.txt
    points3D_file_path = os.path.join(pose_folder, 'points3D.txt')

    # Data to be written to points3D.txt
    points3D_data = """# 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    # Number of points: 3, mean track length: 3.3334
    63390 1.67241 0.292931 0.609726 115 121 122 1.33927 16 6542 15 7345 6 6714 14 7227
    63376 2.01848 0.108877 -0.0260841 102 209 250 1.73449 16 6519 15 7322 14 7212 8 3991
    63371 1.71102 0.28566 0.53475 245 251 249 0.612829 118 4140 117 4473
    """

    # Write the data to points3D.txt (fake, for colmap format only)
    with open(points3D_file_path, 'w') as file:
        file.write(points3D_data)

    pcd = trimesh.PointCloud(pts, pano_img.reshape(-1, 3).cpu().numpy())
    pcd_path = os.path.join(pose_folder, 'points3D.ply')
    pcd.export(pcd_path)

    print('Saved perspective images to ', image_folder)
    print('Saved camera poses to ', pose_folder)