# DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting
Shijie Zhou*, Zhiwen Fan*, Dejia Xu*, Haoran Chang, Pradyumna Chari, Tejas Bharadwaj, Suya You, Zhangyang Wang, Achuta Kadambi (* indicates equal contribution)<br>
| [Webpage](https://dreamscene360.github.io/) | [Full Paper](https://arxiv.org/abs/2404.06903) | [Video](https://www.youtube.com/embed/6rMIQfe7b24?si=cm7cZ-T9r5na7YFD) | <br>
![Teaser image](assets/teaser_v6.png)



Abstract: *The increasing demand for virtual reality applications has highlighted the significance of crafting immersive 3D assets. We present a text-to-3D 360 scene generation pipeline that facilitates the creation of comprehensive 360 scenes for in-the-wild environments in a matter of minutes. Our approach utilizes the generative power of a 2D diffusion model and prompt self-refinement to create a high-quality and globally coherent panoramic image. This image acts as a preliminary "flat" (2D) scene representation. Subsequently, it is lifted into 3D Gaussians, employing splatting techniques to enable real-time exploration. To produce consistent 3D geometry, our pipeline constructs a spatially coherent structure by aligning the 2D monocular depth into a globally optimized point cloud. This point cloud serves as the initial state for the centroids of 3D Gaussians. In order to address invisible issues inherent in single-view inputs, we impose semantic and geometric constraints on both synthesized and input camera views as regularizations. These guide the optimization of Gaussians, aiding in the reconstruction of unseen regions. In summary, our method offers a globally consistent 3D scene within a 360 perspective, providing an enhanced immersive experience over existing techniques.*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{zhou2024dreamscene360,
  title={Dreamscene360: Unconstrained text-to-3d scene generation with panoramic gaussian splatting},
  author={Zhou, Shijie and Fan, Zhiwen and Xu, Dejia and Chang, Haoran and Chari, Pradyumna and Bharadwaj, Tejas and You, Suya and Wang, Zhangyang and Kadambi, Achuta},
  booktitle={European Conference on Computer Vision},
  pages={324--342},
  year={2024},
  organization={Springer}
}</code></pre>
  </div>
</section>




# Environment setup
Create Environment:
```shell
conda create --name dreamscene360 python=3.8
conda activate dreamscene360
```

PyTorch (Please check your CUDA version, we used 12.4)
```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Required packages
```shell
pip install -r requirements.txt
```

Submodules

```shell
pip install submodules/diff-gaussian-rasterization-depth # Rasterizer for RGB and depth
pip install submodules/simple-knn
```

# Checkpoints
1. From project home directory, create folder: **pre_checkpoints**
```
mkdir pre_checkpoints
```

2. Download required pretrained model `omnidata_dpt_depth_v2.ckpt` from this [dropbox link](https://www.dropbox.com/scl/fo/348s01x0trt0yxb934cwe/h?rlkey=a96g2incso7g53evzamzo0j0y&dl=0) into **pre_checkpoints**. (Thanks to [PERF](https://github.com/perf-project/PeRF/tree/master/pre_checkpoints) for providing the models)

3. Download required pretrained models for text2pano:
```
cd stitch_diffusion/pretrained_model
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors -O stable-diffusion-2-1-base.safetensors
cd ../vae
wget https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt -O stablediffusion.vae.pt
cd ..
python download_lora.py
cd ..
```

<!-- <location>
|---pre_checkpoints
|   |---<PERF_checkpoints 0>
|   |---<PERF_checkpoints 1>
|   |---...
|---stitch_diffusion
    |---kohya_trainer
        |---cameras.bin
        |---images.bin
        |---points3D.bin
``` -->
# Generate your 3D scenes
### Text-to-3D
To generate your own designed 360&deg; immersive 3D scene from text, simply write your text prompt in a txt file under your data folder, e.g. `data/YOUR_SCENE/YOUR_SCENE_PROMPT.txt`.

```
python train.py -s data/YOUR_SCENE -m output/OUTPUT_NAME --self_refinement --api_key <Your_OpenAI_GPT4V_Key> --num_prompt 2 --max_rounds 2
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>
  
  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --self_refinement
  Enables self refinement during panorama generation
  
  #### --api_key
  Put your OpenAI GPT4V API Key here


  #### --num_prompt
  Specify how many candidate text prompts you would like to try for prompt revision

  #### --max_rounds
  Specify how many rounds of generation & quality assessment you would like to try for each text prompt

  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>

If you don't want to enable self-refinement with GPT-4V, simply exclude all the arguments starting from --self_refinement.

Please feel free to try our provided example at `data/Italy_text`.

### Panorama-to-3D
Our code also supports turning your own 360&deg; panorama image with any resolution into 3D, simply put it into the folder as `data/YOUR_SCENE/YOUR_SCENE_PANORAMA.png`.
```
python train.py -s data/YOUR_SCENE -m output/OUTPUT_NAME
```
Please feel free to try our provided example at `data/alley_pano`.

Additionally, DreamScene360 is adaptable to any text-to-panorama generator, meaning the `stitch_diffusion` module can be replaced by other diffusion models as well.

PS: If fail to compile the CUDA rasterizer, try this:
```
sudo apt-get install libglm-dev
```

# Render perspective views 
Render from training and test views:
```
python render.py -s data/YOUR_SCENE -m output/OUTPUT_NAME  --iteration 9000
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 

  **The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.** 

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --convert_SHs_python
  Flag to make pipeline render with computed SHs from PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline render with computed 3D covariance from PyTorch instead of ours.

</details>

# Interactive Viewer
To view the 360&deg; 3D scene with an interactive viewer:

## Windows
```
cd viewer_windows/bin
SIBR_gaussianViewer_app.exe -m <Path_to_OUTPUT_NAME>
```

## Ubuntu
First install these dependencies
```
# Dependencies
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
# Project setup
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # add -G Ninja to build faster
cmake --build build -j24 --target install
cd ..
```
To launch the viewer:
```
./<SIBR_install_dir>/bin/SIBR_gaussianViewer_app -m <Path_to_OUTPUT_NAME>
```

## Navigation in SIBR Viewer
The SIBR interface provides several methods of navigating the scene. By default, you will be started with an FPS navigator, which you can control with W, A, S, D, Q, E for camera translation and I, K, J, L, U, O for rotation. Alternatively, you may want to use a Trackball-style navigator (select from the floating menu). You can also snap to a camera from the data set with the Snap to button or find the closest camera with Snap to closest. The floating menues also allow you to change the navigation speed. You can use the Scaling Modifier to control the size of the displayed Gaussians, or show the initial point cloud.



## Acknowledgement
Our repo is developed based on [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [PERF](https://github.com/perf-project/PeRF), [idea2img](https://github.com/zyang-ur/Idea2Img) and [StitchDiffusion](https://github.com/littlewhitesea/StitchDiffusion). Many thanks to the authors for opensoucing the codebase.
