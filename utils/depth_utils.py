import torch
import sys
import os
from torchvision import transforms

import PIL
from PIL import Image

sys.path.append("geo_predictors")
from omnidata.modules.midas.dpt_depth import DPTDepthModel

downsampling = 1
img_size = 512
ckpt_path = 'pre_checkpoints/omnidata_dpt_depth_v2.ckpt'
model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=1)
model.to(torch.device('cpu'))
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[6:]] = v
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
trans_totensor = transforms.Compose([transforms.Normalize(mean=0.5, std=0.5)])

def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]
    img = img.unsqueeze(0)
    model.to(torch.device('cuda'))
    img_tensor = trans_totensor(img)
    if mode == 'test':
        with torch.no_grad():
            prediction = model(img_tensor)
            prediction = prediction.squeeze()
    else:
        prediction = model(img_tensor).squeeze()
    return prediction
