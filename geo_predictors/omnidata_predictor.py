import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image

from .geo_predictor import GeoPredictor
from .omnidata.modules.midas.dpt_depth import DPTDepthModel

class OmnidataPredictor(GeoPredictor):
    def __init__(self):
        super().__init__()
        self.img_size = 512 ### 384 sz: try 512
        ckpt_path = 'pre_checkpoints/omnidata_dpt_depth_v2.ckpt'
        self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=1)
        self.model.to(torch.device('cpu'))
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.trans_totensor = transforms.Compose([transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
                                                  transforms.CenterCrop(self.img_size),
                                                  transforms.Normalize(mean=0.5, std=0.5)])

    def predict_depth(self, img, **kwargs):
        self.model.to(torch.device('cuda'))
        img_tensor = self.trans_totensor(img)
        output = self.model(img_tensor).clip(0., 1.)
        self.model.to(torch.device('cpu'))
        output = output.clip(0., 1.)
        return output[:, None]