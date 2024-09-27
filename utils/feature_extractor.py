#from transformers import AutoImageProcessor, Dinov2Model
import torch
from datasets import load_dataset
from torchvision.transforms import Compose
from torchvision import transforms


model =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()

def get_Feature_from_DinoV2(tensor, model = model):
    transform = Compose([
        transforms.Resize(504, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



    trans_img = transform(tensor).unsqueeze(0)
    # feature = model.get_intermediate_layers(trans_img)
    feature = model(trans_img)

    # print(trans_img)
    # print(trans_img.shape)
    # print(feature)
    # print(feature[0].shape)
    return feature




