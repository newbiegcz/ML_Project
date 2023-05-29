from third_party.segment_anything.build_sam import sam_model_registry
from third_party.segment_anything.predictor import SamPredictor
from model_module import pretrained_checkpoints
from modeling.build_sam import build_pretrained_encoder
import torch

initialized = False
predictor = None

def initialize(model_type):
    global initialized
    global predictor
    if not initialized:
        initialized = True
        model = sam_model_registry[model_type](pretrained_checkpoints[model_type]).cuda()
        model.image_encoder = build_pretrained_encoder(model_type, eval=True).cuda()
        predictor = SamPredictor(model)

import numpy as np

@torch.no_grad()
def get_image_embedding(torch_image, raw=True, encoder=None):
    if encoder is None:
        encoder = predictor.model.image_encoder
    assert initialized
    a = (torch_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    from data.dataset import PreprocessForModel
    if raw: 
        __tmp = (torch_image.cpu() -PreprocessForModel.pixel_mean) / PreprocessForModel.pixel_std
    else :
        __tmp = torch_image.cpu()
    print("my_predictor_scale: ", __tmp.min(), __tmp.max())
    print("my_100_100", __tmp[0][100][100])
    __predictor_input_image = predictor.set_image(a)
    return encoder(__tmp.unsqueeze(0).to("cuda:0")).cpu()
    return_features = predictor.model.image_encoder(__predictor_input_image.to("cuda:0")).cpu()

    print(return_features.shape, predictor.get_image_embedding().shape)

    return return_features
    
    return predictor.get_image_embedding()