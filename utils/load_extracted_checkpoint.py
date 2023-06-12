import torch
from modeling.build_sam import sam_with_label_model_registry, build_pretrained_encoder

def load_extracted_checkpoint(model, checkpoint_path):
    sam_with_label, _ = sam_with_label_model_registry['vit_h'](checkpoint="extracted.pth", build_encoder=False)
    del sam_with_label.image_encoder
    sam_with_label.add_module("image_encoder", build_pretrained_encoder("vit_h"))
    return sam_with_label