from functools import partial
from .sam import SamWithLabel
import torch

from third_party.segment_anything.modeling.image_encoder import ImageEncoderViT
from third_party.segment_anything.modeling.transformer import TwoWayTransformer
from third_party.segment_anything.modeling.prompt_encoder import PromptEncoder
from .decoder import MaskLabelDecoder

def _build_sam_with_label(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    build_encoder=True,
):
    '''
    Build a SAM model with label head.
    
    Args:
        encoder_embed_dim: Embedding dimension of the encoder.
        encoder_depth: Depth of the encoder.
        encoder_num_heads: Number of heads of the encoder.
        encoder_global_attn_indexes: Indexes of the global attention layers of the encoder.
        checkpoint: Path to the checkpoint of the encoder.
        build_encoder: Whether to build the encoder.

    Returns:
        A tuple of (SAM model, encoder_builder). The encoder_builder is a function that can be used to build the encoder.
    '''
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    encoder_builder = partial(ImageEncoderViT, 
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    sam = SamWithLabel(
        image_encoder=encoder_builder() if build_encoder else None,
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskLabelDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            label_head_depth=3,
            label_head_hidden_dim=256,
        ), 
        #TODO : hydra
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam, encoder_builder

def build_sam_with_label_vit_h(checkpoint=None, build_encoder=True):
    '''
    Build a SAM model with label head, using ViT-H as the encoder.
    
    Args:
        checkpoint: Path to the checkpoint of the encoder.
        build_encoder: Whether to build the encoder.
        
    Returns:
        A tuple of (SAM model, encoder_builder). The encoder_builder is a function that can be used to build the encoder.
    '''
    return _build_sam_with_label(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        build_encoder=build_encoder,
    )