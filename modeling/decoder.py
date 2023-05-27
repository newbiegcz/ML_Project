import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from third_party.segment_anything.modeling.common import LayerNorm2d
from third_party.segment_anything.modeling.mask_decoder import MaskDecoder


class MaskLabelDecoder(MaskDecoder):
    def __init__(
        self,
        *,
        label_head_depth,
        label_head_hidden_dim,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.label_head_depth = label_head_depth
        self.label_head_depth_dim = label_head_hidden_dim

        self.label_token = nn.Embedding(1, self.transformer_dim)
        # TODO: 这样写太 naive 了.. 记得改掉 
        self.label_prediction_head = MLP(
            self.transformer_dim, label_head_hidden_dim, 14, label_head_depth
        )
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        already_unfolded = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks, iou_pred, label_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            already_unfolded=already_unfolded
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        label_pred = label_pred.reshape(
                label_pred.shape[0], 14
            )

        # Prepare output
        return masks, iou_pred, label_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        already_unfolded = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.label_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if not already_unfolded:
            # Expand per-image data in batch direction to be per-mask
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
            src = src + dense_prompt_embeddings
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        else :
            src = image_embeddings
            src = src + dense_prompt_embeddings
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) # TODO: 可优化吗?
        
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        label_token_out = hs[:, 1, :]
        mask_tokens_out = hs[:, 2 : (2 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        label_pred = self.label_prediction_head(label_token_out)

        return masks, iou_pred, label_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
