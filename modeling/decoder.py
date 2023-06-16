from functools import partial
import torch
from torch import nn
from torch.nn import functional as F

from typing import Callable, List, Optional, Tuple, Type

from third_party.segment_anything.modeling.common import LayerNorm2d
from third_party.segment_anything.modeling.mask_decoder import MaskDecoder
from torch import Tensor

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=True,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class Bottleneck(nn.Module):
    
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(torch.nn.LayerNorm, eps=1e-6)
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MaskLabelDecoder(MaskDecoder):
    def __init__(
        self,
        *,
        label_head_depth,
        label_head_hidden_dim=256,
        **kwargs,
    ) -> None:
        assert "num_multimask_outputs" not in kwargs, "num_multimask_outputs is not supported"
        super().__init__(num_multimask_outputs=14 - 1, **kwargs) # 会被 + 1
        del self.num_multimask_outputs # 避免混淆

        self.label_head_depth = label_head_depth
        self.label_head_depth_dim = label_head_hidden_dim

        # TODO: 尝试合并 label head 的 MLP
        # TODO: 试着调整全连接层 hidden 大小
        # TODO: 试着使用原本的多个 output token
        # TODO: 确定只用一个 token 输出 label 的结果..? 

        self.label_token = nn.Embedding(1, self.transformer_dim)
        self.label_prediction_head = MLP(
            self.transformer_dim, label_head_hidden_dim, 14, label_head_depth
        )

        self.bottlenecks = nn.Sequential(
            Bottleneck(self.transformer_dim, self.transformer_dim // 4),
            Bottleneck(self.transformer_dim, self.transformer_dim // 4),
            Bottleneck(self.transformer_dim, self.transformer_dim // 4),
            Bottleneck(self.transformer_dim, self.transformer_dim // 4),
        )

    def copy_weights(self):
        # shape 不匹配已经处理过了
        # 这里只需要复制多次重复了的 submodule
        for i in range(1, self.num_mask_tokens):
            old_state_dict = self.output_hypernetworks_mlps[0].state_dict()
            for key in old_state_dict.keys():
                old_state_dict[key] = old_state_dict[key].clone()
            self.output_hypernetworks_mlps[i].load_state_dict(old_state_dict)
        
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        already_unfolded = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_embeddings = image_embeddings.permute(0, 3, 1, 2)
        image_embeddings = self.bottlenecks(image_embeddings)
        image_embeddings = image_embeddings.permute(0, 2, 3, 1)

        masks, iou_pred, label_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            already_unfolded=already_unfolded
        )

        masks = masks[:, :, :, :]
        iou_pred = iou_pred[:, :]
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
