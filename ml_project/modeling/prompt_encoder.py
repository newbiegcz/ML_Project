from ..utils.position_embedding_3d import PositionalEncoding3D

from ..third_party.segment_anything.modeling.prompt_encoder import PromptEncoder
from typing import Optional, Tuple, Type
import torch
import torch.nn as nn

class Prompt3DEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(embed_dim=embed_dim,
                         image_embedding_size=image_embedding_size,
                         input_image_size=input_image_size,
                         mask_in_chans=mask_in_chans,
                         activation=activation)
        self.embedding_3d = nn.Embedding(1, embed_dim)
        self.position_embedding_3d = PositionalEncoding3D(embed_dim)

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        prompt_3ds: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif prompt_3ds is not None:
            return prompt_3ds.shape[0]
        else:
            return 1
        
    def _embed_prompt_3ds(self, prompt_3ds: torch.Tensor) -> torch.Tensor:
        return (self.position_embedding_3d(prompt_3ds) + self.embedding_3d.weight)[:, None, :]

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        prompt_3ds: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self._get_batch_size(points, boxes, masks, prompt_3ds)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if prompt_3ds is not None:
            prompt_3d_embeddings = self._embed_prompt_3ds(prompt_3ds)
            sparse_embeddings = torch.cat([sparse_embeddings, prompt_3d_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings