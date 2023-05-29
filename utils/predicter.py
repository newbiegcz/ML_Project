# Warning: this code has not been tested yet.

from model.sam import SamWithLabel, build_sam_with_label_vit_h
from third_party.segment_anything.utils.transforms import ResizeLongestSide
import torch
import numpy as np
from typing import List

class predicter():
    """
    predict the labels of a single CT data, using grid points as the prompt.
    """

    def __init__(self, sam_model : SamWithLabel, n_grid_points : int):
        """
        Arguments:
        sam_model is a SamWithLabel model;
        n_grid_points is the number of grid points in the prompt per side (n_grid_points^2 points in total)
        """
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.n_grid_points = n_grid_points

    def predict(self, images : List[np.ndarray]) -> List[np.ndarray]:
        """
        Arguments:
        images is a list of np.ndarray, representing a single CT data, each element is a 2D image slice
        each element in the list has shape (H, W, C), where H, W, C is the height, width and channel of the images
        images are in HWC uint8 format, with pixel values in [0, 255]

        Returns:
        a list of np.ndarray, each element is a 2D mask, with shape (H, W).
        """

        # generate grid points
        x_points = np.linspace(0, self.model.image_encoder.img_size - 1, self.n_grid_points)
        y_points = np.linspace(0, self.model.image_encoder.img_size - 1, self.n_grid_points)
        grid_points = np.stack(np.meshgrid(x_points, y_points), axis=-1).reshape(-1, 2) # (n_grid_points^2, 2)
        grid_points = np.expand_dims(grid_points, axis=1) # (n_grid_points^2, 1, 2)


        # generate a list of dicts
        batched_input = []
        for image in images:
            original_size = image.shape[:2]
            image = self.transform.apply_image(image)
            batched_input.append({
                "image": torch.from_numpy(image).permute(2, 0, 1).to(device=self.model.device),
                "original_size": original_size,
                "point_coords": torch.from_numpy(grid_points).to(device=self.model.device),
                "point_labels": torch.ones((grid_points.shape[0], 1)).to(device=self.model.device)
            })

        # predict
        with torch.no_grad():
            outputs = self.model(batched_input, multimask_output=False)

        # collect results
        results = []
        for output in outputs:
            masks = output["masks"].detach().cpu().numpy() # (n_grid_points^2, H, W)
            label_predictions = output["label_predictions"].detach().cpu().numpy() # (n_grid_points^2, 14)
            result = np.zeros(masks.shape[1:], dtype=np.uint8) # (H, W)
            # naive way to collect results, should be optimized
            for mask, label_prediction in zip(masks, label_predictions):
                result[mask.astype(np.bool)] = np.argmax(label_prediction)
            results.append(result)

        return results
