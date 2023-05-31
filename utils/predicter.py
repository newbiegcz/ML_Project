# Warning: this code has not been tested yet.

from model.sam import SamWithLabel
import numpy as np
from typing import List
from automatic_label_generator import SamAutomaticLabelGenerator

class LabelPredicter():
    """
    predict the labels of a single CT data, using grid points as the prompt.
    """

    def __init__(self, sam_model : SamWithLabel):
        """
        Arguments:
        sam_model is a SamWithLabel model;
        """
        self.model = sam_model
        self.automatic_label_generator = SamAutomaticLabelGenerator(self.model)

    def predict(self, images : List[np.ndarray]) -> List[np.ndarray]:
        """
        Arguments:
        images is a list of np.ndarray, representing a single CT data, each element is a 2D image slice
        each element in the list has shape (H, W, C), where H, W, C is the height, width and channel of the images
        images are in HWC uint8 format, with pixel values in [0, 255]

        Returns:
        a list of np.ndarray, each element is a 2D mask, with shape (H, W).
        """
        res = []
        for image in images:
            labels = self.automatic_label_generator.generate_labels(image)
            res.append(labels)
        return res


