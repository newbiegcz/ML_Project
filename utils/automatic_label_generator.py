# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from modeling.sam import SamWithLabel
from modeling.predictor import SamWithLabelPredictor
from third_party.segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamAutomaticLabelGenerator():
    def __init__(
        self,
        model: SamWithLabel,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        # pred_iou_thresh: float = torch.tensor([0,0.8,0.8,0.8,0.8,0.4,0.8,0.7,0.8,0.8,0.5,0.6,0.6,0.4]),
        pred_iou_thresh: float = torch.tensor([0.4]*14),
        label_certainty_thresh: float = 0,
        #stability_score_thresh: float = 0.50,
        stability_score_thresh: float = 0.0,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          label_certainty_thresh (float): A filtering threshold in [0,1], using
            the maximum predicted label probability to filter masks.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamWithLabelPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = torch.tensor(pred_iou_thresh, device=self.predictor.device)
        self.label_certainty_thresh = label_certainty_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

    @torch.no_grad()
    def generate_labels(self, image : np.ndarray, height : float, verbose=False) -> np.ndarray:
        """
        Generates labels for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format,
          not required to be already transformed to the model's input size.

        Returns:
          np.ndarray : The label of each pixel, with shape (H, W).
        """
        if verbose:
            print('generating labels...')
        masks = self.generate(image, height)
        labels = np.zeros(image.shape[:2], dtype=np.uint8)
        vis = [np.zeros(image.shape[:2], dtype=np.uint8)] * 14

        idxs = list(range(len(masks)))
        idxs.sort(key=lambda x : masks[x]["predicted_iou"], reverse=True)

        #tmp = []
        #vis = {}
        for idx in idxs:
            mask = masks[idx]["segmentation"]
            label_pred = masks[idx]["label"]
            label = np.array(label_pred).argmax()
            # grid_labels_mask = self.grid_labels == label
            if np.sum(np.minimum(vis[label],mask)) > 0:
                continue
            vis[label] += mask
            labels[mask > 0] = label
            #tmp.append((mask, label_pred))
        return labels#, tmp

    @torch.no_grad()
    def generate(self, image: np.ndarray, height : float, verbose=False) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.
          height: the height of the image slice in the whole CT data, normalized to [0,1].

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               label (int): The predicted label of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """
        # Generate masks
        if verbose:
            print('Generating masks...')
        mask_data = self._generate_masks(image, height)

        # Filter small disconnected regions and holes in masks
        if verbose:
            print('Filtering small disconnected regions and holes in masks...')
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )
        if verbose:
            print('After filter small disconnected regions and holes in masks:', len(mask_data["rles"]))

        # Encode masks
        if verbose:
            print('Encoding masks...')
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        if verbose:
            print('Writing mask records...')
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            # print('label prediction:', mask_data["label_preds"][idx])
            mask = mask_data["segmentations"][idx]
            np.savetxt('output.txt', mask, fmt='%d')
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "label": mask_data["label_preds"][idx].tolist(),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns


    def _generate_masks(self, image: np.ndarray, height: float, verbose=False) -> MaskData:
        self.grid_labels = np.ones((image.shape[0], image.shape[1]), dtype=torch.int) * -1
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size, height)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        if verbose:
            print('After filter by nms between crops:', len(data["rles"]))

        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
        height : float,
        verbose=False
    ) -> MaskData:
        # print(image.shape)

        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size, height)
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)
        if verbose:
            print('After filter by nms within crop:', len(data["rles"]))

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        height: float,
        verbose=False
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, dtype=torch.float, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        in_points = in_points[:, None, :] # Bx1x2
        in_labels = in_labels[:, None] # Bx1
        prompt_3ds = torch.cat((in_points[:, :, 1:2] / im_size[1], in_points[:, :, 0:1] / im_size[0]
                                , torch.tensor([[[height]] * 1] * in_points.shape[0], device=self.predictor.device)), dim=2) # Bx1x3
        prompt_3ds = prompt_3ds[:, 0, :].type(torch.float) # Bx3

        #print('in_points.shape:', in_points.shape)
        #print('in_labels.shape:', in_labels.shape)
        #print('prompt_3ds.shape:', prompt_3ds.shape)
        masks, iou_preds, label_preds, _ = self.predictor.predict_torch(
            in_points,
            in_labels,
            prompt_3ds = prompt_3ds,
            return_logits=True,
        )

        for i in range(masks.shape[0]):
            self.grid_labels[in_points[i][0][0].numpy(), in_points[i][0][1].numpy()] = label_preds[i].argmax().numpy()

        # Serialize predictions and store in MaskData
        #print(masks.shape) #64x1xHxW
        #print(iou_preds.shape) #64x1
        #print(label_preds.shape) #64x14
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            label_preds=label_preds,
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        if verbose:
            print('Before filter:', len(data["masks"]))

        # remove masks with predicted label = 0
        keep_mask = data["label_preds"].argmax(dim = 1) > 0
        #print('keep_mask shape:', keep_mask.shape)
        data.filter(keep_mask)
        if verbose:
            print('After filter by label prediction:', len(data["masks"]))

        if self.label_certainty_thresh > 0.0:
            prob = torch.nn.functional.softmax(data["label_preds"], dim=1)
            label = data["label_preds"].argmax(dim=1)
            keep_mask = prob[torch.arange(len(label)), label] > self.label_certainty_thresh
            data.filter(keep_mask)

        

        #print('Before filter by IoU:', len(data["masks"]))
        if verbose:
            print('IoU predictions:', data["iou_preds"])

        # Filter by predicted IoU
        if True: #self.pred_iou_thresh > 0.0:
            label = data["label_preds"].argmax(dim=1)
            keep_mask = data["iou_preds"] > self.pred_iou_thresh[label]
            data.filter(keep_mask)

        if verbose:
            print('After filter by IoU:', len(data["masks"]))

        # Calculate stability score
        
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        if verbose:
            print('After filter by stability:', len(data["masks"]))
        

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        if verbose:
            print('After filter by touch crop boundaries:', len(data["masks"]))

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data




    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
