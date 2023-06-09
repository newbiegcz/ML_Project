import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric

def evaluate(predicter, data_iter):
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    for batch in data_iter:
        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
        val_outputs = predicter(val_inputs)
        val_labels_list = decollate_batch(val_labels)
        val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
        val_outputs_list = decollate_batch(val_outputs)
        val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
        
    return mean_dice_val
