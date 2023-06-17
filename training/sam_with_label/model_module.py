import lightning.pytorch as pl
from third_party.segment_anything.build_sam import build_sam_vit_h
from modeling.build_sam import sam_with_label_model_registry
from third_party.segment_anything.build_sam import sam_model_registry
import torch
import wandb
import torch.optim as optim
import torch.nn as nn
from .losses import SegmentationLoss
from utils.visualize import default_label_names
from typing import List
# import utils.visualize as visualize
from modeling.build_sam import pretrained_checkpoints
from third_party.warmup_scheduler.scheduler import GradualWarmupScheduler

class DiceMetric():
    def __init__(self):
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.sum_dice = torch.zeros((14, ), dtype=torch.float32)
        self.n_samples = torch.zeros((14, ), dtype=torch.float32)
    
    @torch.no_grad()
    def update(self, pred_binary_mask, binary_label, mask_cls):
        B = pred_binary_mask.shape[0]
        pred_binary_mask = pred_binary_mask.view(B, -1)
        binary_label = binary_label.view(B, -1)
        TP = torch.count_nonzero(pred_binary_mask & binary_label, dim=1)
        FP = torch.count_nonzero(pred_binary_mask & (~binary_label), dim=1)
        FN = torch.count_nonzero((~pred_binary_mask) & (binary_label), dim=1)
        assert TP.shape[0] == B
        dice = ((2 * TP) / (2 * TP + FP + FN)).cpu()
        mask_cls = mask_cls.cpu().to(torch.int)
        for i in range(B):
            self.n_samples[mask_cls[i]] += 1
            self.sum_dice[mask_cls[i]] += dice[i]

    @torch.no_grad()
    def get_metrics(self, ignore_background=True):
        avg_dice = self.sum_dice / self.n_samples
        if ignore_background:
            avg_dice[0] = torch.nan
        mdice = torch.nanmean(avg_dice)
        return mdice, avg_dice
    
class LabelMetric():
    def __init__(self, label_weights):
        self.reset()
        self.label_weights = label_weights.cpu()

    @torch.no_grad()
    def reset(self):
        self.pd_labels = []
        self.gt_labels = []
    
    @torch.no_grad()
    def update(self, pd_label, gt_label):
        self.pd_labels.append(pd_label.cpu())
        self.gt_labels.append(gt_label.cpu())

    @torch.no_grad()
    def get_metrics(self):
        # both accuracy and confusion matrix are computed on the whole dataset
        pd_labels = torch.cat(self.pd_labels)
        gt_labels = torch.cat(self.gt_labels)
        # print(gt_labels.shape, gt_labels.dtype, self.label_weights.shape)
        acc = torch.sum(pd_labels == gt_labels).float() / pd_labels.shape[0]
        weight = self.label_weights[gt_labels]
        weighted_acc = torch.sum((pd_labels == gt_labels).float() * weight) / torch.sum(weight)
        confusion_matrix = wandb.plot.confusion_matrix(
            y_true=gt_labels.numpy(),
            preds=pd_labels.numpy(),
            class_names=default_label_names,
            title="Confusion Matrix",
        )
        return acc, weighted_acc, confusion_matrix
        
    
def iou_func(pred_binary_mask, binary_label):
    B = pred_binary_mask.shape[0]
    pred_binary_mask = pred_binary_mask.view(B, -1)
    binary_label = binary_label.view(B, -1)
    intersection = torch.count_nonzero(pred_binary_mask & binary_label, dim=1)
    union = torch.count_nonzero(pred_binary_mask | (binary_label), dim=1)
    return intersection / union

def ious_func(pred_binary_masks, binary_label):
        B = pred_binary_masks.shape[0]
        pred_binary_masks = pred_binary_masks.view(B, 14, -1)
        binary_label = binary_label.view(B, 1, -1)
        intersection = torch.count_nonzero(pred_binary_masks & binary_label, dim=2)
        union = torch.count_nonzero(pred_binary_masks | (binary_label), dim=2)
        return intersection / union

class SAMWithLabelModule(pl.LightningModule):
    def __init__(self, 
                 model_type: str = "vit_h",
                 train_image_encoder: bool = False,
                 train_prompt_encoder: bool = True,
                 dice_loss_coef: float = 1.0,
                 focal_loss_coef: float = 1.0,
                 label_loss_coef: float = 1.0,
                 iou_loss_coef: float = 1.0,
                 label_weight: List[float] = [1.0] * 14,
                 optimizer_type: str = "AdamW",
                 model_kwargs: dict = {},
                 prompt_3d_std: float = 0.0,
                 optimizer_kwargs: dict = {
                    "lr": 1e-5,
                    "weight_decay": 0.1,
                 },
                 dice_loss_params: dict = {
                     "p": 1.0,
                     "smooth": 1.0,
                 },
                 focal_loss_params: dict = {
                     "alpha": 0.25,
                     "gamma": 2.0,
                 },
                 debug: bool = False):
        '''
        Args:
            model_type: str, model type, one of ["vit_h", "vit_b", "vit_tiny", "vit_small", "vit_base"]
            train_image_encoder: bool, whether to train image encoder
            train_prompt_encoder: bool, whether to train prompt encoder
            dice_loss_coef: float, dice loss coefficient
            focal_loss_coef: float, focal loss coefficient
            label_loss_coef: float, label loss coefficient
            iou_loss_coef: float, iou loss coefficient
            optimizer_type: str, optimizer type, one of ["AdamW"]
            optimizer_kwargs: dict, optimizer kwargs
        '''
        
        super().__init__()
        self.save_hyperparameters()

        self.prompt_3d_std = prompt_3d_std
        self.pretrained_checkpoint = pretrained_checkpoints[model_type]
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.train_image_encoder = train_image_encoder
        self.train_prompt_encoder = train_prompt_encoder
        self.dice_loss_coef = dice_loss_coef
        self.focal_loss_coef = focal_loss_coef
        self.label_loss_coef = label_loss_coef
        self.iou_loss_coef = iou_loss_coef
        _label_weight = torch.tensor(label_weight, dtype=torch.float32, device=self.device)
        _label_weight = _label_weight / _label_weight.sum() * 14
        self.register_buffer("label_weight", _label_weight, False)
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.debug = debug
        
        self.model, self.encoder_builder  = sam_with_label_model_registry[model_type](None, train_image_encoder, **model_kwargs)
        pretrained_sam = sam_model_registry[model_type](self.pretrained_checkpoint)
        pretrained_state_dict = pretrained_sam.state_dict()
        new_state_dict = self.model.state_dict()
        for k, v in pretrained_state_dict.items():
            if not train_image_encoder and k.startswith("image_encoder"): 
                continue
            assert k in new_state_dict.keys()
            if v.shape != new_state_dict[k].shape:
                assert v.shape[0] == 4 and new_state_dict[k].shape[0] == 14
                assert v.shape[1:] == new_state_dict[k].shape[1:]
                new_state_dict[k] = v[:1].repeat(*((14,) + (1,) * (len(v.shape) - 1)))
            else :
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.mask_decoder.copy_weights()

        self.segmentation_loss = SegmentationLoss(
            dice_loss_coef=dice_loss_coef,
            focal_loss_coef=focal_loss_coef,
            dice_loss_params=dice_loss_params,
            focal_loss_params=focal_loss_params,
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none").to(self.device)
        self.training_dice_metric = DiceMetric()
        self.validation_dice_metric = DiceMetric()
        self.training_label_metric = LabelMetric(self.label_weight)
        self.validation_label_metric = LabelMetric(self.label_weight)

        if self.debug:
            visualize.initialize_window()

    def get_logits(self, batch, mode):
        # 未来如果实现训练 encoder，一定要记得判断 torch.is_grad_enabled()，避免 validate 时保留梯度
        assert not self.train_image_encoder, "Unimplemented"
        assert mode in ["training", "validation"]

        # if self.debug:
        #     import debug
        #     print("\ndebug output: (%s)" % self.model_type)
        #     debug.initialize(self.model_type)
        #     image_embedding = debug.get_image_embedding(batch['image'][0].detach()).cuda()
        #     print(image_embedding[0].shape, "-->", batch['embedding'][0].shape)
        #     print("diff", ((batch['embedding'][0] - image_embedding[0]).max()))
            
        #     batch['embedding'][0] = image_embedding[0]

        B = len(batch['embedding'])

        embeddings = batch['embedding']

        batch = batch.copy()
        batch['prompt'] = torch.cat((batch['prompt'][0].reshape(-1, 1), batch['prompt'][1].reshape(-1, 1)), dim=1)

        point_coords = batch['prompt'][:, None, :]
        point_labels = torch.ones((B, 1), dtype=torch.int, device=self.device)
        prompt_3ds = torch.stack(batch['3d']).permute(1, 0).to(torch.float32)

        if mode == "training":
            prompt_3ds = prompt_3ds + self.prompt_3d_std * torch.randn_like(prompt_3ds)

        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_prompt_encoder):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                                    points=(point_coords, point_labels),
                                    boxes=None,
                                    masks=None,
                                    prompt_3ds=prompt_3ds
                                )
            
        batch_masks, batch_ious, batch_label = self.model.mask_decoder(
            image_embeddings=embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            already_unfolded=True,
        )
        assert batch_masks.shape[1] == 14

        return batch_masks, batch_ious, batch_label
    
    def get_loss_and_update_metric(self, batch, batch_masks, batch_ious, batch_label, dice_metric, label_metric, batch_name):
        B = len(batch['embedding'])

        segmentation_loss = 0.0
        iou_loss = 0.0
        label_loss = 0.0
        
        img_size = 1024

        mask_cls = batch['mask_cls']
        lowres_mask = torch.nn.functional.interpolate(
            (batch['connected_mask']).to(torch.float32),
            (img_size // 4, img_size // 4),
            mode="nearest-exact"
        )
        binary_label = lowres_mask[:, 0]
        
        is_foreground_label = mask_cls != 0
        batch_mask = batch_masks[torch.arange(B), mask_cls]
        segmentation_loss, _dice_loss, _focal_loss = self.segmentation_loss(batch_mask[is_foreground_label], binary_label[is_foreground_label])

        with torch.no_grad():
            pred_binary_masks = (batch_masks > self.model.mask_threshold).to(torch.long)
            
            binary_label = binary_label.to(torch.long)
            
            ious = ious_func(pred_binary_masks, binary_label)
            ious[~is_foreground_label].fill_(0.0) # TODO: 这样做是否合适?
            assert (ious.shape[0] == B)

            dice_metric.update(pred_binary_masks[torch.arange(B), mask_cls], binary_label, mask_cls)
        
        iou_loss = ((ious - batch_ious) ** 2).mean()
        label_loss = (self.cross_entropy_loss(batch_label, mask_cls.to(torch.long)) * self.label_weight[mask_cls]).mean()
        
        label_metric.update(batch_label.argmax(dim=1), mask_cls)
        # if self.debug:
        #     if batch_name is None:
        #         batch_name = "unnamed"

        #     pd_output_label = torch.nn.functional.interpolate(
        #         pred_binary_mask[0:1, None, :, :].to(torch.float32),
        #         (1024, 1024),
        #         mode="nearest-exact"
        #     )[0].cpu().numpy()

        #     gt_output_label = torch.nn.functional.interpolate(
        #         binary_label[0:1, None, :, :].to(torch.float32),
        #         (1024, 1024),
        #         mode="nearest-exact"
        #     )[0].cpu().numpy()

        #     prompt = batch['prompt'][0].detach().cpu().numpy()
        #     visualize.add_object_2d(batch_name + "_img0",
        #             image=batch['image'][0].detach().cpu().numpy(),
        #             pd_label=pd_output_label,
        #             gt_label=gt_output_label,
        #             prompt_points=[(prompt, 1)],
        #             label_name=["negative", "positive"],
        #                 extras={
        #                 "prompt": prompt,
        #                 "prompt_label": default_label_names[mask_cls[0].cpu().to(torch.int)]
        #             }
        #     )
        #     input("")

        return segmentation_loss, iou_loss, label_loss, _dice_loss, _focal_loss


    def training_step(self, batch, batch_idx):
        batch_masks, batch_ious, batch_label = self.get_logits(batch, "training")

        segmentation_loss, iou_loss, label_loss, _dice_loss, _focal_loss = self.get_loss_and_update_metric(batch, batch_masks, batch_ious, batch_label, self.training_dice_metric, self.training_label_metric, "train_%d" % batch_idx)

        loss = self.iou_loss_coef * iou_loss + self.label_loss_coef * label_loss + segmentation_loss

        self.log("train_loss/total_loss", loss)
        self.log("train_loss/label_loss", label_loss)
        self.log("train_loss/iou_loss", iou_loss)
        self.log("train_loss/segmentation_loss", segmentation_loss)
        self.log("train_loss/dice_loss", _dice_loss)
        self.log("train_loss/focal_loss", _focal_loss)

        mdice, avg_dice = self.training_dice_metric.get_metrics()
        self.log("train_Dice/mDice", mdice)
        for i in range(14):
            self.log("train_Dice/%s" % default_label_names[i], avg_dice[i])
        
        
        return loss
    
    def on_train_epoch_end(self):
        acc, weighted_acc, confusion_matric = self.training_label_metric.get_metrics()
        self.logger.experiment.log(
            {
                "train_Label/acc": acc,
                "train_Label/weighted_acc": weighted_acc,
                "train_Label/confusion_matrix": confusion_matric
            }
        )
        self.training_dice_metric.reset()
        self.training_label_metric.reset()
    
    def validation_step(self, batch, batch_idx):
        batch_masks, batch_ious, batch_label = self.get_logits(batch, "validation")

        segmentation_loss, iou_loss, label_loss, _dice_loss, _focal_loss = self.get_loss_and_update_metric(batch, batch_masks, batch_ious, batch_label, self.validation_dice_metric, self.validation_label_metric, "val_%d" % batch_idx)

        loss = self.iou_loss_coef * iou_loss + self.label_loss_coef * label_loss + segmentation_loss

        self.log("val_loss/total_loss", loss)
        self.log("val_loss/label_loss", label_loss)
        self.log("val_loss/iou_loss", iou_loss)
        self.log("val_loss/segmentation_loss", segmentation_loss)
        self.log("val_loss/dice_loss", _dice_loss)
        self.log("val_loss/focal_loss", _focal_loss)
        return loss
    
    def on_validation_epoch_end(self):
        mdice, avg_dice = self.validation_dice_metric.get_metrics()
        self.log("val_Dice/mDice", mdice)
        for i in range(14):
            self.log("val_Dice/%s" % default_label_names[i], avg_dice[i])
        acc, weighted_acc, confusion_matric = self.validation_label_metric.get_metrics()
        self.logger.experiment.log(
            {
                "val_Label/acc": acc,
                "val_Label/weighted_acc": weighted_acc,
                "val_Label/confusion_matrix": confusion_matric
            }
        )
        self.validation_dice_metric.reset()
        self.validation_label_metric.reset()

    def configure_optimizers(self):
        assert self.optimizer_type in ["AdamW"], "Unimplemented"

        if self.optimizer_type == "AdamW":   
            optimizer = optim.AdamW(self.parameters(), **self.optimizer_kwargs)
        
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45], gamma=0.1)

        return [optimizer], [scheduler]