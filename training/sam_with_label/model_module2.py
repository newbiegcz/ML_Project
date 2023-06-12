import lightning.pytorch as pl
from .losses import SegmentationLoss
import torch.nn as nn
from .model_module import iou_func
import torch
import torch.optim as optim
import torch.nn as nn
from sam_grad.build_sam import sam_model_registry
from modeling.build_sam import pretrained_checkpoints

class DiceMetric():
    def __init__(self):
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.dice = torch.zeros((1, ), dtype=torch.float32)
        self.n = torch.zeros((1,), dtype=torch.float32)
    
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
            self.dice[0] += dice[i]
            self.n += 1

    @torch.no_grad()
    def get_metrics(self, ignore_background=True):
        avg_dice = self.dice / self.n
        if (avg_dice != avg_dice).any():
            print("Warning: nan in avg_dice")
        mdice = torch.nanmean(avg_dice)
        return mdice, avg_dice

# create pl.LightningModule named SAMWithInteractiveTraining
class SAMWithInteractiveTraining(pl.LightningModule):
    # init
    def __init__(self, 
                 model_type: str = "vit_b",
                 train_image_encoder: bool = False,
                 train_prompt_encoder: bool = True,
                 dice_loss_coef: float = 1.0,
                 focal_loss_coef: float = 1.0,
                 label_loss_coef: float = 1.0,
                 iou_loss_coef: float = 1.0,
                 optimizer_type: str = "AdamW",
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
                 debug: bool = True):
        # init superclass
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()
        self.train_image_encoder = train_image_encoder
        self.train_prompt_encoder = train_prompt_encoder
        self.debug = debug

        # retrieve pretrained model
        self.pretrained_checkpoint = pretrained_checkpoints[model_type]
        self.model = sam_model_registry[model_type](self.pretrained_checkpoint)

        # init losses
        self.segmentation_loss = SegmentationLoss(
            dice_loss_coef=dice_loss_coef,
            focal_loss_coef=focal_loss_coef,
            dice_loss_params=dice_loss_params,
            focal_loss_params=focal_loss_params,
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)
        self.training_dice_metric = DiceMetric()
        self.validation_dice_metric = DiceMetric()
    
    def get_logits(self,
                   batch: dict,
                   point_coords: torch.Tensor,
                   point_labels: torch.Tensor):
        """
        Args:
            batch: dict, batch of data, containing keys ["embedding", "label", "mask_cls", "prompt"]
            point_coords: a Tensor of dimension (B, N, 2)
            point_labels: a Tensor of dimension (B, N)
        Returns:
            batch_mask: a Tensor of dimension (B, H, W)
            batch_iou: a Tensor of dimension (B, H, W)
            batch_label: a Tensor of dimension (B, H, W)
        """
        # image encoder training is not implemented
        assert not self.train_image_encoder, "Unimplemented"

        B = len(batch['embedding'])

        embeddings = batch['embedding']

        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_prompt_encoder):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                                    points=(point_coords, point_labels),
                                    boxes=None,
                                    masks=None,
                                )
            
        batch_masks, batch_ious, batch_label = self.model.mask_decoder(
            image_embeddings=embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            already_unfolded=True,
        )
        assert batch_masks.shape[1] == 1
        batch_mask = batch_masks[:, 0]
        batch_iou = batch_ious[:, 0]

        return batch_mask, batch_iou, batch_label

    def get_loss_and_update_metric(self, batch, batch_mask, batch_iou, metric):
        B = len(batch['embedding'])

        segmentation_loss = 0.0
        iou_loss = 0.0
        
        img_size = 1024

        mask_cls = batch['mask_cls']
        lowres_labels = torch.nn.functional.interpolate(
            batch['label'],
            (img_size // 4, img_size // 4),
            mode="nearest-exact"
        )
        binary_label = (lowres_labels[:, 0] == mask_cls[:, None, None]).to(torch.float)
        
        is_foreground_label = mask_cls != 0
        segmentation_loss, _dice_loss, _focal_loss = self.segmentation_loss(batch_mask[is_foreground_label], binary_label[is_foreground_label])
        
        with torch.no_grad():
            pred_binary_mask = (batch_mask > self.model.mask_threshold).to(torch.long)
            
            binary_label = binary_label.to(torch.long)
            iou = iou_func(pred_binary_mask[is_foreground_label], binary_label[is_foreground_label])
            assert (iou.shape[0] == B)

            metric.update(pred_binary_mask, binary_label, mask_cls)
        
        iou_loss = ((iou - batch_iou) ** 2).mean()

        return segmentation_loss, iou_loss, _dice_loss, _focal_loss

    def training_step(self, batch, batch_idx):
        # calculate binary_label
        img_size = 1024
        mask_cls = batch['mask_cls']
        lowres_labels = torch.nn.functional.interpolate(
            batch['label'],
            (img_size // 4, img_size // 4),
            mode="nearest-exact"
        )
        binary_label = (lowres_labels[:, 0] == mask_cls[:, None, None]).to(torch.long)

        # concatenate x and y coordinates
        batch = batch.copy()
        batch['prompt'] = torch.cat((batch['prompt'][0].reshape(-1, 1), batch['prompt'][1].reshape(-1, 1)), dim=1)
        point_coords = batch['prompt'][:, None, :]

        # mark these points as foreground points
        point_labels = torch.ones((B, 1), dtype=torch.int, device=self.device)

        ITERATE_OVER = 10
        loss_sum = None
        for i in range(ITERATE_OVER):
            # run SAM on batch
            batch_mask, batch_iou, batch_label = self.get_logits(batch, point_coords, point_labels)
            # calculate predicted binary mask
            pred_binary_mask = (batch_mask > self.model.mask_threshold).to(torch.long)

            # calculate symmetric difference between binary_label and pred_binary_mask
            symmetric_difference = (binary_label != pred_binary_mask).to(torch.long)

            # randomly pick a nonzero position in symmetric_difference (shaped B*H*W) for each image in batch
            batch_nonzero_indices = torch.zeros((B, 2), dtype=torch.long, device=self.device)
            batch_point_label = torch.zeros((B, 1), dtype=torch.long, device=self.device)
            for i in range(B):
                # get the symmetric difference for batch i
                sample = symmetric_difference[i]
                # list all nonzero indices
                nonzero_indices = torch.nonzero(sample)
                if nonzero_indices.shape[0] == 0:
                    # TODO: what happens if the two masks exactly match?
                    continue
                # randomly select an index
                random_index = torch.randint(0, nonzero_indices.shape[0], (1,))[0]
                random_position = nonzero_indices[random_index]
                random_position = random_position.unsqueeze(0)
                batch_nonzero_indices[i] = random_position
                x, y = random_position
                # don't forget to mark foreground/background!
                batch_point_label[i] = binary_label[i][x][y]

            # pick the corresponding position in binary_label and pred_binary_mask

            # point_coords is a Tensor of dimension B*N*2, where N is the number of points
            # batch_nonzero_indices if a Tensor of dimension B*2
            # concatenate batch_nonzero_indices with point_coords to form a Tensor of dimension B*(N+1)*2
            point_coords = torch.cat((point_coords, batch_nonzero_indices.unsqueeze(1)), dim=1)
            # point_labels is a Tensor of dimension B*N, where N is the number of points
            # batch_point_label if a Tensot of dimension B
            # concatenate point_labels with point_coords to form a Tensor of dimension B*(N+1)
            point_labels = torch.cat((point_labels, batch_point_label[:, None]), dim=1)

            # batch?
        
            # get loss
            segmentation_loss, iou_loss, _dice_loss, _focal_loss = self.get_loss_and_update_metric(batch, batch_mask, batch_iou, self.training_dice_metric)
            loss = self.hparams.iou_loss_coef * iou_loss + segmentation_loss
            if loss_sum is None:
                loss_sum = loss
            else:
                loss_sum += loss

        ##### Logging
        self.log("train_loss/total_loss", loss)
        self.log("train_loss/iou_loss", iou_loss)
        self.log("train_loss/segmentation_loss", segmentation_loss)
        self.log("train_loss/dice_loss", _dice_loss)
        self.log("train_loss/focal_loss", _focal_loss)
        mdice, avg_dice = self.training_dice_metric.get_metrics()
        self.log("train_Dice/mDice", mdice)

        return loss_sum / ITERATE_OVER

    def on_train_epoch_end(self):
        self.training_dice_metric.reset()

    def validation_step(self, batch, batch_idx):
        batch_mask, batch_iou, batch_label = self.get_logits(batch)

        segmentation_loss, iou_loss, label_loss, _dice_loss, _focal_loss = self.get_loss_and_update_metric(batch, batch_mask, batch_iou, self.validation_dice_metric)

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
        self.validation_dice_metric.reset()

    def configure_optimizers(self):
        assert self.optimizer_type in ["AdamW"], "Unimplemented"

        if self.optimizer_type == "AdamW":   
            optimizer = optim.AdamW(self.parameters(), **self.optimizer_kwargs)
        return optimizer

if __name__ == "__main__":
    # create model
    model = SAMWithInteractiveTraining()
    # print("model:", model)
