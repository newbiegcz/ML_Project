import lightning.pytorch as pl
from third_party.segment_anything.build_sam import build_sam_vit_h
from modeling.build_sam import sam_with_label_model_registry
from third_party.segment_anything.build_sam import sam_model_registry
import torch
import torch.optim as optim
import torch.nn as nn
from .losses import SegmentationLoss
from utils.visualize import default_label_names
import utils.visualize as visualize
from modeling.build_sam import pretrained_checkpoints

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
        if (avg_dice != avg_dice).any():
            print("Warning: nan in avg_dice")
        if ignore_background:
            avg_dice[0] = torch.nan
        mdice = torch.nanmean(avg_dice)
        return mdice, avg_dice
    
def iou_func(pred_binary_mask, binary_label):
    B = pred_binary_mask.shape[0]
    pred_binary_mask = pred_binary_mask.view(B, -1)
    binary_label = binary_label.view(B, -1)
    intersection = torch.count_nonzero(pred_binary_mask & binary_label, dim=1)
    union = torch.count_nonzero(pred_binary_mask | (binary_label), dim=1)
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

        self.pretrained_checkpoint = pretrained_checkpoints[model_type]
        self.model_type = model_type
        self.train_image_encoder = train_image_encoder
        self.train_prompt_encoder = train_prompt_encoder
        self.dice_loss_coef = dice_loss_coef
        self.focal_loss_coef = focal_loss_coef
        self.label_loss_coef = label_loss_coef
        self.iou_loss_coef = iou_loss_coef
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.debug = debug
        
        self.model, self.encoder_builder  = sam_with_label_model_registry[model_type](None, train_image_encoder)
        pretrained_sam = sam_model_registry[model_type](self.pretrained_checkpoint)
        pretrained_state_dict = pretrained_sam.state_dict()
        new_state_dict = self.model.state_dict()
        for k, v in pretrained_state_dict.items():
            if not train_image_encoder and k.startswith("image_encoder"): 
                continue
            assert k in new_state_dict.keys()
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)

        self.segmentation_loss = SegmentationLoss(
            dice_loss_coef=dice_loss_coef,
            focal_loss_coef=focal_loss_coef,
            dice_loss_params=dice_loss_params,
            focal_loss_params=focal_loss_params,
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)
        self.training_dice_metric = DiceMetric()
        self.validation_dice_metric = DiceMetric()

        if self.debug:
            visualize.initialize_window()

    def get_logits(self, batch):
        # 未来如果实现训练 encoder，一定要记得判断 torch.is_grad_enabled()，避免 validate 时保留梯度
        assert not self.train_image_encoder, "Unimplemented"

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

        point_coords = batch['prompt'][:, None, :]
        point_labels = torch.ones((B, 1), dtype=torch.int, device=self.device)

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
    
    def get_loss_and_update_metric(self, batch, batch_mask, batch_iou, batch_label, metric, batch_name):
        B = len(batch['embedding'])

        segmentation_loss = 0.0
        iou_loss = 0.0
        label_loss = 0.0
        
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
            iou = iou_func(pred_binary_mask, binary_label)
            assert (iou.shape[0] == B)
            # bincount for each data point
            intersection = (lowres_labels[:, 0] * pred_binary_mask).reshape(B, -1)
            _ids = (intersection + (14 * torch.arange(B, device=self.device).reshape(-1, 1))).to(torch.long)
            count = torch.bincount(_ids.reshape(-1), minlength=14 * B).reshape(B, 14)
            count[:, 0] -= ((pred_binary_mask==0).reshape(B, -1)).sum(dim=1)
            # print(count[0])
            S = torch.sum(count, dim=1, keepdim=True)
            label_density = count / S
            assert (label_density.shape[0] == B)

            one_hot_background = torch.zeros((14,), dtype=label_density.dtype, device=label_density.device)
            one_hot_background[0] = 1

            if torch.isnan(label_density).any():
                print("\n\n\nWarning: nan in label_density!!!!! \n 可能的一种解释是针对背景，你的 model 要求 loss，所以他可能逐渐学会不把背景分出来。现在这样处理，之后有待验证。\n\n\n")

            label_density[S[:, 0] == 0] = one_hot_background.unsqueeze(0)

            metric.update(pred_binary_mask, binary_label, mask_cls)
            # print(iou)
        
        iou_loss = ((iou - batch_iou) ** 2).mean()
        label_loss = self.cross_entropy_loss(batch_label, label_density)

        if self.debug:
            if batch_name is None:
                batch_name = "unnamed"

            pd_output_label = torch.nn.functional.interpolate(
                pred_binary_mask[0:1, None, :, :].to(torch.float32),
                (1024, 1024),
                mode="nearest-exact"
            )[0].cpu().numpy()

            gt_output_label = torch.nn.functional.interpolate(
                binary_label[0:1, None, :, :].to(torch.float32),
                (1024, 1024),
                mode="nearest-exact"
            )[0].cpu().numpy()

            prompt = batch['prompt'][0].detach().cpu().numpy()
            visualize.add_object_2d(batch_name + "_img0",
                    image=batch['image'][0].detach().cpu().numpy(),
                    pd_label=pd_output_label,
                    gt_label=gt_output_label,
                    prompt_points=[(prompt, 1)],
                    label_name=["negative", "positive"],
                        extras={
                        "prompt": prompt,
                        "prompt_label": default_label_names[mask_cls[0].cpu().to(torch.int)]
                    }
            )
            input("")


        return segmentation_loss, iou_loss, label_loss, _dice_loss, _focal_loss


    def training_step(self, batch, batch_idx):
        batch_mask, batch_iou, batch_label = self.get_logits(batch)

        segmentation_loss, iou_loss, label_loss, _dice_loss, _focal_loss = self.get_loss_and_update_metric(batch, batch_mask, batch_iou, batch_label, self.training_dice_metric, "train_%d" % batch_idx)

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
        self.training_dice_metric.reset()
    
    def validation_step(self, batch, batch_idx):
        batch_mask, batch_iou, batch_label = self.get_logits(batch)

        segmentation_loss, iou_loss, label_loss, _dice_loss, _focal_loss = self.get_loss_and_update_metric(batch, batch_mask, batch_iou, batch_label, self.validation_dice_metric, "val_%d" % batch_idx)

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
        self.validation_dice_metric.reset()

    def configure_optimizers(self):
        assert self.optimizer_type in ["AdamW"], "Unimplemented"

        if self.optimizer_type == "AdamW":   
            optimizer = optim.AdamW(self.parameters(), **self.optimizer_kwargs)
        return optimizer