from pytorch_lightning.loggers import WandbLogger
import lightning.pytorch as pl

import numpy as np
import torch

from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchmetrics.classification import BinaryJaccardIndex
from third_party.segment_anything.build_sam import build_sam_vit_h
from model.sam import build_sam_with_label_vit_h
from model.sam import SamWithLabel
import torch.optim as optim
import torch.nn as nn

from data.dataset import get_data_loader
import random

# TODO: 优化 optimizer 的内存 (parameters & 更好的优化器)
# TODO: Dice loss 与 Focal loss 的调参
# TODO: 降低精度 && warning 中的建议
# TODO: 定义 validation save 的频率
# TODO: 考虑 freeze 一部分，或者直接不要丢进 optimizer
# TODO: 采用和论文中同样的 Loss
# TODO: 允许 Encoder 的微调 (现在的问题是 vram 不太够)
# TODO: Finetune 时的 Prompt 改为 多点、矩形、或者一个 mask && 注意考虑 negative 的店
# TODO: 正确考虑选在 background 时的处理。分为整个 background 不合适，现在没有计算 Loss。
#           可以考虑对数据集预先分割，又或者限制选在 background 时和原本模型不能相差太多
# TODO: 仔细阅读原论文训练方法 (drop path?)  learning rate decay? .... 
# TODO: 按 mask 抽取的 prompt
# TODO: 尝试加入位置信息
# TODO: 尝试为器官加入分别的 token (?)
# TODO: 同时训练非 multimask 的部分
# TODO: 打乱 prompt 顺序以降低独立性

seed = 42
pl.seed_everything(seed)

cpu_only = False

data_device = torch.device("cpu") # 节省 vram

# TODO: 现在的 分 batch 策略似乎不能优化 vram，因为你得维护梯度
# TODO: fix hyperparameters
max_epochs = 10
batch_size = 1 # 主要是 encoder 的
batch_size_decoder = 64 # 同时最多处理多少 image & prompt pair
points_per_mask = 4
optimizer_type = "AdamW"
optimizer_kwargs = {
    "lr": 1e-5,
    "weight_decay": 0.1,
}
train_image_encoder = False
train_prompt_encoder = False
dice_loss_coef = 1.0
focal_loss_coef = 1.0
label_loss_coef = 1.0
iou_loss_coef = 1.0

class SoftDiceLoss(nn.Module):
    def __init__(self, p=1.0, smooth=1.0):
        super().__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss
    
soft_dice_loss = SoftDiceLoss()

class SAMWithLabelModule(pl.LightningModule):
    def __init__(self, model : SamWithLabel):
        super().__init__()
        self.model = model

    # def gen_grid_points(self, )
    def random_point_loss(self, features, labels, train_prompt_encoder=train_prompt_encoder):
        # TODO: 正确处理 不是 multimask 的情况
        # TODO: 正确处理 Background
        # backprop only minimum loss!
        # 注意全部是单点 prompt
        # 谨慎 detach & no_grad，额外梯度会很占用内存
        
        B, _, _, _ = labels.shape
        img_size = self.model.image_encoder.img_size
        inds = np.array(list(range(img_size * img_size)))

        decoder_inputs = []

        labels_cpu = labels.cpu().numpy()

        for b in range(B):
            bucket = [[]] * 14        # background + 13 个器官

            np.random.shuffle(inds)
            for t in inds:
                i = t // img_size
                j = t % img_size
                label = int (labels_cpu[b][0][i][j])
                if len(bucket[label]) >= points_per_mask:
                    continue
                bucket[label].append((i, j))

            for row in bucket:
                for point in row:
                    point_coords = torch.as_tensor(point, device=self.device)
                    point_labels = torch.scalar_tensor(1, dtype=torch.int, device=self.device)
                    with torch.set_grad_enabled(train_prompt_encoder):
                        # TODO: 还能优化
                        # TODO: 引入 dense prompt，现在这个不能有 dense prompt
                        sparse_embeddings, _ = self.model.prompt_encoder(
                                points=(point_coords[None, None, :], point_labels[None, None]),
                                boxes=None,
                                masks=None,
                            )
                    decoder_inputs.append((b, sparse_embeddings, (b, point)))

        feature_inds, sparse_embeddings_list, metas = zip(*decoder_inputs)
        n_inputs = len(metas)
        
        pred_lowres_masks = torch.zeros(n_inputs, 3, img_size // 4, img_size // 4, device=self.device)
        pred_iou = torch.zeros(n_inputs, 3, device=self.device)
        pred_label = torch.zeros(n_inputs, 14, device=self.device)
        
        with torch.set_grad_enabled(train_prompt_encoder):
            _image_pe = self.model.prompt_encoder.get_dense_pe()

        repeated_image_pe = torch.repeat_interleave(_image_pe, batch_size_decoder, dim=0)
        expanded_dense_embeddings = self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                    batch_size_decoder, -1, self.model.prompt_encoder.image_embedding_size[0], self.model.prompt_encoder.image_embedding_size[1]
                )

        for i in range(0, n_inputs, batch_size_decoder):
            inds = slice(i, min(n_inputs, i + batch_size_decoder))
            inds_size = min(i + batch_size_decoder, n_inputs) - i
            image_pe = repeated_image_pe
            dense_embeddings = expanded_dense_embeddings
            if inds_size != batch_size_decoder:
                image_pe = image_pe[:inds_size]
                dense_embeddings = expanded_dense_embeddings[:inds_size] # TODO: 这样只对没有 dense embedding 时有效

            image_embeddings = features[feature_inds[inds], :, :, :]
            sparse_embeddings = torch.cat(sparse_embeddings_list[inds])
   
            batch_masks, batch_iou, batch_label = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                already_unfolded=True,
            )

            pred_lowres_masks[inds] = batch_masks
            pred_iou[inds] = batch_iou
            pred_label[inds] = batch_label

        # TODO: 考虑是否应该放大 mask... && 确认原本的实现。这里为了方便选择缩小 label
        # 其实图像原本就很小的..
        lowres_labels = torch.nn.functional.interpolate(
            labels,
            (img_size // 4, img_size // 4),
            mode="nearest-exact"
        )

        # pred_masks = torch.nn.functional.interpolate(
        #     pred_lowres_masks,
        #     (img_size, img_size),
        #     mode="bilinear",
        #     align_corners=False,
        # )

        sum_segmentation_loss = 0.0
        sum_iou_loss = 0.0
        sum_label_loss = 0.0

        for i, decoder_input in enumerate(decoder_inputs):
            b, point = decoder_input[-1]
            label = int(labels_cpu[b][0][point[0]][point[1]])
            segmentation_losses = torch.zeros(3, device=self.device)
            for mask_id in range(3):
                if label == 0:
                    # TODO: fix
                    dice_loss = 0.0
                    iou_loss = 0.0
                    focal_loss = 0.0
                else:
                    binary_label = (lowres_labels[b][0] == label).to(torch.float)
                    dice_loss = soft_dice_loss(pred_lowres_masks[b][mask_id], binary_label)
                    # TODO: 慎重考虑: 真的是 mean?
                    focal_loss = sigmoid_focal_loss(pred_lowres_masks[b][mask_id], binary_label, reduction='mean')
                    binary_jaccard_index = BinaryJaccardIndex().to(self.device)
                    with torch.no_grad():
                        iou = binary_jaccard_index((pred_lowres_masks[b][mask_id] > self.model.mask_threshold).to(torch.long), binary_label.to(torch.long))
                    iou_loss = (iou - pred_iou[b][mask_id]) ** 2
                    

                # TODO: 仔细思考这里传的对不对...
                # 错误的，只有一个 mask 贡献
                sum_iou_loss += iou_loss_coef * iou_loss / 3 # 每次都贡献，避免 norm 不一致
                segmentation_losses[mask_id] = dice_loss_coef * dice_loss + focal_loss_coef * focal_loss 

            _cross_entropy_loss = nn.CrossEntropyLoss()
            label_loss = _cross_entropy_loss(pred_label[b], torch.scalar_tensor(label, dtype=torch.long, device=self.device))
            sum_label_loss += label_loss_coef * label_loss
            sum_segmentation_loss += torch.min(segmentation_losses)
        
        # TODO: 加入 encoder 后这么平均就不一定有意义了
        return {
            "label_loss": sum_label_loss / len(decoder_inputs),
            "iou_loss": sum_iou_loss / len(decoder_inputs),
            "segmentation_loss": sum_segmentation_loss / len(decoder_inputs)
        }


    def training_step(self, batch, batch_idx):
        # TODO: [WARN] 因为 vram 原因没有训练 encoder，但这个之后是要做的
        with torch.no_grad():
            features = self.model.image_encoder(batch['image'])
        loss_dict = self.random_point_loss(features, batch['label'])
        loss = loss_dict['label_loss'] + loss_dict['iou_loss'] + loss_dict['segmentation_loss']
        self.log("train_total_loss", loss)
        self.log("train_label_loss", loss_dict['label_loss'])
        self.log("train_iou_loss", loss_dict['iou_loss'])
        self.log("train_segmentation_loss", loss_dict['segmentation_loss'])
        return loss
    
    def validation_step(self, batch, batch_idx):
        features = self.model.image_encoder(batch['image']) # without grad by default
        loss_dict = self.random_point_loss(features, batch['label'])
        # TODO: 加入实际分割的 DICE 指标
        
        loss = loss_dict['label_loss'] + loss_dict['iou_loss'] + loss_dict['segmentation_loss']
        self.log("val_total_loss", loss)
        self.log("val_label_loss", loss_dict['label_loss'])
        self.log("val_iou_loss", loss_dict['iou_loss'])
        self.log("val_segmentation_loss", loss_dict['segmentation_loss'])
        return loss

    def configure_optimizers(self):
        assert optimizer_type in ["AdamW"]
        #assert not train_image_encoder and not train_prompt_encoder, "Unimplemented"
        #if not train_image_encoder and not train_prompt_encoder:
        #    params = self.model.mask_decoder.parameters()
        if optimizer_type == "AdamW":   
            optimizer = optim.AdamW(self.parameters(), **optimizer_kwargs)
        return optimizer
    
def get_sam_with_label(pretrained_checkpoint):
    org_sam = build_sam_vit_h(pretrained_checkpoint)
    org_state_dict = org_sam.state_dict()
    sam_with_label = build_sam_with_label_vit_h(None)
    new_state_dict = sam_with_label.state_dict()
    for k, v in org_state_dict.items():
        assert k in new_state_dict.keys()
        new_state_dict[k] = v
    sam_with_label.load_state_dict(new_state_dict)
    return sam_with_label

if cpu_only:
    print("[Warn] 当前训练没有使用 GPU")
    input("Type anything to continue...")

checkpoint_path = "checkpoint/sam_vit_h_4b8939.pth"

sam_module = SAMWithLabelModule(get_sam_with_label(checkpoint_path))

# first_only 用于 debug，记得改回来
train_dataloader = get_data_loader("training", "naive_to_rgb_and_normalize", batch_size, True, device=data_device, first_only=False)
val_dataloader = get_data_loader("validation", "naive_to_rgb_and_normalize", batch_size, False, device=data_device, first_only=False)

wandb_logger = WandbLogger(name="task3_debug",
                           project="SAM with Labels",
                           entity='ml-project-2023',
                           dir="./wandb")
trainer = pl.Trainer(max_epochs=max_epochs, 
                     profiler="advanced", 
                     accelerator="cpu" if cpu_only else "auto",
                     logger=wandb_logger,
                     log_every_n_steps=10)
trainer.fit(model=sam_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
