import lightning.pytorch as pl
from .losses import SegmentationLoss
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn as nn
from sam_grad.build_sam import sam_with_gradients_model_registry, pretrained_checkpoints
from third_party.segment_anything.build_sam import sam_model_registry


# TODO
# - [x?] 整多个 metric[]，对每个metric进行log
# - [x?] 把batch["label"]改成batch["connected_mask"]
# - [x?] 随机决定选点/矩形改掉
# NOTE
# - 没有在新prompt里传入前一个prompt的mask
# - self.optimizers()

ITERATE_OVER = 8 # NOTE 论文里是8

def iou_func(pred_binary_mask, binary_label):
    B = pred_binary_mask.shape[0]
    pred_binary_mask = pred_binary_mask.view(B, -1)
    binary_label = binary_label.view(B, -1)
    intersection = torch.count_nonzero(pred_binary_mask & binary_label, dim=1)
    union = torch.count_nonzero(pred_binary_mask | (binary_label), dim=1)
    return intersection / union

# def clean_up_batch(batch):
#     mask_cls = batch['mask_cls']
#     is_foreground_label = mask_cls != 0
#     # print(is_foreground_label.shape, is_foreground_label)
#     batch['embedding'] = batch['embedding'][is_foreground_label]
#     # print(batch['prompt'])
#     # print(batch['prompt'][0])
#     batch['prompt'] = [batch['prompt'][0][is_foreground_label], batch['prompt'][1][is_foreground_label]]
#     batch['connected_mask'] = batch['connected_mask'][is_foreground_label]
#     batch['mask_cls'] = batch['mask_cls'][is_foreground_label]
#     return batch

def print_all_tensors(model):
    # Get a list of all tensors allocated on the GPU
    all_tensors = list(model.parameters())

    # Sort the tensors based on their memory usage
    sorted_tensors = sorted(all_tensors, key=lambda tensor: tensor.numel() * tensor.element_size(), reverse=True)

    # Print the top N tensors with the most memory usage
    N = 5  # Number of tensors to print
    for i, tensor in enumerate(sorted_tensors[:N]):
        print(f"Tensor {i+1}: Size: {tensor.size()}, Memory Usage: {tensor.numel() * tensor.element_size()} bytes")
    print(f"Total Usage: {sum(map(lambda tensor: tensor.numel() * tensor.element_size(), all_tensors))}")

def print_memory_usage(tensor):
    print(f"Size: {tensor.size()}, Memory Usage: {tensor.numel() * tensor.element_size()} bytes")

def print_miniature(sample, step = 8):
    print(sample.shape)
    for xx in range(0, sample.shape[0], step):
        for yy in range(0, sample.shape[1], step):
            print(int(sample[xx, yy]), end='')
        print()

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

# create pl.LightningModule named SAMWithInteractiveTraining
class SAMWithInteractiveTraining(pl.LightningModule):
    # init
    def __init__(self, 
                 model_type: str = "vit_h",
                 train_image_encoder: bool = False,
                 train_prompt_encoder: bool = True,
                 dice_loss_coef: float = 1.0,
                 focal_loss_coef: float = 1.0,
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
        self.model_type = model_type
        self.train_image_encoder = train_image_encoder
        self.train_prompt_encoder = train_prompt_encoder
        self.dice_loss_coef = dice_loss_coef
        self.focal_loss_coef = focal_loss_coef
        self.iou_loss_coef = iou_loss_coef
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.debug = debug

        # WARNING! Automatic optimization is disabled!
        self.automatic_optimization=False

        # retrieve pretrained model
        self.pretrained_checkpoint = pretrained_checkpoints[model_type]
        self.model, self.encoder_builder = sam_with_gradients_model_registry[model_type](None, train_image_encoder)
        pretrained_sam = sam_model_registry[model_type](self.pretrained_checkpoint)
        pretrained_state_dict = pretrained_sam.state_dict()
        new_state_dict = self.model.state_dict()
        for k, v in pretrained_state_dict.items():
            if not train_image_encoder and k.startswith("image_encoder"): 
                continue
            assert k in new_state_dict.keys()
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        # self.model.mask_decoder.copy_weights()

        # init losses
        self.segmentation_loss = SegmentationLoss(
            dice_loss_coef=dice_loss_coef,
            focal_loss_coef=focal_loss_coef,
            dice_loss_params=dice_loss_params,
            focal_loss_params=focal_loss_params,
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)
        self.training_dice_metrics = [DiceMetric() for i in range(ITERATE_OVER+1)]
        self.validation_dice_metric = DiceMetric()
    
    def get_logits(self,
                   batch: dict,
                   point_coords: torch.Tensor,
                   point_labels: torch.Tensor,
                   bounding_boxes: torch.Tensor,
                   prev_masks: torch.Tensor,):
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

        # print("get_logits0", torch.cuda.memory_allocated())
        # batch = clean_up_batch(batch)
        B = len(batch['embedding'])
        embeddings = batch['embedding']
        # print(point_coords, point_labels)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_prompt_encoder):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                                    points=(point_coords, point_labels) if point_coords is not None else None,
                                    boxes=bounding_boxes,
                                    masks=None, # TODO? prev_masks.unsqueeze(1) if prev_masks is not None else None,
                                )
        # print("get_logits1", torch.cuda.memory_allocated(), torch.cuda.memory_cached())
        batch_masks, batch_ious = self.model.mask_decoder(
            image_embeddings=embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            already_unfolded=True,
        )
        # print("get_logits2", torch.cuda.memory_allocated(), torch.cuda.memory_cached())
        assert batch_masks.shape[1] == 1
        batch_mask = batch_masks[:, 0]
        batch_iou = batch_ious[:, 0]

        return batch_mask, batch_iou

    def get_loss_and_update_metric(self, batch, batch_mask, batch_iou, metric):
        # batch = clean_up_batch(batch)
        B = len(batch['embedding'])
        # print("get_loss_and_update_metric0", torch.cuda.memory_allocated())
        # print(B)

        segmentation_loss = 0.0
        iou_loss = 0.0
        
        img_size = 1024

        mask_cls = batch['mask_cls']
        is_foreground_label = mask_cls != 0
        lowres_mask = torch.nn.functional.interpolate(
            (batch['connected_mask']).to(torch.float32),
            (img_size // 4, img_size // 4),
            mode="nearest-exact"
        )
        binary_label = lowres_mask[:, 0]
        segmentation_loss, _dice_loss, _focal_loss = self.segmentation_loss(batch_mask[is_foreground_label], binary_label[is_foreground_label])
        # segmentation_loss, _dice_loss, _focal_loss = self.segmentation_loss(batch_mask, binary_label)
        
        with torch.no_grad():
            # print(batch_mask.shape)
            pred_binary_mask = (batch_mask > self.model.mask_threshold).to(torch.long)
            # print(pred_binary_mask.shape)
            binary_label = binary_label.to(torch.long)
            # print(binary_label.shape)
            iou = iou_func(pred_binary_mask, binary_label)
            # print(iou.shape)
            # 现在可能存在背景点了
            # assert (iou.shape[0] == B)

            metric.update(pred_binary_mask, binary_label, mask_cls)
        
        batch_iou = batch_iou[is_foreground_label]
        iou = iou[is_foreground_label]
        iou_loss = ((iou - batch_iou) ** 2).mean()

        return segmentation_loss, iou_loss, _dice_loss, _focal_loss

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        # print("training_step0", torch.cuda.memory_allocated())
        # calculate binary_label

        # batch = clean_up_batch(batch)
        img_size = 1024
        B = len(batch['connected_mask'])
        lowres_mask = torch.nn.functional.interpolate(
            (batch['connected_mask']).to(torch.float32),
            (img_size // 4, img_size // 4),
            mode="nearest-exact"
        )
        binary_label = lowres_mask[:, 0]

        # concatenate x and y coordinates
        # print("training_step1", torch.cuda.memory_allocated())
        batch = batch.copy()
        # if coin flip is heads, start with a point prompt
        for prompt in range(2):
            if prompt == 0:
                point_coords = torch.cat((batch['prompt'][0].reshape(-1, 1), batch['prompt'][1].reshape(-1, 1)), dim=1)[:, None, :]

                # mark these points as foreground points
                point_labels = torch.ones((B, 1), dtype=torch.int, device=self.device)
                bounding_boxes = None
            else:
                point_coords = None
                point_labels = None
                bounding_boxes = torch.zeros((B, 4), dtype=torch.float, device=self.device)
                for i in range(B):
                    # get the bounding box for batch i
                    # print("batch['connected_mask']")
                    # print(batch['connected_mask'].shape)
                    sample = batch['connected_mask'][i][0]
                    # print(sample.shape)
                    # print a miniature version of the sample
                    # print(sample[::4, ::4])
                    # print_miniature(sample)

                    nonzero_indices = torch.nonzero(sample)
                    assert(nonzero_indices.shape[0] != 0)

                    W, H = sample.shape
                    x1, y1 = nonzero_indices.min(dim=0)[0]
                    x2, y2 = nonzero_indices.max(dim=0)[0]
                    
                    def jiggle(x, w, W):
                        # randomly jiggle by at most 10% of the bounding box size, at most 20 pixels
                        w = int(0.1 * w)
                        w = min(w, 20)
                        if (w == 0):
                            return x
                        dx = torch.randint(-w, w, (1,)).item()
                        x = min(max(x + dx, 0), W)
                        return x
                    w = x2 - x1
                    h = y2 - y1
                    # print(torch.Tensor([y1, x1, y2, x2]))
                    x1 = jiggle(x1, w, W)
                    x2 = jiggle(x2, w, W)
                    y1 = jiggle(y1, h, H)
                    y2 = jiggle(y2, h, H)
                    bounding_boxes[i] = torch.tensor([y1, x1, y2, x2], dtype=torch.float, device=self.device)
                    # print(bounding_boxes[i])

            prev_masks = None
            for _ in range(ITERATE_OVER if prompt == 0 else 1):
                # run SAM on batch
                batch_mask, batch_iou = self.get_logits(batch, point_coords, point_labels, bounding_boxes, prev_masks)
                # let prev_masks = batch_mask, but don't save grad
                prev_masks = batch_mask.detach()

                # calculate predicted binary mask
                pred_binary_mask = (batch_mask > self.model.mask_threshold).to(torch.long)
                if self.debug:
                    pass
                    # for i in range(B):
                    #     print(f"binary_label[{i}]")
                    #     print_miniature(binary_label[i])
                    #     print(f"bounding_boxes[{i}]")
                    #     print(bounding_boxes[i])
                    #     print(point_coords, point_labels)
                    #     print(f"pred_binary_mask[{i}]")
                    #     print_miniature(pred_binary_mask[i])

                # calculate symmetric difference between binary_label and pred_binary_mask
                symmetric_difference = (binary_label != pred_binary_mask).to(torch.long)

                # randomly pick a nonzero position in symmetric_difference (shaped B*H*W) for each image in batch
                batch_nonzero_indices = torch.zeros((B, 2), dtype=torch.long, device=self.device)
                batch_point_label = torch.zeros((B, 1), dtype=torch.long, device=self.device)
                for j in range(B):
                    # get the symmetric difference for batch i
                    sample = symmetric_difference[j]
                    # list all nonzero indices
                    nonzero_indices = torch.nonzero(sample)
                    if nonzero_indices.shape[0] == 0:
                        # TODO: what happens if the two masks exactly match?
                        continue
                    # randomly select an index
                    random_index = torch.randint(0, nonzero_indices.shape[0], (1,))[0]
                    random_position = nonzero_indices[random_index]
                    # print(random_position.shape)
                    x, y = random_position
                    # random_position = random_position.unsqueeze(0)
                    # print(random_position.shape)
                    batch_nonzero_indices[j] = random_position
                    # unpack Tensor random_position
                    # don't forget to mark foreground/background!
                    batch_point_label[j] = binary_label[j][x][y]

                # pick the corresponding position in binary_label and pred_binary_mask
                # print("training_step5", _, torch.cuda.memory_allocated())
                # point_coords is a Tensor of dimension B*N*2, where N is the number of points
                # batch_nonzero_indices if a Tensor of dimension B*2
                # concatenate batch_nonzero_indices with point_coords to form a Tensor of dimension B*(N+1)*2
                if point_coords is not None:
                    point_coords = torch.cat((point_coords, batch_nonzero_indices.unsqueeze(1)), dim=1)
                else:
                    point_coords = batch_nonzero_indices.unsqueeze(1)
                # point_labels is a Tensor of dimension B*N, where N is the number of points
                # batch_point_label if a Tensor of dimension B
                # concatenate point_labels with point_coords to form a Tensor of dimension B*(N+1)
                if point_labels is not None:
                    point_labels = torch.cat((point_labels, batch_point_label), dim=1)
                else:
                    point_labels = batch_point_label

                # batch?
            
                # get loss
                segmentation_loss, iou_loss, _dice_loss, _focal_loss = self.get_loss_and_update_metric(batch, batch_mask, batch_iou, self.training_dice_metrics[_+1-prompt])
                loss = self.iou_loss_coef * iou_loss + segmentation_loss
                self.manual_backward(loss)
            

        ##### Logging
        self.log("train_loss/total_loss", loss)
        self.log("train_loss/iou_loss", iou_loss)
        self.log("train_loss/segmentation_loss", segmentation_loss)
        self.log("train_loss/dice_loss", _dice_loss)
        self.log("train_loss/focal_loss", _focal_loss)
        for _ in range(ITERATE_OVER+1):
            mdice, avg_dice = self.training_dice_metrics[_].get_metrics()
            self.log(f"train_Dice/prompt{_}/mDice", mdice)
            for i in range(14):
                self.log(f"train_Dice/prompt{_}/Dice{i}", avg_dice[i])
            

        opt.step()
        return loss / ITERATE_OVER

    def on_train_epoch_end(self):
        for i in range(ITERATE_OVER):
            self.training_dice_metrics[i].reset()

    def validation_step(self, batch, batch_idx):
        # batch = clean_up_batch(batch)
        B = len(batch['embedding'])
        batch = batch.copy()
        point_coords = torch.cat((batch['prompt'][0].reshape(-1, 1), batch['prompt'][1].reshape(-1, 1)), dim=1)[:, None, :]

        # mark these points as foreground points
        point_labels = torch.ones((B, 1), dtype=torch.int, device=self.device)

        batch_mask, batch_iou = self.get_logits(batch, point_coords, point_labels, None, None)

        segmentation_loss, iou_loss, _dice_loss, _focal_loss = self.get_loss_and_update_metric(batch, batch_mask, batch_iou, self.validation_dice_metric)

        loss = self.iou_loss_coef * iou_loss + segmentation_loss

        self.log("val_loss/total_loss", loss)
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
            l = list(self.parameters())
            # print(l, self.optimizer_kwargs)
            optimizer = optim.AdamW(l, **self.optimizer_kwargs)
        return optimizer

if __name__ == "__main__":
    # create model
    model = SAMWithInteractiveTraining()
    print_all_tensors(model.model)
    # print("model:", model)
    # print("model:", model.model)
