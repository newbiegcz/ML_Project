import lightning.pytorch as pl
from modeling.build_sam import pretrained_checkpoints, sam_with_label_model_registry
from .losses import SegmentationLoss
from third_party.segment_anything.build_sam import sam_model_registry
import torch.nn as nn
from .model_module import DiceMetric, iou_func

# create pl.LightningModule named SAMWithInteractiveTraining
class SAMWithInteractiveTraining(pl.LightningModule):
    
    # init
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
                 debug: bool = True):
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
        # init superclass
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()

        # retrieve pretrained model
        self.pretrained_checkpoint = pretrained_checkpoints[model_type]
        pretrained_sam = sam_model_registry[model_type](self.pretrained_checkpoint)

        # load state dict from corresponding pretrained state dict
        self.model, self.encoder_builder = sam_with_label_model_registry[model_type](None, train_image_encoder)
        pretrained_state_dict = pretrained_sam.state_dict()
        new_state_dict = self.model.state_dict()
        for k, v in pretrained_state_dict.items():
            if not train_image_encoder and k.startswith("image_encoder"):
                # freeze image encoder
                continue
            
            assert k in new_state_dict.keys()
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)

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
    
    def get_logits(self, batch):
        """
        Args:
            batch: dict, batch of data, should contain keys ["embedding", "label", "mask_cls", "prompt"]
        Returns:
        """
        # image encoder training is not implemented
        assert not self.train_image_encoder, "Unimplemented"

        B = len(batch['embedding'])

        embeddings = batch['embedding']

        batch = batch.copy()

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


if __name__ == "__main__":
    # create model
    model = SAMWithInteractiveTraining()
    print("model:", model)