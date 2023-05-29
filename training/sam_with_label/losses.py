import torch
from torchvision.ops.focal_loss import sigmoid_focal_loss

class SoftDiceLoss():
    def __init__(self, p=1.0, smooth=1.0):
        super().__init__()
        self.p = p
        self.smooth = smooth

    def __call__(self, logits, labels):
        B = logits.shape[0]
        probs = torch.sigmoid(logits)
        numer = (probs * labels).reshape(B, -1).sum(dim=1)
        denor = (probs.pow(self.p) + labels.pow(self.p)).reshape(B, -1).sum(dim=1)
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        assert loss.shape[0] == B
        return loss.mean()

class SegmentationLoss():
    def __init__(self, 
                 dice_loss_coef,
                 focal_loss_coef,
                 dice_loss_params={
                     "p": 1.0,
                     "smooth": 1.0,
                 },
                 focal_loss_params={
                     "alpha": 0.25,
                     "gamma": 2.0,
                 },
                 ):
        self.dice_loss_coef = dice_loss_coef
        self.focal_loss_coef = focal_loss_coef
        self.dice_loss_params = dice_loss_params
        self.focal_loss_params = focal_loss_params
        self.dice_loss = SoftDiceLoss(**self.dice_loss_params)

    def __call__(self, logits, labels):
        B = logits.shape[0]
        if B == 0:
            return 0.0, 0.0, 0.0
        focal_loss = sigmoid_focal_loss(logits, labels, reduction='mean', **self.focal_loss_params)
        dice_loss = self.dice_loss(logits, labels)
        return self.dice_loss_coef * dice_loss + self.focal_loss_coef * focal_loss, dice_loss, focal_loss
