import torch
from torchvision.ops.focal_loss import sigmoid_focal_loss

# TODO: 应当对不同类别分开计算 Dice Loss
# 但这可能需要较大的 batch size 才能保证每个类别都有足够的样本
class SoftDiceLoss():
    def __init__(self, p=1.0, smooth=1.0):
        super().__init__()
        self.p = p
        self.smooth = smooth

    def __call__(self, logits, labels):
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss

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
        focal_loss = sigmoid_focal_loss(logits, labels, reduction='mean', **self.focal_loss_params)
        return self.dice_loss_coef * self.dice_loss(logits, labels) + self.focal_loss_coef * focal_loss
