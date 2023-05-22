import torch
from torchmetrics import Dice

def evaluate(predicter, data_iter):

    dice_val = 0.00
    num = 0

    for batch in data_iter:
        val_inputs, val_labels = (batch["image"], batch["label"])
        _, _ , n, _, _ = val_inputs.shape
        dice = Dice(ignore_index = 0)
        for i in range(n):
            val_outputs = predicter(val_inputs[0,0,:,:,i], val_labels[0,0,:,:,i])
            cur = dice(torch.tensor(val_outputs, dtype = torch.int), torch.tensor(val_labels[0,0,:,:,i], dtype = torch.int))
            print("DICE: %.6lf" %cur)
            dice_val += cur
            num += 1
        break
        
    return dice_val / float(num)
