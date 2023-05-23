import torch
from torchmetrics import Dice

def evaluate(predicter, data_iter):

    dice_val = 0.00
    num = 0

    lst = 180

    for batch in data_iter:
        lst -= 1
        if lst > 0: 
            continue
        val_inputs, val_labels = (batch["image"], batch["label"])
        dice = Dice(ignore_index = 0)
        print(val_inputs[0])
        val_outputs = predicter(val_inputs[0], val_labels[0][0])
        cur = dice(torch.tensor(val_outputs, dtype = torch.int), torch.tensor(val_labels[0][0], dtype = torch.int))
        print("DICE: %.6lf" %cur)
        dice_val += cur
        num += 1
        break
        
    return dice_val / float(num)
