import torch
from torchmetrics import Dice

def evaluate(predicter, data):

    c = len(data)
    dice_val = 0.00

    for i in range(c):
        dz = len(data.data_list[i])
        dx, dy = data.data_list[i][0]["label"].shape
        for j in range(dz):
            nonzero_indexes = torch.nonzero(data.data_list[i][j]["label"])
            if nonzero_indexes.shape[0] == 0:
                continue
            else:
                lz = min(lz, j)
                rz = max(rz, j)
                x1 = nonzero_indexes[:, 0].min().item()
                x2 = nonzero_indexes[:, 0].max().item()
                y1 = nonzero_indexes[:, 1].min().item()
                y2 = nonzero_indexes[:, 1].max().item()

                lx = min(lx, x1)
                rx = max(rx, x2)
                ly = min(ly, y1)
                ry = max(ry, y2)
    
        deltaz = int(float(rz - lz) * 0.1)
        deltax = int(float(rx - lx) * 0.1)
        deltay = int(float(ry - ly) * 0.1)

        lz = max(0, lz - deltaz)
        rz = min(dz - 1, rz + deltaz)
        lx = max(0, lx - deltax)
        rx = min(dx - 1, rx + deltax)
        ly = max(0, ly - deltay)
        ry = min(dy - 1, ry + deltay)

    for batch in data_iter:
        val_inputs, val_labels = (batch["image"], batch["label"])
        _, _ , n, _, _ = val_inputs.shape
        dice = Dice(ignore_index = 0)
        for i in range(170, 171):
            val_outputs = predicter(val_inputs[0,0,:,:,i], val_labels[0,0,:,:,i])
            cur = dice(torch.tensor(val_outputs, dtype = torch.int), torch.tensor(val_labels[0,0,:,:,i], dtype = torch.int))
            print("DICE: %.6lf" %cur)
            dice_val += cur
            num += 1
        break
        
    return dice_val / float(num)
