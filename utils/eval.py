import numpy as np
from tqdm import tqdm

def evaluate(predicter, data):

    dice_val = 0.00
    c = len(data)
    print(c)
    for i in range(c):
        num = 0
        chn, x, y, z = data[i]["label"].shape
        pred = np.array([], dtype = np.int8)
        labels = np.array([], dtype = np.int8)

        for j in tqdm(range(z)):
            val_inputs, val_labels = (data[i]["image"][:,:,:,j], data[i]["label"][0,:,:,j])
            val_outputs = predicter(val_inputs, val_labels)
            val_outputs = np.array(val_outputs, dtype = np.int8)
            val_labels = np.array(val_labels, dtype = np.int8)
            pred = np.append(pred, val_outputs)
            labels = np.append(labels, val_labels)

        print(pred.shape)
        print(labels.shape)    
        nn = 0
        cur_dice = 0.00
        for i in range(1, 14):
            a = np.zeros(pred.shape)
            a[pred == i] = 1
            b = np.zeros(labels.shape)
            b[labels == i] = 1
            c = np.zeros(pred.shape)
            c[(pred == i) & (labels == i)] = 1
            if (np.sum(a) + np.sum(b) != 0):
                cur_dice += ((2.00 * np.sum(c)) / (np.sum(a) + np.sum(b)))
                nn += 1
        
        dice_val += (cur_dice / float(nn))
        print(cur_dice / float(nn))
            
    return dice_val / float(c)
