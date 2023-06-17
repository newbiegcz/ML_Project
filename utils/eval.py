import numpy as np
from tqdm import tqdm

def evaluate(predicter, data, max_point, require_box, consider_connecting):

    num_people = len(data)
    res_p = []
    res_b = 0.00
    for n_c in range(1, max_point + 1):
        dice_val = 0.00
        for i in range(num_people):
            chn, x, y, z = data[i]["label"].shape
            pred = np.array([], dtype = np.int8)
            labels = np.array([], dtype = np.int8)

            for j in tqdm(range(z)):
                val_inputs, val_labels = (data[i]["image"][:,:,:,j], data[i]["label"][0,:,:,j])
                val_outputs = predicter(val_inputs, val_labels, n_c, True, False, consider_connecting)
                val_outputs = np.array(val_outputs, dtype = np.int8)
                val_labels = np.array(val_labels, dtype = np.int8)
                pred = np.append(pred, val_outputs)
                labels = np.append(labels, val_labels)

            nn = 0
            cur_dice = 0.00
            for j in range(1, 14):
                a = np.zeros(pred.shape)
                a[pred == j] = 1
                b = np.zeros(labels.shape)
                b[labels == j] = 1
                c = np.zeros(pred.shape)
                c[(pred == j) & (labels == j)] = 1
                if (np.sum(a) + np.sum(b) != 0):
                    cur_dice += ((2.00 * np.sum(c)) / (np.sum(a) + np.sum(b)))
                    nn += 1
        
            dice_val += (cur_dice / nn)
            print(cur_dice / nn)
        
        print(dice_val / num_people)
        res_p.append(dice_val / num_people)
    
    if require_box:
        dice_val = 0.00
        for i in range(num_people):
            chn, x, y, z = data[i]["label"].shape
            pred = np.array([], dtype = np.int8)
            labels = np.array([], dtype = np.int8)

            for j in tqdm(range(z)):
                val_inputs, val_labels = (data[i]["image"][:,:,:,j], data[i]["label"][0,:,:,j])
                val_outputs = predicter(val_inputs, val_labels, 1, False, True, consider_connecting)
                val_outputs = np.array(val_outputs, dtype = np.int8)
                val_labels = np.array(val_labels, dtype = np.int8)
                pred = np.append(pred, val_outputs)
                labels = np.append(labels, val_labels)

            nn = 0
            cur_dice = 0.00
            for j in range(1, 14):
                a = np.zeros(pred.shape)
                a[pred == j] = 1
                b = np.zeros(labels.shape)
                b[labels == j] = 1
                c = np.zeros(pred.shape)
                c[(pred == j) & (labels == j)] = 1
                if (np.sum(a) + np.sum(b) != 0):
                    cur_dice += ((2.00 * np.sum(c)) / (np.sum(a) + np.sum(b)))
                    nn += 1
        
            dice_val += (cur_dice / nn)
            print(cur_dice / nn)
        
        print(dice_val / num_people)
        res_b = dice_val / num_people
            
    return res_p, res_b
