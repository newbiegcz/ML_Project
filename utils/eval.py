import numpy as np

def evaluate(predicter, data_iter):

    dice_val = 0.00
    num = 0

    lst = 99

    for batch in data_iter:
        lst -= 1
        if lst > 0: 
            continue
        val_inputs, val_labels = (batch["image"], batch["label"])
        val_outputs = predicter(val_inputs[0], val_labels[0][0])
        val_outputs = np.array(val_outputs, dtype = np.int8)
        val_labels = np.array(val_labels, dtype = np.int8)
        cur = 0.00
        nn = 0
        for i in range(1, 14):
            a = np.zeros(val_outputs.shape)
            a[val_outputs == i] = 1
            b = np.zeros(val_labels[0][0].shape)
            b[val_labels[0][0] == i] = 1
            c = np.zeros(val_outputs.shape)
            c[(val_outputs == i) & (val_labels[0][0] == i)] = 1
            if (np.sum(a) + np.sum(b) != 0):
                cur += ((2.00 * np.sum(c)) / (np.sum(a) + np.sum(b)))
                nn += 1
        
        cur /= float(nn)
        print("DICE: %.6lf" %cur)
        dice_val += cur
        num += 1
        break
        
    return dice_val / float(num)
