import numpy as np

def evaluate(predicter, data_iter):

    dice_val = 0.00
    num = 0
    pred = np.array([], dtype = np.int8)
    labels = np.array([], dtype = np.int8)

    for batch in data_iter:
        val_inputs, val_labels = (batch["image"], batch["label"])
        val_outputs = predicter(val_inputs[0], val_labels[0][0])
        val_outputs = np.array(val_outputs, dtype = np.int8)
        val_labels = np.array(val_labels, dtype = np.int8)
        pred = np.append(pred, val_outputs)
        labels = np.append(labels, val_labels[0][0])

    print(pred.shape)
    print(labels.shape)    
    nn = 0
    for i in range(1, 14):
        a = np.zeros(pred.shape)
        a[pred == i] = 1
        b = np.zeros(labels.shape)
        b[labels == i] = 1
        c = np.zeros(pred.shape)
        c[(pred == i) & (labels == i)] = 1
        if (np.sum(a) + np.sum(b) != 0):
            dice_val += ((2.00 * np.sum(c)) / (np.sum(a) + np.sum(b)))
            nn += 1
            
    return dice_val / float(nn)
