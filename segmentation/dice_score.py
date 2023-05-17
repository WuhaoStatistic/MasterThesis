import nibabel as nib
import numpy as np
import os
import pandas as pd


def dice_score(prediction, ground_truth, class_index):
    pred_mask = prediction == class_index
    true_mask = ground_truth == class_index
    intersection = np.sum(pred_mask * true_mask)
    dices = (2. * intersection) / (np.sum(pred_mask) + np.sum(true_mask))
    return dices


index = ['002', '004', '009', '010', '012', '013', '018', '020', '023',
         '024', '039', '057', '065', '066', '075', '085', '095', '114',
         '118', '122', '126', '132', '137', '141', '153', '154', '158',
         '167', '168', '191', '206', '207', '215', '222', '225', '226',
         '245', '251', '260', '262', '263', '276', '277', '285', '288',
         '293', '297', '313', '334', '341', '342', '348', '350', '359', '363', '368']

col = ['51', '102', '204']
df = pd.DataFrame(index=index, columns=col)
# -----------------------------------------------
# change this manually
ground_truth_path = "../test_label_64"   # Todo change here
type = 'aug_every_5'  # Todo change here
prediction_path = "../seg_res/"+type
record = "../seg_record/"+type+".csv"  # Todo change
# ------------------------------------------------
pre_f = os.listdir(prediction_path)
gd_f = os.listdir(ground_truth_path)
pre_f.sort()
gd_f.sort()
count = 0
for p, g in zip(pre_f, gd_f):

    prediction_image = nib.load(prediction_path + '/' + p)
    prediction_volume = prediction_image.get_fdata()
    ground_truth_image = nib.load(ground_truth_path + '/' + g)
    ground_truth_volume = ground_truth_image.get_fdata()
    if prediction_volume.shape != ground_truth_volume.shape:
        raise ValueError("Prediction and ground truth volumes must have the same shape")
    class_labels = np.unique(ground_truth_volume)
    num_classes = len(class_labels)
    for i in range(num_classes):
        class_label = class_labels[i]
        if class_label == 0:
            continue
        dice = dice_score(prediction_volume, ground_truth_volume, class_label)
        df.loc[index[count], class_label] = dice
        # print(f"Dice score for class {class_label}: {dice:.4f}")
    count += 1
df.to_csv(record)
