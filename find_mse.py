import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

p1 = './brats_online_LNUS_t1ce_seg_5000_64/t1ce/'
p2 = './brats_train_64/t1ce/'

target1 = './test_corre/brats_online_LNUS_t1ce_seg_64/dif/'
target2 = './test_corre/brats_online_LNUS_t1ce_seg_64/tr/'
if not os.path.exists(target1):
    os.makedirs(target1)

if not os.path.exists(target2):
    os.makedirs(target2)
f1l = os.listdir(p1)
f2l = os.listdir(p2)
count = 0
for i in f1l:
    low = 999999
    low_p = '1'
    img1 = cv2.imread(p1 + i, cv2.IMREAD_GRAYSCALE)
    for j in f2l:
        for rot in [0, 1, -1]:
            img3 = np.flatten(cv2.flip(img1, rot) / 255.0)
            img2 = np.flatten(cv2.imread(p2 + j, cv2.IMREAD_GRAYSCALE) / 255.0)
            mse = np.sum((img3 - img2) ** 2)
            if mse < low:
                low = mse
                low_p = j
    t1 = cv2.imread(p1 + i, cv2.IMREAD_GRAYSCALE)
    t2 = cv2.imread(p2 + low_p, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(target1 + 'f{}'.format(count) + '.png', t1)
    cv2.imwrite(target2 + 'r{}'.format(count) + '.png', t2)

    count += 1
    if count % 1000 == 0:
        print(count)
