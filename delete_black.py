import sys
import os
from PIL import Image
import numpy as np
import cv2

prefix = './BRATS2021_train/brats2021_1195/'
target1 = './train_augimg2021_healthy_64/'
# target2 = './test_2021img_64/'
if not os.path.exists(target1):
    os.mkdir(target1)
# if not os.path.exists(target2):
#     os.mkdir(target2)

flair_dir = prefix + 'FLAIR'
seg_dir = prefix + 'Seg'
t1_dir = prefix + 'T1'
t1ce_dir = prefix + 'T1CE'
t2_dir = prefix + 'T2'

flair_l = os.listdir(flair_dir)
seg_l = os.listdir(seg_dir)
t1_l = os.listdir(t1_dir)
t1ce_l = os.listdir(t1ce_dir)
t2_l = os.listdir(t2_dir)

tot = len(t2_l)
current = 0
cont = 0
for fl, se, t1, t1ce, t2 in zip(flair_l, seg_l, t1_l, t1ce_l, t2_l):
    nl = fl.split('_')
    name = nl[1] + nl[3].zfill(3)

    seg_img = cv2.cvtColor(cv2.imread(seg_dir + '//' + se), cv2.COLOR_BGR2GRAY)
    if np.all(seg_img == 0):
        seg_img = cv2.resize(seg_img, (64, 64), interpolation=cv2.INTER_NEAREST)
        flair_img = cv2.cvtColor(cv2.imread(flair_dir + '//' + fl), cv2.COLOR_BGR2GRAY)
        flair_img = cv2.resize(flair_img, (64, 64), interpolation=cv2.INTER_LINEAR)
        t1_img = cv2.cvtColor(cv2.imread(t1_dir + '//' + t1), cv2.COLOR_BGR2GRAY)
        t1_img = cv2.resize(t1_img, (64, 64), interpolation=cv2.INTER_LINEAR)
        t1ce_img = cv2.cvtColor(cv2.imread(t1ce_dir + '//' + t1ce), cv2.COLOR_BGR2GRAY)
        t1ce_img = cv2.resize(t1ce_img, (64, 64), interpolation=cv2.INTER_LINEAR)
        t2_img = cv2.cvtColor(cv2.imread(t2_dir + '//' + t2), cv2.COLOR_BGR2GRAY)
        t2_img = cv2.resize(t2_img, (64, 64), interpolation=cv2.INTER_NEAREST)
        if current == 7:
            # np.savez_compressed(target2 + name,
            #                     t1=t1_img,
            #                     t1ce=t1ce_img,
            #                     t2=t2_img,
            #                     flair=flair_img,
            #                     seg=seg_img)  # **
            current = 0
        else:
            np.savez_compressed(target1 + name,
                                t1=t1_img,
                                t1ce=t1ce_img,
                                t2=t2_img,
                                flair=flair_img,
                                seg=seg_img)  # **
    # current += 1
    cont += 1
    if cont % 5000 == 0:
        print('{}/{} finished'.format(cont, tot))
