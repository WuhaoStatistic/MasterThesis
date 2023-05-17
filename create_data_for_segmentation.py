import sys
import os
from PIL import Image
import numpy as np
import cv2

prefix = './BRATS_test/test_image/'
target1 = './abcde/'
# target2 = './valid_2021img_64/'
if not os.path.exists(target1):
    os.mkdir(target1)
# if not os.path.exists(target2):
#     os.mkdir(target2)


seg_dir = prefix + 'Seg'
t1ce_dir = prefix + 'T1CE'
t2_dir = prefix + 'T2'

seg_l = os.listdir(seg_dir)
t1ce_l = os.listdir(t1ce_dir)
t12_l = os.listdir(t2_dir)

tot = len(t1ce_l)
current = 0
cont = 0
for se, t1ce,t2 in zip(seg_l, t1ce_l, t12_l):
    if np.random.rand() < 0.95:
        continue
    nl = se.split('_')
    name = nl[1] + nl[3].zfill(3)
    t1ce_img = cv2.cvtColor(cv2.imread(t1ce_dir + '//' + t1ce), cv2.COLOR_BGR2GRAY)
    seg_img = cv2.cvtColor(cv2.imread(seg_dir + '//' + se), cv2.COLOR_BGR2GRAY)
    t2_img = cv2.cvtColor(cv2.imread(t2_dir + '//' + t2), cv2.COLOR_BGR2GRAY)
    if len(np.unique(seg_img))<2:
        continue
    np.savez_compressed(target1 + name, t1ce=t1ce_img, t2=t2_img, seg=seg_img)
    cont += 1
    if cont % 35 == 0:
        break
# # npz format see line 31 each image is a gray image 256*256
