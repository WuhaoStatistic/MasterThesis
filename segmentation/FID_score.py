import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import subprocess

# 在terminal里面打这段话就可以了 , 路径对上就好
# terminal 的当前路径为 (new_torch) E:\pycharm\new_torch\thesis_test>

# top1 = './BRATS_test/test_image' os.system('python -m pytorch_fid ./brats_train_64/t1ce
# ./brats_LNUS_t1ce_seg_64_nk/LNUS_t1ce_seg_64_1000k --device cuda:0')


# todo change -----------------------
top = 'E:\\pycharm\\new_torch\\thesis_test'
dif_lst = [
    'brats_every_5_t1ce_64_nk\\every5_07',
    'brats_every_5_t1ce_64_nk\\every5_14',
    'brats_every_5_t1ce_64_nk\\every5_21',
    'brats_every_5_t1ce_64_nk\\every5_28',
    'brats_every_5_t1ce_64_nk\\every5_35'
]

train_dataset = ['brats_2021_train_64', 'brats_2021_test_64', 'brats_every_5_original']
cha_lst = ['t1ce', 'seg']
train_channel = ['t1ce', 'Seg']
col = ['red', 'blue','purple']
x_axis = ['70k', '140k', '210k', '280', '350k']
# todo change -----------------------

assert len(x_axis) == len(dif_lst)
assert len(cha_lst) == len(train_channel)

for cha in range(len(cha_lst)):
    for re in range(len(train_dataset)):
        res_l = []
        for dif_index in range(len(dif_lst)):
            path_dif = os.path.join(top, dif_lst[dif_index], cha_lst[cha])
            path_train = os.path.join(top, train_dataset[re], train_channel[cha])
            cmd = 'python -m pytorch_fid ' + path_dif + ' ' + path_train + ' --device cuda:0'
            res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
            ot = res.stdout.decode().strip()
            print(ot)
            FID = ot.split('  ')[1].split('\r')[0]
            FID = float(FID)
            res_l.append(FID)
        print(cha_lst[cha])
        print(res_l)
        if train_dataset[re].split('_')[2] == 'train':  # todo change here to match your directory name
            plt.plot(x_axis, res_l, label=train_dataset[re].split('_')[2] + '_' + cha_lst[cha], color=col[re])
        if train_dataset[re].split('_')[2] == 'test':
            plt.plot(x_axis, res_l, label=train_dataset[re].split('_')[2] + '_' + cha_lst[cha], color=col[re])
        if train_dataset[re].split('_')[2] == '5':
            plt.plot(x_axis, res_l, label='every_5' + '_' + cha_lst[cha], color=col[re])

    # cmd = 'python -m pytorch_fid ' + \
    #       os.path.join(top, train_dataset[0], train_channel[cha]) + ' ' + \
    #       os.path.join(top, train_dataset[1], train_channel[cha]) + ' --device cuda:0'
    #
    # res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    # ot = res.stdout.decode().strip()
    # FID = ot.split('  ')[1].split('\r')[0]
    # FID = float(FID)
    # plt.axhline(y=FID, color=col[cha], linestyle='-.', label='train_test_' + train_channel[cha], )
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('FID score')
    plt.show()
