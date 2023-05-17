import os
import numpy as np
import cv2

# source path
path = 'dif_samples/every_5/'  # Todo: change here
path2 = 'every5_35.npz'  # Todo: change here

targ_path = './difnpz_every_5_t1ce_64_nk/' + path2.split('.')[0]  # path here dont need '/' at the end

l = 18400  # Todo: change here
f = np.load(path + path2)['arr_0']

if not os.path.exists(targ_path):
    os.makedirs(targ_path)


# transfer to train img

def post_seg(x):
    discrete_values = np.array([0, 51, 102, 204])
    diff_matrix = np.abs(x[:, :, None] - discrete_values[None, None, :])
    index_matrix = np.argmin(diff_matrix, axis=2)
    dis_image = discrete_values[index_matrix]
    return dis_image


current = 0
for i in range(l):

    # Todo: change here
    # flair_img = f[i, :, :, 1]
    # t1_img = f[i, :, :, 0]
    t1ce_img = f[i, :, :, 0]
    # t2_img = f[i, :, :, 1]
    seg_img = post_seg(f[i, :, :, 1])
    np.savez_compressed(targ_path + '/f{}'.format(i), t1ce=t1ce_img,  # ,flair=flair_img,  # Todo: change here
                        seg=seg_img
                        )
    current += 1
    if current % 100 == 0:
        print('{}/{} finished'.format(current, l))

# transfer to brats

target_path = './brats_every_5_t1ce_64_nk/' + path2.split('.')[0] + '/'   # Todo: change here
# 0,1,2,3,4
# t1,t1ce,t2,flair,seg
channel = [1]  # Todo: change here
channel += [4]
dic = {0: 't1', 1: 't1ce', 2: 't2', 3: 'flair', 4: 'seg'}
for c in channel:
    if not os.path.exists(target_path + dic[c]):
        os.makedirs(target_path + dic[c])

for i in range(len(channel)):
    c = channel[i]
    p = target_path + dic[c] + '/'
    for j in range(l):
        if c != 4:
            cv2.imwrite(p + 'f{}'.format(j) + '.png', f[j, :, :, i])
        else:
            cv2.imwrite(p + 'f{}'.format(j) + '.png', post_seg(f[j, :, :, i]))

    print(dic[c] + '  finish ')
