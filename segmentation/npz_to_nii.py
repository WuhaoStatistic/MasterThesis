import os
import numpy as np
import nibabel as nib
import imageio.v2 as imageio
import cv2

# input_folder = "../brats_2021_test/Seg"
#
# output_file = ".nii.gz"
# file_list = os.listdir(input_folder)
# file_list.sort()
# first_image_path = os.path.join(input_folder, file_list[0])
# first_image = imageio.imread(first_image_path)
# image_shape = first_image.shape
#
# ln = file_list[0].split('_')[1]
# i = 0
# while i < len(file_list):
#     volume = []
#     cn = file_list[i].split('_')[1]
#     j = 0
#     while cn == ln:
#         filepath = os.path.join(input_folder, file_list[i])
#         image = imageio.imread(filepath)
#         image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_NEAREST)
#         volume.append(image[np.newaxis, ...])
#         i += 1
#         if i == len(file_list):
#             break
#         j += 1
#         cn = file_list[i].split('_')[1]
#     volume = np.concatenate(volume, axis=0)
#     volume = volume.transpose((1, 2, 0))
#     nifti_image = nib.Nifti1Image(volume, affine=np.eye(4))
#     o = ln + output_file
#     nib.save(nifti_image, "../test_2021label_64/" + o)
#     print(ln, j)
#     ln = cn

input_folder = "../test_2021img_64"

file_list = os.listdir(input_folder)
file_list.sort()
ln = file_list[0][:4]
i = 0
while i < len(file_list):
    cn = file_list[i][:4]
    j = 0
    img_list = []
    while cn == ln:
        filepath = os.path.join(input_folder, file_list[i])
        arrs = np.load(filepath)
        t1_lst = np.asarray(arrs['t1'], dtype=np.float32)[np.newaxis, ...]
        t1ce_lst = np.asarray(arrs['t1ce'], dtype=np.float32)[np.newaxis, ...]
        t2_lst = np.asarray(arrs['t2'], dtype=np.float32)[np.newaxis, ...]
        flair_lst = np.asarray(arrs['flair'], dtype=np.float32)[np.newaxis, ...]
        img_list.append(np.concatenate((t1_lst, t1ce_lst, t2_lst, flair_lst))[np.newaxis, ...])
        i += 1
        if i == len(file_list):
            break
        j += 1
        cn = file_list[i][:4]
    o = np.concatenate(img_list)
    np.savez_compressed('../test_2021x_64/' + ln, x=o)
    ln = cn
