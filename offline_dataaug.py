import numpy as np
import os


def change_c(image):
    mean = np.mean(image)
    std = np.std(image)
    a = np.random.randint(0, 30)
    b = np.random.randint(0, 120)
    new_image = (image - mean) * (b / std) + a
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    return new_image


def bri(img):
    brightness_factor = np.random.randint(20)
    if np.random.rand() < 0.5:
        brightened_img = img + brightness_factor
    else:
        brightened_img = img - brightness_factor
    brightened_img = np.clip(brightened_img, 0, 255).astype(np.uint8)
    return brightened_img


# flip contract brightness
def flip(x1, x2,
         # x3,
         # x4, x5,
         p=0.4, pc=0.5, pb1=0.5):
    if np.random.rand() < p:
        x1 = np.flipud(x1)
        x2 = np.flipud(x2)
        # x3 = np.flipud(x3)
        # x4 = np.flipud(x4)
        # x5 = np.flipud(x5)
    if np.random.rand() < p:
        x1 = np.fliplr(x1)
        x2 = np.fliplr(x2)
        # x3 = np.fliplr(x3)
        # x4 = np.fliplr(x4)
        # x5 = np.fliplr(x5)
    if np.random.rand() < pc:
        x1 = change_c(x1)
        #x2 = change_c(x2)
        # x3 = change_c(x3)
        # x4 = change_c(x4)
    if np.random.rand() < pb1:
        x1 = bri(x1)
        #x2 = bri(x2)
        # x3 = bri(x3)
        # x4 = bri(x4)
    return x1, x2  # x3,  # x4, #x5


path1 = './train_augimg2021_sick_64/'
pathlst = os.listdir(path1)
tot = len(pathlst)
path2 = './train_sick/'
if not os.path.exists(path2):
    os.mkdir(path2)
cont = 0
for npz in pathlst:
    arrs = np.load(path1 + npz)
    # t1_lst = np.asarray(arrs['t1'], dtype=np.float32)
    t1ce_lst = np.asarray(arrs['t1ce'], dtype=np.float32)
    # t2_lst = np.asarray(arrs['t2'], dtype=np.float32)
    # flair_lst = np.asarray(arrs['flair'], dtype=np.float32)
    seg_lst = np.asarray(arrs['seg'], dtype=np.float32)
    t1ce_lst, seg_lst = flip(t1ce_lst, seg_lst)

    # t1_lst, t1ce_lst, t2_lst, flair_lst, seg_lst = flip(t1_lst, t1ce_lst, t2_lst, flair_lst, seg_lst)
    # -----------------------------
    # change this manually when use different channel
    con = np.concatenate((
        # t1_lst[np.newaxis, ...],
        t1ce_lst[np.newaxis, ...],
        # t2_lst[np.newaxis, ...],
        # flair_lst[np.newaxis, ...],
        seg_lst[np.newaxis, ...]
    ))

    np.savez_compressed(path2 + npz,
                        # t1=con[0, :, :],
                        t1ce=con[0, :, :],
                        # t2=con[2, :, :],
                        # flair=con[1, :, :],
                        seg=con[1, :, :]
                        )
    cont += 1
    if cont % 5000 == 0:
        print('{}/{} finished'.format(cont, tot))
