import numpy as np
from torch.utils.data.dataset import Dataset
import os
import torchvision.transforms as T
import torch


# 256
# 64 48
def randomcrop(x):
    c = T.RandomCrop(size=(48, 48))
    b = T.RandomHorizontalFlip(0.4)
    v = T.RandomVerticalFlip(0.4)
    return v(b(c(x)))


def change_c(image):
    if np.random.rand() < 0.1:
        mean = np.mean(image)
        std = np.std(image)
        a = np.random.randint(0, 30)
        b = np.random.randint(0, 100)
        new_image = (image - mean) * (b / std) + a
        new_image = np.clip(new_image, 0, 255).astype(np.uint8)
        return new_image
    return image


def bri(img):
    brightness_factor = np.random.randint(20)
    if np.random.rand() < 0.2:
        brightened_img = img + brightness_factor
    else:
        brightened_img = img - brightness_factor
    brightened_img = np.clip(brightened_img, 0, 255).astype(np.uint8)
    return brightened_img


class ImageDataset(Dataset):
    def __init__(self, is_train=True, resolution=256, image_dir='..\\train_img_256\\', fake_img_dir=''):
        super().__init__()
        self.resolution = resolution
        self.train = is_train
        # Todo 2020
        image_dir = '..\\train_img_{}\\'.format(resolution) if self.train else '..\\test_img_{}\\'.format(resolution)
        # Todo 2021
        #image_dir = '..\\train_2021img_{}\\'.format(resolution) if self.train else '..\\test_2021img_{}\\'.format(
        #    resolution)
        #image_dir = '../train2021_every5_t1ce_flair_img_64/' if self.train else '..\\test_2021img_{}\\'.format(resolution)
        #image_dir = '../abcde/'
        all_path = os.listdir(image_dir)
        tot_true = len(all_path)
        # -------------------------------------------
        # change this manually

        self.fake = True  # Todo change the setting
        fakeimage_dir1 = '../difnpz_every_5_t1ce_64_nk/every5_{}/'.format(fake_img_dir)
        #fakeimage_dir2 = '../difnpz_sick_only_t1ce_64_nk/sick_{}/'.format(fake_img_dir)
        fake_rate = 0.20
        self.num_fake = int(fake_rate * tot_true)
        # -------------------------------------------
        self.weights = []
        if self.fake and fake_img_dir != '':
            fr = fake_rate
            # self.part_real_path = np.random.choice(self.all_path, tot_true - self.num_fake, replace=False)
            fake_path = os.listdir(fakeimage_dir1)
            fake_path = np.random.choice(fake_path, int(fr * tot_true), replace=False)
            f1 = [fakeimage_dir1 + x for x in fake_path if '.npz' in x]
            # self.all_path = t + f
            # fr = fake_rate * 0.5
            # fake_path = os.listdir(fakeimage_dir2)
            # fake_path = np.random.choice(fake_path, int(fr * tot_true), replace=False)
            ran_true_path = np.random.choice(all_path, tot_true - self.num_fake, replace=False)
            t = [image_dir + x for x in ran_true_path]
            # f2 = [fakeimage_dir2 + x for x in fake_path if '.npz' in x]
            self.all_path = t + f1
        else:
            self.all_path = [image_dir + x for x in all_path]
        if is_train:
            counts = {0: 0, 51: 0, 102: 0, 204: 0}
            n = tot_true // 2
            ran = np.random.choice(self.all_path, n, replace=False)
            for i in ran:
                data = np.load(i)
                seg_mask = data['seg']
                unique, counts_ = np.unique(seg_mask, return_counts=True)
                counts.update(dict(zip(unique, counts_)))
            self.weights = [n // 64 / counts[0], n // 64 / counts[51], n // 64 / counts[102], n // 64 / counts[204]]
            self.weights = [x if x < 40 else 30 for x in self.weights]

    def __len__(self):
        return len(self.all_path)

    def __getitem__(self, idx):
        path = self.all_path[idx]
        arrs = np.load(path)
        # Todo change the channel
        # t1_lst = torch.from_numpy(np.asarray(arrs['t1'], dtype=np.float32)[np.newaxis, ...])
        #if self.train:
         #   t1ce_lst = torch.from_numpy(bri(change_c(np.asarray(arrs['t1ce'], dtype=np.float32)))[np.newaxis, ...])
        #else:
        t1ce_lst = torch.from_numpy(np.asarray(arrs['t1ce'], dtype=np.float32)[np.newaxis, ...])
        #t2_lst = torch.from_numpy(np.asarray(arrs['t2'], dtype=np.float32)[np.newaxis, ...])
        #flair_lst = torch.from_numpy(np.asarray(arrs['flair'], dtype=np.float32)[np.newaxis, ...])
        seg_lst = torch.from_numpy(np.asarray(arrs['seg'], dtype=np.float32)[np.newaxis, ...])
        # -----------------------------
        # change this manually when use different channel
        con = torch.cat((
            t1ce_lst,
            #t2_lst,
            #flair_lst,
            seg_lst
        ))

        if self.train:
            con = randomcrop(con)
        con[:1, :, :] = con[:1, :, :] / 255.0
        return con[:1, :, :], con[1, :, :]
        # -----------------------------
