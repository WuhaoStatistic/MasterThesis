import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random

p1 = './brats_5000_online_aug_t1ce_flair_seg_64/flair/'  # todo synthetic data
p2 = './brats_train_64/flair/'  # todo train data
p3 = './brats_test_64/flair/'  # todo test data
# p4 = './brats_every_5_original/seg/'
n_images = 1500  # how many images will be sampled from each dataset


# p4 = './brats_every_5_original/t1ce/'

# Load images for group A
def change_c(image):
    mean = np.mean(image)
    std = np.std(image)
    a = np.random.randint(0, 50)
    b = np.random.randint(0, 100)
    new_image = (image - mean) * (b / (std + 1e-4)) + a
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    return new_image


def bri(img):
    brightness_factor = np.random.randint(30)
    if np.random.rand() < 0.5:
        brightened_img = img + brightness_factor
    else:
        brightened_img = img - brightness_factor
    brightened_img = np.clip(brightened_img, 0, 255).astype(np.uint8)
    return brightened_img


group_a_images = os.listdir(p1)
random.shuffle(group_a_images)
# group_a_images = group_a_images[:n_images]
ga = []
count = 0
for i in group_a_images:  # load 10 images
    img = cv2.imread(p1 + i)
    if np.sum(img) < 1:
        continue
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat_img = gray_img.flatten()
    ga.append(flat_img)
    count += 1
    if count % n_images == 0:
        print('p1' + '_{}'.format(count))
        break

count = 0
# Load images for group B
group_b_images = os.listdir(p2)
random.shuffle(group_b_images)
# group_b_images = group_b_images[:n_images]
gb = []
for i in group_b_images:  # load 10 images
    img = cv2.imread(p2 + i)
    if np.sum(img) < 1:
        continue
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = bri(change_c(gray_img))  # todo data augmentation
    flat_img = gray_img.flatten()
    gb.append(flat_img)
    count += 1
    if count % n_images == 0:
        print('p2' + '_{}'.format(count))
        break

count = 0
group_c_images = os.listdir(p3)
random.shuffle(group_c_images)
# group_c_images = group_c_images[:n_images]
gc = []
for i in group_c_images:  # load 10 images
    img = cv2.imread(p3 + i)
    if np.sum(img) < 1:
        continue
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = bri(change_c(gray_img))  # todo data augmentation
    flat_img = gray_img.flatten()
    gc.append(flat_img)
    count += 1
    if count % n_images == 0:
        print('p3' + '_{}'.format(count))
        break

# count = 0
# group_d_images = os.listdir(p4)
# random.shuffle(group_d_images)
# gd = []
# for i in group_d_images:  # load 10 images
#     img = cv2.imread(p4 + i)
#     if np.sum(img) < 1:
#         continue
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_img = bri(gray_img)
#     flat_img = gray_img.flatten()
#     gd.append(flat_img)
#     count += 1
#     if count % n_images == 0:
#         print('p4' + '_{}'.format(count))
#         break

# Calculate correlation coefficients
count = 0
corr_coeffs = []
for a_img in ga:
    max_corr = -1.0
    for b_img in gb:
        corr_matrix = np.corrcoef(a_img + 1e-4, b_img + 1e-4)
        corr_coef = corr_matrix[0, 1]
        if corr_coef > 1 or corr_coef < -1:
            print(corr_coef)
        if corr_coef > max_corr:
            max_corr = corr_coef

    corr_coeffs.append(max_corr)
    count += 1
    if count % 1500 == 0:
        print('tr' + '_{}'.format(count))

count = 0
corr_coeffs2 = []
for a_img in ga:
    max_corr = -1.0
    for b_img in gc:
        corr_matrix = np.corrcoef(a_img + 1e-4, b_img + 1e-4)
        corr_coef = corr_matrix[0, 1]
        if corr_coef > 1 or corr_coef < -1:
            print(corr_coef)
        if corr_coef > max_corr:
            max_corr = corr_coef
    corr_coeffs2.append(max_corr)
    count += 1
    if count % 1500 == 0:
        print('te' + '_{}'.format(count))

count = 0
corr_coeffs3 = []
for a_img in gb:
    max_corr = -1.0
    for b_img in gc:
        corr_matrix = np.corrcoef(a_img + 1e-4, b_img + 1e-4)
        corr_coef = corr_matrix[0, 1]
        if corr_coef > 1 or corr_coef < -1:
            print(corr_coef)
        if corr_coef > max_corr:
            max_corr = corr_coef
    corr_coeffs3.append(max_corr)
    count += 1
    if count % 1500 == 0:
        print('te_tr' + '_{}'.format(count))

# count = 0
# corr_coeffs4 = []
# for a_img in ga:
#     max_corr = -1.0
#     for b_img in gd:
#         corr_matrix = np.corrcoef(a_img + 1e-4, b_img + 1e-4)
#         corr_coef = corr_matrix[0, 1]
#         if corr_coef > 1 or corr_coef < -1:
#             print(corr_coef)
#         if corr_coef > max_corr:
#             max_corr = corr_coef
#     corr_coeffs4.append(max_corr)
#     count += 1
#     if count % 1500 == 0:
#         print('eve5_' + '_{}'.format(count))

sns.set_style('darkgrid')
sns.set_palette('bright')
fig, ax = plt.subplots()

sns.kdeplot(corr_coeffs, fill=True, alpha=0.5, label='syn_train', ax=ax)
sns.kdeplot(corr_coeffs2, fill=True, alpha=0.3, label='syn_test', ax=ax)
sns.kdeplot(corr_coeffs3, fill=True, alpha=0.5, label='train_test', ax=ax)
# sns.kdeplot(corr_coeffs4, fill=True, alpha=0.5, label='syn_every5', ax=ax)
# Plot density distribution of correlation coefficients

ax.set_xlabel('The Highest Correlation Coefficient')
ax.set_ylabel('Frequency')
plt.xlim(0.3, 1.2)
# Show legend
plt.legend(loc='upper right')
# Show the plot
plt.show()
