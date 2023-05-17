import os
import numpy as np
counts = {0: 0, 51: 0, 102: 0, 204: 0}
fake_dir = './difnpz_LNUS_t1ce_seg_64_nk/LNUS_aug_t1ce_seg_64_500k'

# old = np.array([0.27163427, 10.66462364, 6.64531657, 13.45360825])*21000
for i in range(5000):
    # load the ith NPZ file
    data = np.load(fake_dir+'/f{}.npz'.format(i))
    seg_mask = data['seg']
    unique, counts_ = np.unique(seg_mask, return_counts=True)
    counts.update(dict(zip(unique, counts_)))

total_pixels = sum(counts.values())
# calculate the ratio of each pixel value
ratios = {k: v / total_pixels for k, v in counts.items()}
print(ratios)