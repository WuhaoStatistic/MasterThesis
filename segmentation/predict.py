import torch
import models
from dataset import ImageDataset
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import dataloader
from torch import Tensor
import torch.nn.functional as F
import nibabel as nib
import os

# ------------------------------
# change this manually
t_fold = '../test_x_64'  # Todo change here
model_type = 'aug_every_5'  # Todo change here
# T1 T1CE T2 FLAIR 0,1,2,3
index = [1]  # Todo change here
shape = 64      # Todo change here
# ------------------------------
if not os.path.exists('../seg_res/' + model_type):
    os.mkdir('../seg_res/' + model_type)
file_list = os.listdir(t_fold)
file_list.sort()
model_unet = models.UNet(len(index), 4).to('cuda')
state_dict = torch.load('../segmodel/' + model_type + '.pt', map_location='cuda')
model_unet.load_state_dict(state_dict)
i = 0
model_unet.eval()

for i in file_list:
    img = np.load(os.path.join(t_fold, i))['x'][:, index, :, :] / 255.0
    volume = np.zeros((shape, shape, img.shape[0]))
    batch = torch.from_numpy(img)
    for n in range(img.shape[0]):
        with torch.no_grad():
            x = batch[n, :, :, :].unsqueeze(0).to(device='cuda', dtype=torch.float32)
            output = model_unet(x).cpu()
            mask = output.argmax(dim=1)
        mask = mask.squeeze().numpy() / 1.0
        mask *= 51
        mask[mask == 153] = 204
        volume[:, :, n] = mask
    nifti_image = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(nifti_image, "../seg_res/" + model_type + "/" + i.split('.')[0] + ".nii.gz")
