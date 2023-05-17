import torch
import models
from dataset import ImageDataset
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import dataloader
from torch import Tensor
import torch.nn.functional as F

dataset = ImageDataset(is_train=False, resolution=64)
modelu = models.UNet(1, 4).to('cuda')
state_dict = torch.load('../segmodel/every_5_35_005.pt', map_location='cuda')
modelu.load_state_dict(state_dict)

mask_l = []
y_l = []


def print_res(net=modelu, device='cuda'):
    tag = True
    while tag:
        j = np.random.randint(0, 3353, size=100)
        if len(np.unique(j)) == 100:
            tag = False
    count = 0
    for i in range(100):
        x, y = dataset.__getitem__(j[i])
        if len(np.unique(y)) < 2:
            continue
        net.eval()
        with torch.no_grad():
            x = x.to(device=device, dtype=torch.float32)
            output = net(x.unsqueeze(0)).cpu()
            mask = output.argmax(dim=1)
        mask = mask.squeeze().numpy()
        y = y.numpy()
        mask *= 51
        mask[mask == 153] = 204
        plt.imsave('../256/y{}.png'.format(count), y, cmap='gray')
        plt.imsave('../256/m{}.png'.format(count), mask, cmap='gray')
        count += 1
        if count ==15:
            break


print_res()
