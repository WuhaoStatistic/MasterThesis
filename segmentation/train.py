import os
import torch
from dataset import ImageDataset
import models
import layers
from torch.optim import Adam, lr_scheduler
import numpy as np
import cv2
from tqdm import tqdm
import gc
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as T

dev = 'cuda'
reso = 64
dataset = ImageDataset(resolution=reso, fake_img_dir='35')  # todo change here
valid_data = ImageDataset(is_train=False, resolution=reso)
# CLASS_WEIGHT = np.array([1, 1, 1, 1])
CLASS_WEIGHT = np.array(dataset.weights)
print(CLASS_WEIGHT)
CLASS_WEIGHT = torch.from_numpy(CLASS_WEIGHT[None, :, None, None]).to(dev)

# T1 T1CE T2 FLAIR 0,1,2,3
# todo valid
batch_size = 256  # in total 732 groups of data
valid_iter = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=True, drop_last=True)

modelu = models.UNet(1, 4).to(dev)
modelpath = '../segmodel/every_5_35_020.pt'  # Todo change this
color_map = [0, 51, 102, 204]
classes = ['bk', 'ed', 'ncr', 'et']
# todo train
batch_size = 256  # in total 732 groups of data
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

opt = Adam(modelu.parameters(), weight_decay=1e-7, lr=4e-3)
sch = lr_scheduler.CosineAnnealingLR(opt, T_max=500,
                                     eta_min=5e-4)
grad_scaler = torch.cuda.amp.GradScaler(enabled=False)


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6
               ):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes

    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


# def criterion(res, label):
#     label = one_hot(label)
def weighted_dice(input, target, classweight=CLASS_WEIGHT):
    nu = (input * target * classweight).sum() + 1e-5
    de = ((input + target) * classweight).sum() + 1e-5
    return 1 - 2 * nu / de


def train_one_epoch(epoch, model=modelu, optimizer=opt, scheduler=sch, dataloader=train_iter, device=dev):
    model.train()

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data[0].to(device, dtype=torch.float32)
        labels = data[1].to(device, dtype=torch.long)

        labels[labels == 204] = 153
        labels //= 51
        with torch.autocast(device, enabled=False):
            outputs = model(images)
            # print(outputs.shape)  # batch, n_class, h=input_h, w=input_w
            # target batch, input_h, input_w
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            # x = F.softmax(outputs, dim=1).float()
            # print(x.min())
            #
            # dl = dice_loss(
            #     F.softmax(outputs, dim=1).float(),
            #     F.one_hot(labels.long(), 4).permute(0, 3, 1, 2).float(),
            #     multiclass=True
            # )
            dl = weighted_dice(F.softmax(outputs, dim=1).float(),
                               F.one_hot(labels.long(), 4).permute(0, 3, 1, 2).float())
            if dl < 0:
                print(dl)
            loss += dl
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
        bar.set_postfix(Epoch=epoch, Train_Loss=loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    return 1


@torch.inference_mode()
def valid_one_epoch(epoch, model=modelu, dataloader=valid_iter, device=dev):
    model.eval()
    loss = 0
    iter = 1
    for i in range(iter):
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:
            images = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.long)
            labels[labels == 204] = 153
            labels //= 51
            outputs = model(images)
            # print(outputs.shape)  # batch, n_class, h=input_h, w=input_w
            loss += dice_loss(
                F.softmax(outputs, dim=1).float(),
                F.one_hot(labels.long(), 4).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
            bar.set_postfix(Epoch=epoch, valid_Loss=loss)
    gc.collect()

    torch.cuda.empty_cache()
    return loss / iter


def run_train(tot_ep):
    tot_best_loss = 100
    p = modelu.state_dict()
    for i in range(tot_ep):
        train_one_epoch(i)
        if i % 4 == 0 and i > 10:
            loss = valid_one_epoch(i)
            print('valid loss {} at epoch {}'.format(loss, i))
            if loss < tot_best_loss:
                tot_best_loss = loss
                torch.save(p, modelpath)


def resume(already, another_epoch, path):
    tot_best_loss = 0.421
    modelu.load_state_dict(torch.load(path, map_location=dev))
    opt = Adam(modelu.parameters(), weight_decay=1e-7, lr=2.5e-4)
    sch = lr_scheduler.CosineAnnealingLR(opt, T_max=500,
                                         eta_min=1e-4)
    for i in range(already, another_epoch + already):
        train_one_epoch(i, optimizer=opt, scheduler=sch)
        if i % 4 == 0 and i > 10:
            loss = valid_one_epoch(i)
            print('valid loss {} at epoch {}'.format(loss, i))
            if loss < tot_best_loss:
                tot_best_loss = loss
                torch.save(modelu.state_dict(), modelpath)


# resume(36, 73, '../segmodel/2020_64_aug_t1ce_real.pt')
run_train(100)
