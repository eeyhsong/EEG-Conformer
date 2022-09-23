"""
Class activation topography (CAT) for EEG model visualization, combining class activity map and topography
Code: Class activation map (CAM) and then CAT

refer to high-star repo on github: 
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/grad_cam

Salute every open-source researcher and developer!
"""


import argparse
import os
gpus = [1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
from torch.backends import cudnn
# from tSNE import plt_tsne
# from grad_cam.utils import GradCAM, show_cam_on_image
from utils import GradCAM, show_cam_on_image

cudnn.benchmark = False
cudnn.deterministic = True


# keep the overall model class, omitted here
class ViT(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(
            # ... the model
        )

# test the class activity map of convolutional layer 
# without einops 
data = np.load('./grad_cam/train_data.npy')  
test = torch.as_tensor(data[:1, :, :, :], dtype=torch.float32)  # # the data used for visualization, which will be processed by saved model
test = torch.autograd.Variable(test, requires_grad=True)
print(np.shape(data))


device = torch.device("cpu")
model = ViT()
model.load_state_dict(torch.load('./model/model_cnn.pth', map_location=device))
target_layers = [model[0].projection]  # set the layer you want to visualize, you can use torchsummary here to find the layer index
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

target_category = 2  # set the class (class activation mapping)
grayscale_cam = cam(input_tensor=test)


nptest = test.cpu().detach().numpy()
nptest = (nptest - np.min(nptest)) / (np.max(nptest) - np.min(nptest))  # use normal or standard for better plot
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(nptest, grayscale_cam, use_rgb=True)


plt.figure(figsize=(6, 0.5), dpi=1200)
plt.imshow(grayscale_cam, aspect='auto', cmap='RdBu_r')
plt.colorbar()


# test the class activity map of transformer layer 

# ! A crucial step for adaptation on Transformer
# reshape_transform  b 61 40 -> b 40 1 61
def reshape_transform(tensor):
    result = rearrange(tensor, 'b (h w) e -> b e (h) (w)', h=1)
    return result


device = torch.device("cpu")
model = ViT()
model.load_state_dict(torch.load('./model/sub%d.pth'%nSub, map_location=device))
target_layers = [model[1]]  # set the target layer 

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_transform)

grayscale_cam = cam(input_tensor=test)

# nptest = test.cpu().detach().numpy()
# nptest = (nptest - np.min(nptest)) / (np.max(nptest) - np.min(nptest))
grayscale_cam = grayscale_cam[0, :]


plt.figure(figsize=(6, 0.8), dpi=600)
plt.imshow(grayscale_cam, aspect='auto', cmap='RdBu_r')
cb = plt.colorbar(aspect=10)
cb.set_ticks([0, 0.5, 1])

plt.xticks([])
plt.xlabel('Time')
plt.yticks([])
plt.ylabel('Channel')



# TODO: Class Activation Topography (proposed in the paper)
import mne
from matplotlib import mlab as mlab

biosemi_montage = mne.channels.make_standard_montage('biosemi64')
index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56, 29]
biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
biosemi_montage.dig = [biosemi_montage.dig[i+3] for i in index]
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')


all_cam = []
# this loop is used to obtain the cam of each trial/sample
for i in range(288):
    test = torch.as_tensor(data[i:i+1, :, :, :], dtype=torch.float32)
    test = torch.autograd.Variable(test, requires_grad=True)

    grayscale_cam = cam(input_tensor=test)
    grayscale_cam = grayscale_cam[0, :]
    all_cam.append(grayscale_cam)


# the mean of all data
test_all_data = np.squeeze(np.mean(data, axis=0))
# test_all_data = (test_all_data - np.mean(test_all_data)) / np.std(test_all_data)
mean_all_test = np.mean(test_all_data, axis=1)

# the mean of all cam
test_all_cam = np.mean(all_cam, axis=0)
# test_all_cam = (test_all_cam - np.mean(test_all_cam)) / np.std(test_all_cam)
mean_all_cam = np.mean(test_all_cam, axis=1)

# apply cam on the input data
hyb_all = test_all_data * test_all_cam
# hyb_all = (hyb_all - np.mean(hyb_all)) / np.std(hyb_all)
mean_hyb_all = np.mean(hyb_all, axis=1)

evoked = mne.EvokedArray(test_all_data, info)
evoked.set_montage(biosemi_montage)

fig, [ax1, ax2, ax3] = plt.subplots(nrows=3)

# print(mean_all_test)
plt.subplot(311)
im1, cn1 = mne.viz.plot_topomap(mean_all_test, evoked.info, show=False, axes=ax1, res=1200)

plt.subplot(312)
# draw EEG data in a circle type, color corresponds to value
size = np.ones(1000)
cam_bar = np.mean(test_all_cam, axis=0)
cycle_cam = (cam_bar - np.min(cam_bar)) / (np.max(cam_bar) - np.min(cam_bar))
plt.pie(size, colors=plt.cm.RdBu_r(cycle_cam), wedgeprops=dict(width=0.2), startangle=90, counterclock=False)

plt.subplot(313)
im3, cn3 = mne.viz.plot_topomap(mean_hyb_all, evoked.info, show=False, axes=ax3)



