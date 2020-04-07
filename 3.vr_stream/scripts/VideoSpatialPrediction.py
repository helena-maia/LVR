'''
A sample function for classification using spatial network
Customize as needed:
e.g. num_categories, layer for feature extraction, batch_size
'''

import os
import sys
import numpy as np
import math
import cv2
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

sys.path.insert(0, "../../")
import video_transforms


def VideoSpatialPrediction(
        vid_name,
        net,
        num_categories,
        num_samples=25,
        new_size = 299,
        batch_size = 2
        ):

    clip_mean = [0.5]*num_samples
    clip_std = [0.226]*num_samples

    normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
    val_transform = video_transforms.Compose([
            video_transforms.ToTensor(),
            normalize,
        ])

    deep = 1

    # inception = 299,299, resnet = 224,224
    dims = (new_size,new_size,deep,num_samples)
    rgb = np.zeros(shape=dims, dtype=np.float64)
    rgb_flip = np.zeros(shape=dims, dtype=np.float64)
   
    for i in range(num_samples):
        img_file = os.path.join(vid_name, 'vr_{0:02d}.png'.format(i))
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)   
        rgb[:,:,0,i] = img
        rgb_flip[:,:,0,i] = img[:,::-1]    

    _, _, _, c = rgb.shape
    rgb_list = []
    for c_index in range(c):
        cur_img = rgb[:,:,:,c_index]
        cur_img_tensor = val_transform(cur_img)
        rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
        
    rgb_np = np.concatenate(rgb_list,axis=0)

    prediction = np.zeros((num_categories,rgb.shape[3]))
    num_batches = int(math.ceil(float(rgb.shape[3])/batch_size))

    for bb in range(num_batches):
        span = range(batch_size*bb, min(rgb.shape[3],batch_size*(bb+1)))
        input_data = rgb_np[span,:,:,:]
        imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
        imgDataVar = torch.autograd.Variable(imgDataTensor)
        output = net(imgDataVar)
        result = output.data.cpu().numpy()
        prediction[:, span] = np.transpose(result)

    return prediction
