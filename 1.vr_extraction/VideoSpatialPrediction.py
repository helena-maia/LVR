'''
A sample function for classification using spatial network
Customize as needed:
e.g. num_categories, layer for feature extraction, batch_size
'''

import os
import numpy as np
import math
import cv2

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import video_transforms


def VideoSpatialPrediction(
        vid_name,
        net,
        num_categories,
        num_frames=0,
        ext_batch_sz=100,
        int_batch_sz=5,
        new_size = 299
        ):

    if num_frames == 0:
        imglist = os.listdir(vid_name)
        duration = len(imglist)
    else:
        duration = num_frames  

    clip_mean = [0.485, 0.456, 0.406]
    clip_std = [0.229, 0.224, 0.225]

    normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)

    val_transform = video_transforms.Compose([
            video_transforms.ToTensor(),
            normalize,
        ])

    deep = 3

    # inception = 320,360, resnet = 240, 320
    width = 320 if new_size==299 else 240
    height = 360 if new_size==299 else 320
    predictions = []
    for i in range(len(num_categories)):
        predictions.append(np.zeros((num_categories[i],num_frames*10)))

    #control memory (RAM) usage
    num_ext_batch = int(math.ceil(float(num_frames)/ext_batch_sz))
   
    for i in range(num_ext_batch):
        start = i*ext_batch_sz
        end = min(start+ext_batch_sz, num_frames)

        dims = (width,height,deep,end-start)
        rgb = np.zeros(shape=dims, dtype=np.float64)
        rgb_flip = np.zeros(shape=dims, dtype=np.float64)

        for j in range(end-start):
            img_file = os.path.join(vid_name, 'img_{0:05d}.jpg'.format(j+start+1))
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, dims[1::-1])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb[:,:,:,j] = img
            rgb_flip[:,:,:,j] = img[:,::-1,:]

        # crop 299 = inception, 224 = resnet
        size = new_size
        corner = [(height-size)//2, (width-size)//2]
        rgb_1 = rgb[:size, :size, :,:]
        rgb_2 = rgb[:size, -size:, :,:]
        rgb_3 = rgb[corner[1]:corner[1]+size, corner[0]:corner[0]+size, :,:]
        rgb_4 = rgb[-size:, :size, :,:]
        rgb_5 = rgb[-size:, -size:, :,:]
        rgb_f_1 = rgb_flip[:size, :size, :,:]
        rgb_f_2 = rgb_flip[:size, -size:, :,:]
        rgb_f_3 = rgb_flip[corner[1]:corner[1]+size, corner[0]:corner[0]+size, :,:]
        rgb_f_4 = rgb_flip[-size:, :size, :,:]
        rgb_f_5 = rgb_flip[-size:, -size:, :,:]

        rgb = np.concatenate((rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,rgb_f_1,rgb_f_2,rgb_f_3,rgb_f_4,rgb_f_5), axis=3)

        rgb_1, rgb_2, rgb_3, rgb_4, rgb_5 = [],[],[],[],[]
        rgb_f_1, rgb_f_2, rgb_f_3, rgb_f_4, rgb_f_5 = [],[],[],[],[]
        rgb_flip = []

        _, _, _, c = rgb.shape
        rgb_list = []
        for c_index in range(c):
            cur_img = rgb[:,:,:,c_index]
            cur_img_tensor = val_transform(cur_img)
            rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
        
        rgb_shape = rgb.shape
        rgb = []

        rgb_np = np.concatenate(rgb_list,axis=0)

         #control memory (GPU) usage
        num_int_batches = int(math.ceil(float(rgb_shape[3])/int_batch_sz))

        rgb_list = []

        for bb in range(num_int_batches):
            span = range(int_batch_sz*bb, min(rgb_shape[3],int_batch_sz*(bb+1)))
            input_data = rgb_np[span,:,:,:]
            imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
            imgDataVar = torch.autograd.Variable(imgDataTensor)
            output = net(imgDataVar)

            for ii in range(len(output)):
                output_ = output[ii].reshape(-1, num_categories[ii])
                result = output_.data.cpu().numpy()
                pos = [ x%(end-start) + start + int(x/(end-start))*num_frames  for x in span ]
                predictions[ii][:, pos] = np.transpose(result)

        rgb_np = []

    result = []
    for ii in range(len(predictions)):
        result.append(np.split(predictions[ii],10,axis=1))

    return result
