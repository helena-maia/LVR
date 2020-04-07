#!/usr/bin/env python

import os, sys
import collections
import numpy as np
import argparse
import cv2
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as tv_models

sys.path.insert(0, "../")
import models
#import models.inception_features as i_features
from VideoSpatialPrediction import VideoSpatialPrediction

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition - Test')
parser.add_argument('--dataset', '-d', metavar='DATASET', default='ucf101',
                    choices=["ucf101", "hmdb51"])
parser.add_argument('--data_dir', '-f', default='/home/Datasets/UCF-101-OF_CPU')
parser.add_argument('--architecture', '-a', metavar='ARCHITECTURE', default='inception_v3',
                    choices=["resnet152", "inception_v3"])
parser.add_argument('--list_files', '-l', default='spatial_testlist01_with_labels.txt')
parser.add_argument('--ext_batch_sz', '-e', default=100) #CPU
parser.add_argument('--int_batch_sz', '-i', default=5) #GPU
parser.add_argument('--start_instance', '-s', default=0)
parser.add_argument('--end_instance', '-t', default=-1)
parser.add_argument('--checkpoint_path','-c', type=str, default="")

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z

def normalize_maxmin(mat):
    norm_mat = [ line-line.min() for line in mat]
    norm_mat = [ line/line.max() for line in norm_mat]
    
    return np.array(norm_mat)

def main():
    args = parser.parse_args()

    data_dir = args.data_dir
    val_file = args.list_files
    ext_batch_sz = int(args.ext_batch_sz)
    int_batch_sz = int(args.int_batch_sz)
    start_instance = int(args.start_instance)
    end_instance = int(args.end_instance)
    checkpoint = args.checkpoint_path
    
    model_start_time = time.time()
    if args.architecture == "inception_v3":
        new_size=299
        num_categories = 3528,3468,2048
        spatial_net = models.inception_v3(pretrained=(checkpoint==""), num_outputs=len(num_categories))
    else: #resnet
        new_size= 224
        num_categories = 8192,4096,2048
        spatial_net = models.resnet152(pretrained=(checkpoint==""), num_outputs=len(num_categories))

    if os.path.isfile(checkpoint):    
        print('loading checkpoint {} ...'.format(checkpoint))    
        params = torch.load(checkpoint)
        model_dict = spatial_net.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in params['state_dict'].items() if k in model_dict}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        spatial_net.load_state_dict(model_dict)
        print('loaded')
    else:
        print(checkpoint)
        print('ERROR: No checkpoint found')

    spatial_net.cuda()
    spatial_net.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))

    f_val = open(val_file, "r")
    val_list = f_val.readlines()[start_instance:end_instance]
    print("we got %d test videos" % len(val_list))

    line_id = 1
    match_count = 0

    for line_id, line in enumerate(val_list):
        print("sample %d/%d" % (line_id+1, len(val_list)))
        line_info = line.split(" ")
        clip_path = os.path.join(data_dir,line_info[0])
        num_frames = int(line_info[1])
        input_video_label = int(line_info[2])

        spatial_prediction = VideoSpatialPrediction(
                clip_path,
                spatial_net,
                num_categories,
                num_frames,
                ext_batch_sz,
                int_batch_sz,
                new_size
                )

        for ii in range(len(spatial_prediction)):
            for vr_ind,vr in enumerate(spatial_prediction[ii]):
                folder_name = args.architecture + "_" + args.dataset + "_VR" + str(ii)
                if not os.path.isdir(folder_name+'/'+line_info[0]):
                    print("creating folder: "+folder_name+"/"+line_info[0])
                    os.makedirs(folder_name+"/"+line_info[0])
                vr_name = folder_name+'/'+line_info[0]+'/vr_{0:02d}.png'.format(vr_ind)
                vr_gray = normalize_maxmin(vr.transpose()).transpose()*255.
                cv2.imwrite(vr_name, vr_gray)

if __name__ == "__main__":
    main()
