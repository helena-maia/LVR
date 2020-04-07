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

sys.path.insert(0, "../")
import models
from VideoSpatialPrediction import VideoSpatialPrediction

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='PyTorch Action Recognition - Test')
parser.add_argument('data_dir')
parser.add_argument('model_path')
parser.add_argument('--dataset', '-d', default='ucf101',
                    choices=["ucf101", "hmdb51"])
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('--architecture', '-a', default='inception_v3',
                    choices=["resnet152", "inception_v3"])
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    help='Batch size(default: 2)')

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z

def main():
    args = parser.parse_args()

    model_path = args.model_path
    data_dir = args.data_dir
 
    start_frame = 0
    num_samples = 10
    num_categories = 51 if args.dataset=='hmdb51' else 101
    
    model_start_time = time.time()

    new_size= 224
    if args.architecture == "inception_v3":
        new_size=299
        spatial_net = models.flow_inception_v3(pretrained=False, channels = 1, num_classes=num_categories)
    else:
        spatial_net = models.flow_resnet152(pretrained=False, channels = 1, num_classes=num_categories)

    params = torch.load(model_path)
    spatial_net.load_state_dict(params['state_dict'])
    spatial_net.cuda()
    spatial_net.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))

    val_file = "../dataset/settings/"+args.dataset+"/val_rgb_split%d.txt"%(args.split)

    f_val = open(val_file, "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))

    line_id = 1
    match_count = 0

    result = []
    for line in val_list:
        line_info = line.split(" ")
        clip_path = os.path.join(data_dir,line_info[0])
        num_frames = int(line_info[1])
        input_video_label = int(line_info[2])
        spatial_prediction = VideoSpatialPrediction(
                clip_path,
                spatial_net,
                num_categories,
                num_samples,
                new_size,
                args.batch_size
                )
        avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
        result.append(avg_spatial_pred_fc8)
        # avg_spatial_pred = softmax(avg_spatial_pred_fc8)

        pred_index = np.argmax(avg_spatial_pred_fc8)
        
        print("Rhythm split "+str(args.split)+", sample %d/%d: GT: %d, Prediction: %d ==> correct: %d" % (line_id, len(val_list), input_video_label, pred_index, match_count))

        if pred_index == input_video_label:
            match_count += 1
        line_id += 1

    print(match_count)
    print(len(val_list))
    print("Accuracy is : %4.4f" % ((float(match_count)/len(val_list))))
    np.save(args.dataset+"_rhythm_"+args.architecture+"_s"+str(args.split)+".npy", np.array(result))

if __name__ == "__main__":
    main()

