import argparse
import numpy as np
import cv2
import os

import torch
import torch.nn as nn

def vertical_pooling(model, img):
    img = np.swapaxes(img,0,2) #swap height (0) and channels (2)
    img = torch.from_numpy(img).float() #npy to tensor
    
    output = model(img)
    
    output = np.array(output) #tensor to npy
    output = np.swapaxes(output,0,2)
    
    return output

def horizontal_simmetric_extension(img, new_width):
    width = img.shape[1]

    new_shape = (img.shape[0], new_width, img.shape[2])
    output = np.zeros(new_shape, dtype=img.dtype)

    for i in range(new_width):
        if int(i / width)%2 == 0 : pos = i%width
        else: pos = width - i%width - 1

        output[:,i,:] = img[:,pos,:]

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-src_dir', type=str, help='')
    parser.add_argument('--list_files', '-l', default='dataset_list_ucf.txt')
    parser.add_argument('-new_width', default=299, type=int, help='')
    parser.add_argument('--w_method', '-wm', default='resize',
                    choices=["none","sim_ext", "resize"])
    parser.add_argument('-new_height', default=299, type=int, help='')
    parser.add_argument('--h_method', '-hm', default='pool',
                    choices=["none","pool"]) #TODO other methods
    parser.add_argument('-output_dir', type=str, help='')
    parser.add_argument('-ext', type=str,default='png', help='')
    args = parser.parse_args()

    model = nn.AdaptiveAvgPool1d(args.new_height)
    in_fmt = "vr_%02d."+args.ext

    video_list = np.loadtxt(args.list_files, delimiter=" ", comments="$", dtype='U200')[:,0]

    if not os.path.isdir(args.output_dir):
        print("creating folder: "+args.output_dir)
        os.makedirs(args.output_dir)

    for v in video_list:
        video_path = os.path.join(args.src_dir,v)
        out_path = os.path.join(args.output_dir,v)

        if not os.path.isdir(out_path):
            print("creating folder: "+out_path)
            os.makedirs(out_path)

        for i in range(10):
            vr_path = os.path.join(video_path, in_fmt%i)
            out_vr_path = os.path.join(out_path, in_fmt%i)
            img = cv2.imread(vr_path)

            output = img.copy()

            if args.h_method == 'pool':
                output = vertical_pooling(model, output)

            if args.w_method == 'resize':
                output = cv2.resize(output,(args.new_width,output.shape[0]))
            elif args.w_method == 'sim_ext':
                output = horizontal_simmetric_extension(output, args.new_width)

            cv2.imwrite(out_vr_path,output)
