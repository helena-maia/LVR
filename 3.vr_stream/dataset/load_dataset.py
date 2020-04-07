import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2
from pipes import quote

def make_dataset(root, source):
    '''
        Method to obtain the path, duration(number of frame) and target of each
        video of the dataset from a file(train or test) that contain this detalls
        in each line.
    '''
    if not os.path.exists(source):
        print('Setting file %s for the dataset doesnt exist.' % (source))
        sys.exit()
    else:
        clips = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                clip_path = os.path.join(root, line_info[0])
                duration = int(line_info[1])
                target = int(line_info[2])
                item = (clip_path, duration, target)
                clips.append(item)
    return clips

def color(is_color):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0    
    return cv_read_flag
    
def read_single_segment(path, offsets, new_height, new_width, is_color, name_pattern, index):
    '''
        Takes visual_rhythm, history_motion or RGB images, one frame by video,
        and this correspond to some specific images(type of visual_rhythm or
        history_motion). 
    '''
    cv_read_flag = color(is_color)
    interpolation = cv2.INTER_LINEAR    
    sampled_list = []

    for offset in offsets:
        frame_name = name_pattern % (index+offset)
        frame_path = os.path.join(path, frame_name)
        cv_img_origin = cv2.imread(frame_path, cv_read_flag)
        if cv_img_origin is None:
            print('Could not load file %s' % (frame_path))
            sys.exit()
            # TODO: error handling here
        if new_width > 0 and new_height > 0:
            # use OpenCV3, use OpenCV2.4.13 may have error
            cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
        else:
            cv_img = cv_img_origin

        sampled_list.append(np.expand_dims(cv_img, 2))

    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input 

class dataset(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 name_pattern=None,
                 is_color=True,
                 n_images=1,
                 new_width=0,
                 new_height=0,
                 transform=None,
                 target_transform=None,
                 video_transform=None):
        clips = make_dataset(root, source)

        if len(clips) == 0:
            raise(RuntimeError('Found 0 video clips in subfolders of: ' + root + '\n'
                               'Check your data directory.'))

        self.root = root
        self.source = source
        self.phase = phase
        self.dataset = source.split('/')[3]

        self.clips = clips
        self.direction =[]
        if name_pattern:
            self.name_pattern = name_pattern
        else:
            self.name_pattern = 'vr_%02d.png'

        self.is_color = is_color
        self.n_images = n_images
        self.new_width = new_width
        self.new_height = new_height
        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):
        path, duration, target = self.clips[index]

        offsets = list(range(self.n_images))
        target = np.array([target]*self.n_images)

        
        clip_input = read_single_segment(path, offsets, self.new_height, self.new_width,
                                         self.is_color, self.name_pattern, 0)            

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)

        return clip_input, target


    def __len__(self):
        return len(self.clips)
