#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 09:47:34 2018

@author: hhhhhhhhhh
"""
import numpy as np
import os.path as osp

def make_palette(num_classes):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    Takes:
        num_classes: the number of classes,
        value in the img array, if 0-20, then num_class should be 22, for0-21, 
        and config.classes == 20 , so should add 2 below
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    num_classes = num_classes+2
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in xrange(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette
    
    
# remove the 'module.' str before every sate_dict key
def get_new_dict(state_dict):
#    print state_dict.keys()
    for key in state_dict.keys():
        key_new = key[7:]
        state_dict[key_new] = state_dict.pop(key)
    return state_dict
    
def get_txt_weightbias(path):
    w  = np.loadtxt(osp.join(path, 'fc_weight.txt'))
    b  = np.loadtxt(osp.join(path, 'fc_bias.txt'))
    return w,  b
    
    
    
    
    