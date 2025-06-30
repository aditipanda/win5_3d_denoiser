# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:11:55 2018

@author: Aditi Panda
"""

import argparse
import glob
from PIL import Image
import PIL
import random
from utils import *
import six
from six.moves import xrange
import numpy as np
import os

# the pixel value range is '0-1' of training data
# the pixel value range is '0-255'(uint8 ) of training data -- added on 12-11-17

# macro
DATA_AUG_TIMES = 1  # transform a sample to a different sample for DATA_AUG_TIMES times, changed from 4 on 12-11-17

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/Clean_Training_Data/1', help='dir of data')
parser.add_argument('--save_dir', dest='save_dir', default='./data/Clean_Training_Data/1/all_clean_pats_patient_1/', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=32, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=10, help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--num_pic', dest='num_pic', type=int, default=10, help='number of pic to pick')
args = parser.parse_args()


def generate_patches(isDebug=False):
    global DATA_AUG_TIMES
    count = 0
    filepaths = glob.glob(args.src_dir + '/*.png')
    if isDebug:
        filepaths = filepaths[:10]

    scales = [1] # for WIN5 -- done on 8th Feb 2018
    output_name = "clean_pats_patient_1_slice_"
    
#    # data matrix 4-D
    inputs = np.zeros((529, args.pat_size, args.pat_size, 1), dtype="uint8")
    
    count = 0
    # name_array = np.arange(1,150,2)
    
    # generate patches
    for i in xrange(len(filepaths)): 	
    	print(i)    	
        img = Image.open(filepaths[i]).convert('L')
#        print(img.size)
#        img.show()
        
        for s in xrange(len(scales)):
            newsize = (int(img.size[0] * scales[s]), int(img.size[1] * scales[s]))
#            print(newsize)
            img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)
            img_s = np.reshape(np.array(img_s, dtype="uint8"),
                               (img_s.size[0], img_s.size[1], 1))  # extend one dimension
            
            for j in xrange(DATA_AUG_TIMES):
                im_h, im_w, _ = img_s.shape
                for x in range(0 + args.step, im_h - args.pat_size + 1, args.stride):
                    for y in range(0 + args.step, im_w - args.pat_size + 1, args.stride):
                        inputs[count, :, :, :] = data_augmentation(img_s[x:x + args.pat_size, y:y + args.pat_size, :], \
                                                                   random.randint(0, 7))                                        
#                        
#                        ## display current patch
#                        print(count)
#                        g = inputs[count, :, :, 0]
#                        img1 = PIL.Image.fromarray(g)
#                        img1.show()
                        
                        count += 1
                        
                        
        print("Another Image/Slice Done !")
        print(count)  
        fname = output_name + str(i+1)
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        np.save(os.path.join(args.save_dir, fname), inputs)
        inputs.fill(0)
        count = 0

    print(inputs.shape)                                      



if __name__ == '__main__':
    generate_patches()
