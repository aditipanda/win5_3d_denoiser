# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:44:26 2018

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

# the pixel value range is '0-1' of training data
# the pixel value range is '0-255'(uint8 ) of training data -- added on 12-11-17

# macro
NUM_OF_SLICES = 150  
NUM_OF_PATCHES = 529

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/Clean_Training_Data/18', help='dir of data')
parser.add_argument('--save_dir', dest='save_dir', default='./data/Clean_Training_Data/18/all_clean_3D_pats_patient_18', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=32, help='patch size')
args = parser.parse_args()

def generate_patches(isDebug=False):
#    name_array = np.arange(1,151,1)
    fname = "/all_clean_pats_patient_18/clean_pats_patient_18_slice_"
    patches_3d = np.zeros((NUM_OF_PATCHES, args.pat_size, args.pat_size, 3), dtype="uint8")
    output_temp = "clean_3D_pats_patient_18_slice_"
    
    #start loop from second slice
    for i in xrange(1,151):
        # j = i+1
        print("Slice no. " + str(i))
        path = args.src_dir + fname + str(i) + ".npy"
        print(path)
        curr_slice = load_data(filepath = path)
            
        if i == 1:
            print("first slice")
            path = args.src_dir + fname + str(i+2) + ".npy"     #print(path)
            prev_slice = load_data(filepath = path)           
                
            path = args.src_dir + fname + str(i+1) + ".npy"
            next_slice = load_data(filepath = path)
        
        elif i == 150:
            print("last slice")     
            path = args.src_dir + fname + str(i-1) + ".npy"     #print(path)
            prev_slice = load_data(filepath = path)
            
            path = args.src_dir + fname + str(i-2) + ".npy"
            next_slice = load_data(filepath = path)
        
        else:
            path = args.src_dir + fname + str(i-1) + ".npy"     #print(path)
            prev_slice = load_data(filepath = path)
            
            path = args.src_dir + fname + str(i+1) + ".npy"
            next_slice = load_data(filepath = path)
            
        patches_3d.fill(0)
        for k in xrange(NUM_OF_PATCHES):
            prev_patch = prev_slice[k, :, :, :]
#            prev_patch = np.array(prev_patch / 255.0, dtype=np.float32) # normalize the data to 0-1, line added for 12-11-17
#            print(prev_patch.shape)
            
#            ## display previous patch
            f = prev_patch[:,:,0]
#            print(f.shape)
#            f_img = PIL.Image.fromarray(f)
#            f_img.show()
            
            curr_patch = curr_slice[k, :, :, :]
#            curr_patch = np.array(curr_patch / 255.0, dtype=np.float32)
#            print(curr_patch.shape)
            
            ## display current patch
            g = curr_patch[:,:,0]
#            g_img = PIL.Image.fromarray(g)
#            g_img.show()
#            
            
            next_patch = next_slice[k, :, :, :]
#            next_patch = np.array(next_patch / 255.0, dtype=np.float32)
#            print(next_patch.shape)
            
            ## display next patch
            h = next_patch[:,:,0]
#            h_img = PIL.Image.fromarray(h)
#            h_img.show()            
            
#            
#            prev_patch = np.reshape(prev_patch, (args.pat_size, args.pat_size))
#            curr_patch = np.reshape(curr_patch, (args.pat_size, args.pat_size))
#            next_patch = np.reshape(next_patch, (args.pat_size, args.pat_size))
            
            patches_3d[k,:,:,0] = f
            patches_3d[k,:,:,1] = g
            patches_3d[k,:,:,2] = h
            
        print(patches_3d.shape)
            
        output_file_name = output_temp + str(i) + ".npy"
            
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        np.save(os.path.join(args.save_dir, output_file_name), patches_3d)
            

























if __name__ == '__main__':
    generate_patches()