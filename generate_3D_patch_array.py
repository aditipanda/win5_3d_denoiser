# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:55:06 2018

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

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/Clean_Training_Data/', help='dir of data')
parser.add_argument('--save_dir', dest='save_dir', default='./data/Clean_Training_Data/', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=32, help='patch size')
args = parser.parse_args()

selected_patients = [1, 3, 8, 12, 18] ## randomly select 10 patients
name_of_3D_patch_folder = "all_clean_3D_pats_patient_"
name_of_3D_patch_file_part1 = "clean_3D_pats_patient_" ## add patient no. acc. to selected_patients array at the end
name_of_3D_patch_file_part2 = "_slice_" ## add slice no. at the end
total_num_of_3D_patches = 198375 ## 5 patients, 75 slices per patient, 529 patches per slice
clean_3D_patch_array = np.zeros((total_num_of_3D_patches, 3, args.pat_size, args.pat_size, 1), dtype="uint8")
total_patches_added_so_far = 0
num_of_patches_per_slice = 529
fname = "clean_3D_patch_array.npy"  

## integrate all slices, patient-wise
for i in range(5):
    patient_no = selected_patients[i]
    print(patient_no)
    path = args.src_dir + str(patient_no) + "/" + name_of_3D_patch_folder + str(patient_no) + "/"   
    print(path)        
    for j in xrange(1,151,2):## other slices have more background, they increase no. of patches w/o any significance
        path = path + name_of_3D_patch_file_part1 + str(patient_no) + name_of_3D_patch_file_part2 + str(j) + ".npy"
        print(path)    
        curr_slice = load_data(filepath = path)       
        for k in range(num_of_patches_per_slice):
            curr_patch = curr_slice[k,:,:,:]  
#            print(curr_patch.shape)
             
            first = curr_patch[:,:,0]
            second = curr_patch[:,:,1]
            third = curr_patch[:,:,2]
#            
#            ## display a patch
#            print(curr_patch.shape)
#            g = curr_patch[:,:,2]
#            print(g.shape)
#            img = PIL.Image.fromarray(g)
#            img.show()
            
            clean_3D_patch_array[total_patches_added_so_far,0,:,:,0] = first
            clean_3D_patch_array[total_patches_added_so_far,1,:,:,0] = second
            clean_3D_patch_array[total_patches_added_so_far,2,:,:,0] = third
            print(total_patches_added_so_far)
            total_patches_added_so_far = total_patches_added_so_far + 1
      
        path = args.src_dir + str(patient_no) + "/" + name_of_3D_patch_folder + str(patient_no) + "/" ## reset the path variable to 
#    print(path)
        ## remove slice no., else subsequent slice no. info just keeps on appending, resulting in long, meaningless path name.
#    print('Another Noise Level Done !')

print(fname)
np.save(os.path.join(args.save_dir, fname), clean_3D_patch_array)   
print(clean_3D_patch_array.shape)


     



# integrate all patient's data       
    