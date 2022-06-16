import numpy as np
import pandas as pd
import os
import json
from tqdm import trange
from PIL import Image, ImageSequence, ImageOps
import ast
import csv
# import tensorflow as tf

# print(tf.__version__)

import sys
sys.path.insert(0, '/cluster/tufts/hugheslab/zhuang12/general_utilities')

from shared_utilities import save_json, make_dir_if_not_exists

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_save_dir')
parser.add_argument('--class_to_integer_mapping_dir')
parser.add_argument('--suggested_split_file_path')
parser.add_argument('--raw_data_rootdir')


def LoadImageFeature(file_path):
    im = Image.open(file_path)
    im = np.asarray(im)
    im = im[:,:, np.newaxis] #make it (64, 64, 1) 1:channel
    return im

        
def main(result_save_dir, class_to_integer_mapping_dir, suggested_split_file_path, raw_data_rootdir):
    
    #read from the suggested split csv
    suggested_split_csv = pd.read_csv(suggested_split_file_path)
    
    make_dir_if_not_exists(result_save_dir)
    
    with open(os.path.join(class_to_integer_mapping_dir, 'view_class_to_integer_mapping.json')) as view_file:
        view_class_to_integer_mapping = json.load(view_file)
    
    
    image_list = []
    label_list = []
   
    
    for i in trange(suggested_split_csv.shape[0]):
        query_key = suggested_split_csv.iloc[i].query_key
        split = suggested_split_csv.iloc[i].view_classifier_split
        view_label = suggested_split_csv.iloc[i].view_label
        source_folder = suggested_split_csv.iloc[i].SourceFolder
        
        im = LoadImageFeature(os.path.join(raw_data_rootdir, source_folder, query_key))
        im = np.stack([im.squeeze(), im.squeeze(), im.squeeze()], axis=-1)

        if split != 'unlabeled':
            raise NameError('this script is for train unlabeled set')
        
        else:
            image_list.append(im)
            label_list.append(view_class_to_integer_mapping[view_label])
            
        
     
    image_list = np.array(image_list)
    label_list = np.array(label_list)
    
    
    with open(os.path.join(result_save_dir, 'Unlabeled_image.npy'), 'wb') as f:
        np.save(f, image_list)
    
    with open(os.path.join(result_save_dir, 'Unlabeled_label.npy'), 'wb') as f:
        np.save(f, label_list)
        

    
    
    
    
if __name__=="__main__":
    
    args = parser.parse_args()
    
    result_save_dir = args.result_save_dir
    class_to_integer_mapping_dir = args.class_to_integer_mapping_dir
    suggested_split_file_path = args.suggested_split_file_path
    raw_data_rootdir = args.raw_data_rootdir
    
    main(result_save_dir, class_to_integer_mapping_dir, suggested_split_file_path, raw_data_rootdir)


    
    
        
