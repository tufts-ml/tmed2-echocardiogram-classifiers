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
    
    
    train_PLAX_image_list = []
    train_PLAX_label_list = []
    
    train_PSAX_image_list = []
    train_PSAX_label_list = []
    
    train_A4C_image_list = []
    train_A4C_label_list = []

    train_A2C_image_list = []
    train_A2C_label_list = []

    train_A4CorA2CorOther_image_list = []
    train_A4CorA2CorOther_label_list = []

    val_image_list = []
    val_label_list = []
    test_image_list = []
    test_label_list = []
    
    for i in trange(suggested_split_csv.shape[0]):
        query_key = suggested_split_csv.iloc[i].query_key
        split = suggested_split_csv.iloc[i].view_classifier_split
        view_label = suggested_split_csv.iloc[i].view_label
        source_folder = suggested_split_csv.iloc[i].SourceFolder
        
        im = LoadImageFeature(os.path.join(raw_data_rootdir, source_folder, query_key))
        im = np.stack([im.squeeze(), im.squeeze(), im.squeeze()], axis=-1)

        if split == 'train':
            if view_label == 'PLAX':
                train_PLAX_image_list.append(im)
                train_PLAX_label_list.append(view_class_to_integer_mapping['PLAX'])
                
            elif view_label == 'PSAX':
                train_PSAX_image_list.append(im)
                train_PSAX_label_list.append(view_class_to_integer_mapping['PSAX'])

            elif view_label == 'A4C':
                train_A4C_image_list.append(im)
                train_A4C_label_list.append(view_class_to_integer_mapping['A4C'])

            elif view_label == 'A2C':
                train_A2C_image_list.append(im)
                train_A2C_label_list.append(view_class_to_integer_mapping['A2C'])

            elif view_label == 'A4CorA2CorOther':
                train_A4CorA2CorOther_image_list.append(im)
                train_A4CorA2CorOther_label_list.append(view_class_to_integer_mapping['A4CorA2CorOther'])

            else:
                raise NameError('invalide view label')    
                
            
        elif split == 'val':
            val_image_list.append(im)
            val_label_list.append(view_class_to_integer_mapping[view_label])
            
        elif split == 'test':
            test_image_list.append(im)
            test_label_list.append(view_class_to_integer_mapping[view_label])
    
     
    train_PLAX_image_list = np.array(train_PLAX_image_list)
    train_PLAX_label_list = np.array(train_PLAX_label_list)

    train_PSAX_image_list = np.array(train_PSAX_image_list)
    train_PSAX_label_list = np.array(train_PSAX_label_list)
    
    train_A4C_image_list = np.array(train_A4C_image_list)
    train_A4C_label_list = np.array(train_A4C_label_list)
    
    train_A2C_image_list = np.array(train_A2C_image_list)
    train_A2C_label_list = np.array(train_A2C_label_list)
   
    train_A4CorA2CorOther_image_list = np.array(train_A4CorA2CorOther_image_list)
    train_A4CorA2CorOther_label_list = np.array(train_A4CorA2CorOther_label_list)
   
    val_image_list = np.array(val_image_list)
    val_label_list = np.array(val_label_list)
    test_image_list = np.array(test_image_list)
    test_label_list = np.array(test_label_list)
    
    with open(os.path.join(result_save_dir, 'train_PLAX_image.npy'), 'wb') as f:
        np.save(f, train_PLAX_image_list)
    
    with open(os.path.join(result_save_dir, 'train_PLAX_label.npy'), 'wb') as f:
        np.save(f, train_PLAX_label_list)
        
    with open(os.path.join(result_save_dir, 'train_PSAX_image.npy'), 'wb') as f:
        np.save(f, train_PSAX_image_list)
    
    with open(os.path.join(result_save_dir, 'train_PSAX_label.npy'), 'wb') as f:
        np.save(f, train_PSAX_label_list)

    with open(os.path.join(result_save_dir, 'train_A4C_image.npy'), 'wb') as f:
        np.save(f, train_A4C_image_list)
    
    with open(os.path.join(result_save_dir, 'train_A4C_label.npy'), 'wb') as f:
        np.save(f, train_A4C_label_list)
    
    with open(os.path.join(result_save_dir, 'train_A2C_image.npy'), 'wb') as f:
        np.save(f, train_A2C_image_list)
    
    with open(os.path.join(result_save_dir, 'train_A2C_label.npy'), 'wb') as f:
        np.save(f, train_A2C_label_list)
    
    with open(os.path.join(result_save_dir, 'train_A4CorA2CorOther_image.npy'), 'wb') as f:
        np.save(f, train_A4CorA2CorOther_image_list)
    
    with open(os.path.join(result_save_dir, 'train_A4CorA2CorOther_label.npy'), 'wb') as f:
        np.save(f, train_A4CorA2CorOther_label_list)
        
    with open(os.path.join(result_save_dir, 'val_image.npy'), 'wb') as f:
        np.save(f, val_image_list)
    
    with open(os.path.join(result_save_dir, 'val_label.npy'), 'wb') as f:
        np.save(f, val_label_list)
        
    with open(os.path.join(result_save_dir, 'test_image.npy'), 'wb') as f:
        np.save(f, test_image_list)
    
    with open(os.path.join(result_save_dir, 'test_label.npy'), 'wb') as f:
        np.save(f, test_label_list)
        

    
    
    
    
if __name__=="__main__":
    
    args = parser.parse_args()
    
    result_save_dir = args.result_save_dir
    class_to_integer_mapping_dir = args.class_to_integer_mapping_dir
    suggested_split_file_path = args.suggested_split_file_path
    raw_data_rootdir = args.raw_data_rootdir
    
    main(result_save_dir, class_to_integer_mapping_dir, suggested_split_file_path, raw_data_rootdir)


    
    
        
