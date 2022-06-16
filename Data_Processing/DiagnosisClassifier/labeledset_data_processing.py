import numpy as np
import pandas as pd
import os
import json
from tqdm import trange
from PIL import Image, ImageSequence, ImageOps
import ast
import csv
import tensorflow as tf

print(tf.__version__)


import sys
sys.path.insert(0, '/cluster/tufts/hugheslab/zhuang12/general_utilities')

from shared_utilities import save_json, make_dir_if_not_exists

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_save_dir')
parser.add_argument('--class_to_integer_mapping_dir')
parser.add_argument('--suggested_split_file_path')
parser.add_argument('--raw_data_rootdir')

def _encode_png(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)
        for x in trange(images.shape[0], desc='PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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
    
    with open(os.path.join(class_to_integer_mapping_dir, 'diagnosis_class_to_integer_mapping.json')) as diagnosis_file:
        diagnosis_class_to_integer_mapping = json.load(diagnosis_file)
    
    
    train_image_list = []
    train_viewlabel_list = []
    train_diagnosislabel_list = []
    
    val_image_list = []
    val_viewlabel_list = []
    val_diagnosislabel_list = []

    test_image_list = []
    test_viewlabel_list = []
    test_diagnosislabel_list = []
    
    for i in trange(suggested_split_csv.shape[0]):
        query_key = suggested_split_csv.iloc[i].query_key
        split = suggested_split_csv.iloc[i].diagnosis_classifier_split
        view_label = suggested_split_csv.iloc[i].view_label
        diagnosis_label = suggested_split_csv.iloc[i].diagnosis_label
        source_folder = suggested_split_csv.iloc[i].SourceFolder
        
        im = LoadImageFeature(os.path.join(raw_data_rootdir, source_folder, query_key))

        if split == 'train':
            train_image_list.append(im)
            train_viewlabel_list.append(view_class_to_integer_mapping[view_label])
            train_diagnosislabel_list.append(diagnosis_class_to_integer_mapping[diagnosis_label])
            
        elif split == 'val':
            val_image_list.append(im)
            val_viewlabel_list.append(view_class_to_integer_mapping[view_label])
            val_diagnosislabel_list.append(diagnosis_class_to_integer_mapping[diagnosis_label])
            
        elif split == 'test':
            test_image_list.append(im)
            test_viewlabel_list.append(view_class_to_integer_mapping[view_label])
            test_diagnosislabel_list.append(diagnosis_class_to_integer_mapping[diagnosis_label])

        else:
            continue
    
     
    train_image_list = np.array(train_image_list)
    n_train = train_image_list.shape[0]
    train_image_list = _encode_png(train_image_list)
    train_viewlabel_list = np.array(train_viewlabel_list)
    train_diagnosislabel_list = np.array(train_diagnosislabel_list)
    
    val_image_list = np.array(val_image_list)
    n_val = val_image_list.shape[0]
    val_image_list = _encode_png(val_image_list)
    val_viewlabel_list = np.array(val_viewlabel_list)
    val_diagnosislabel_list = np.array(val_diagnosislabel_list)
    
    test_image_list = np.array(test_image_list)
    n_test = test_image_list.shape[0]
    test_image_list = _encode_png(test_image_list)
    test_viewlabel_list = np.array(test_viewlabel_list)
    test_diagnosislabel_list = np.array(test_diagnosislabel_list)
    
     ######PLAX
    with tf.python_io.TFRecordWriter(os.path.join(result_save_dir, 'train.tfrecord')) as writer:
        for i in trange(n_train, desc='Writing train tfrecrods'):
            feat = dict(image = _bytes_feature(train_image_list[i]),
                        diagnosis_label = _int64_feature(train_diagnosislabel_list[i]),
                        view_label = _int64_feature(train_viewlabel_list[i]))
            
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
        
    print('Finished saving: ', os.path.join(result_save_dir, 'train.tfrecord'))
    
    
    ######val
    with tf.python_io.TFRecordWriter(os.path.join(result_save_dir, 'val.tfrecord')) as writer:
        for i in trange(n_val, desc='Writing val tfrecrods'):
            feat = dict(image = _bytes_feature(val_image_list[i]),
                        diagnosis_label = _int64_feature(val_diagnosislabel_list[i]),
                        view_label = _int64_feature(val_viewlabel_list[i]))
            
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
        
    print('Finished saving: ', os.path.join(result_save_dir, 'val.tfrecord'))
    
    ######test
    with tf.python_io.TFRecordWriter(os.path.join(result_save_dir, 'test.tfrecord')) as writer:
        for i in trange(n_test, desc='Writing val tfrecrods'):
            feat = dict(image = _bytes_feature(test_image_list[i]),
                        diagnosis_label = _int64_feature(test_diagnosislabel_list[i]),
                        view_label = _int64_feature(test_viewlabel_list[i]))
            
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
        
    print('Finished saving: ', os.path.join(result_save_dir, 'test.tfrecord'))
    
    
    
    
    
if __name__=="__main__":
    
    args = parser.parse_args()
    
    result_save_dir = args.result_save_dir
    class_to_integer_mapping_dir = args.class_to_integer_mapping_dir
    suggested_split_file_path = args.suggested_split_file_path
    raw_data_rootdir = args.raw_data_rootdir
    
    main(result_save_dir, class_to_integer_mapping_dir, suggested_split_file_path, raw_data_rootdir)


    
    
        
