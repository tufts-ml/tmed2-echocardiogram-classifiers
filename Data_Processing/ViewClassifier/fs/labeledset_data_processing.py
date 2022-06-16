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
    
    
    train_PLAX_image_list = []
    train_PSAX_image_list = []
    train_A4C_image_list = []
    train_A2C_image_list = []
    train_A4CorA2CorOther_image_list = []
    
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

        if split == 'train':
            if view_label == 'PLAX':
                train_PLAX_image_list.append(im)
            elif view_label == 'PSAX':
                train_PSAX_image_list.append(im)
            elif view_label == 'A4C':
                train_A4C_image_list.append(im)
            elif view_label == 'A2C':
                train_A2C_image_list.append(im)
            elif view_label == 'A4CorA2CorOther':
                train_A4CorA2CorOther_image_list.append(im)
            else:
                raise NameError('invalide view label')    
            
        elif split == 'val':
            val_image_list.append(im)
            val_label_list.append(view_class_to_integer_mapping[view_label])
            
        elif split == 'test':
            test_image_list.append(im)
            test_label_list.append(view_class_to_integer_mapping[view_label])
        
        else:
            raise NameError('invalid split')
    
     
    train_PLAX_image_list = np.array(train_PLAX_image_list)
    n_train_PLAX = train_PLAX_image_list.shape[0]
    train_PLAX_image_list = _encode_png(train_PLAX_image_list)

    train_PSAX_image_list = np.array(train_PSAX_image_list)
    n_train_PSAX = train_PSAX_image_list.shape[0]
    train_PSAX_image_list = _encode_png(train_PSAX_image_list)

    train_A4C_image_list = np.array(train_A4C_image_list)
    n_train_A4C = train_A4C_image_list.shape[0]
    train_A4C_image_list = _encode_png(train_A4C_image_list)

    train_A2C_image_list = np.array(train_A2C_image_list)
    n_train_A2C = train_A2C_image_list.shape[0]
    train_A2C_image_list = _encode_png(train_A2C_image_list)

    train_A4CorA2CorOther_image_list = np.array(train_A4CorA2CorOther_image_list)
    n_train_A4CorA2CorOther = train_A4CorA2CorOther_image_list.shape[0]
    train_A4CorA2CorOther_image_list = _encode_png(train_A4CorA2CorOther_image_list)

    val_image_list = np.array(val_image_list)
    n_val = val_image_list.shape[0]
    val_image_list = _encode_png(val_image_list)
    val_label_list = np.array(val_label_list)
    
    test_image_list = np.array(test_image_list)
    n_test = test_image_list.shape[0]
    test_image_list = _encode_png(test_image_list)
    test_label_list = np.array(test_label_list)
    
     ######PLAX
    with tf.python_io.TFRecordWriter(os.path.join(result_save_dir, 'train_PLAX.tfrecord')) as writer:
        for i in trange(n_train_PLAX, desc='Writing train_PLAX tfrecrods'):
            feat = dict(image = _bytes_feature(train_PLAX_image_list[i]),
                        label = _int64_feature(view_class_to_integer_mapping['PLAX']))
            
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
        
    print('Finished saving: ', os.path.join(result_save_dir, 'train_PLAX.tfrecord'))
    
    ######PSAX
    with tf.python_io.TFRecordWriter(os.path.join(result_save_dir, 'train_PSAX.tfrecord')) as writer:
        for i in trange(n_train_PSAX, desc='Writing train_PSAX tfrecrods'):
            feat = dict(image = _bytes_feature(train_PSAX_image_list[i]),
                        label = _int64_feature(view_class_to_integer_mapping['PSAX']))
            
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
        
    print('Finished saving: ', os.path.join(result_save_dir, 'train_PSAX.tfrecord'))
    
    ######A4C
    with tf.python_io.TFRecordWriter(os.path.join(result_save_dir, 'train_A4C.tfrecord')) as writer:
        for i in trange(n_train_A4C, desc='Writing train_A4C tfrecrods'):
            feat = dict(image = _bytes_feature(train_A4C_image_list[i]),
                        label = _int64_feature(view_class_to_integer_mapping['A4C']))
            
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
        
    print('Finished saving: ', os.path.join(result_save_dir, 'train_A4C.tfrecord'))
    
    ######A2C
    with tf.python_io.TFRecordWriter(os.path.join(result_save_dir, 'train_A2C.tfrecord')) as writer:
        for i in trange(n_train_A2C, desc='Writing train_A2C tfrecrods'):
            feat = dict(image = _bytes_feature(train_A2C_image_list[i]),
                        label = _int64_feature(view_class_to_integer_mapping['A2C']))
            
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
        
    print('Finished saving: ', os.path.join(result_save_dir, 'train_A2C.tfrecord'))
    
    ######A4CorA2CorOther
    with tf.python_io.TFRecordWriter(os.path.join(result_save_dir, 'train_A4CorA2CorOther.tfrecord')) as writer:
        for i in trange(n_train_A4CorA2CorOther, desc='Writing train_A4CorA2CorOther tfrecrods'):
            feat = dict(image = _bytes_feature(train_A4CorA2CorOther_image_list[i]),
                        label = _int64_feature(view_class_to_integer_mapping['A4CorA2CorOther']))
            
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
        
    print('Finished saving: ', os.path.join(result_save_dir, 'train_A4CorA2CorOther.tfrecord'))
    
    
    
    ######val
    with tf.python_io.TFRecordWriter(os.path.join(result_save_dir, 'val.tfrecord')) as writer:
        for i in trange(n_val, desc='Writing val tfrecrods'):
            feat = dict(image = _bytes_feature(val_image_list[i]),
                        label = _int64_feature(val_label_list[i]))
            
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
        
    print('Finished saving: ', os.path.join(result_save_dir, 'val.tfrecord'))
    
    ######test
    with tf.python_io.TFRecordWriter(os.path.join(result_save_dir, 'test.tfrecord')) as writer:
        for i in trange(n_test, desc='Writing val tfrecrods'):
            feat = dict(image = _bytes_feature(test_image_list[i]),
                        label = _int64_feature(test_label_list[i]))
            
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


    
    
        
