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
parser.add_argument('--split')
parser.add_argument('--result_save_dir')
parser.add_argument('--class_to_integer_mapping_dir')
parser.add_argument('--suggested_split_file_path_labeledpart')
parser.add_argument('--suggested_split_file_path_unlabeledpart')
parser.add_argument('--raw_data_rootdir')

class study_level_count_dicts():
    
    def __init__(self):
        self.dict=dict()
        
    def update(self, study, diagnosislabel):
        if study not in self.dict:
            self.dict[study] = {
                               'diagnosislabels_count':{'no_AS':0, 'mild_AS':0, 'mildtomod_AS':0, 'moderate_AS':0, 'severe_AS':0}}
        self.dict[study]['diagnosislabels_count'][diagnosislabel] +=1
        
        
    def __call__(self):
        return self.dict
    
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

     
def main(split, result_save_dir, class_to_integer_mapping_dir, suggested_split_file_path_labeledpart, suggested_split_file_path_unlabeledpart, raw_data_rootdir):
    
    count_dicts = study_level_count_dicts() 
    study_order_list = []
    
    #read from the suggested split csv
    suggested_split_csv_labeledpart = pd.read_csv(suggested_split_file_path_labeledpart)
    suggested_split_csv_labeledpart = suggested_split_csv_labeledpart[suggested_split_csv_labeledpart['diagnosis_classifier_split']==split]
    
    suggested_split_csv_unlabeledpart = pd.read_csv(suggested_split_file_path_unlabeledpart)
    suggested_split_csv_unlabeledpart_studyids = np.array([i.split('_')[0] for i in suggested_split_csv_unlabeledpart.query_key.values])
    
    make_dir_if_not_exists(result_save_dir)
    
    with open(os.path.join(class_to_integer_mapping_dir, 'view_class_to_integer_mapping.json')) as view_file:
        view_class_to_integer_mapping = json.load(view_file)
    
    with open(os.path.join(class_to_integer_mapping_dir, 'diagnosis_class_to_integer_mapping.json')) as diagnosis_file:
        diagnosis_class_to_integer_mapping = json.load(diagnosis_file)
    
    image_list = []
    viewlabel_list = []
    diagnosislabel_list = []

    
    current_studyid = None
    for i in trange(suggested_split_csv_labeledpart.shape[0]):
        labeledpart_query_key = suggested_split_csv_labeledpart.iloc[i].query_key
        labeledpart_studyid = labeledpart_query_key.split('_')[0]
        this_split = suggested_split_csv_labeledpart.iloc[i].diagnosis_classifier_split
        assert this_split == split
        
        if current_studyid != labeledpart_studyid:
            print('new studyid: {}, current studyid: {}'.format(labeledpart_studyid, current_studyid))
            #meaning a new studyid is encountered:
            #first extract all the unlabeled images for this labeled set's studyid, then extract the labeled images
            unlabeledpart_mask = suggested_split_csv_unlabeledpart_studyids==labeledpart_studyid
            this_studyid_unlabeledpart_images_df = suggested_split_csv_unlabeledpart.loc[unlabeledpart_mask]
            print('unlabeledpart to extract: {}'.format(this_studyid_unlabeledpart_images_df.shape[0]))
            for j in trange(this_studyid_unlabeledpart_images_df.shape[0]):
                unlabeledpart_query_key = this_studyid_unlabeledpart_images_df.iloc[j].query_key
                unlabeledpart_studyid = unlabeledpart_query_key.split('_')[0]
                
                assert unlabeledpart_studyid == labeledpart_studyid
                unlabeledpart_view_label = this_studyid_unlabeledpart_images_df.iloc[j].view_label
                unlabeledpart_diagnosis_label = this_studyid_unlabeledpart_images_df.iloc[j].diagnosis_label
                unlabeledpart_source_folder = this_studyid_unlabeledpart_images_df.iloc[j].SourceFolder
        
                im = LoadImageFeature(os.path.join(raw_data_rootdir, unlabeledpart_source_folder, unlabeledpart_query_key))
                image_list.append(im)
                viewlabel_list.append(-1)
                diagnosislabel_list.append(diagnosis_class_to_integer_mapping[unlabeledpart_diagnosis_label])
                study_order_list.append(unlabeledpart_studyid)
                count_dicts.update(unlabeledpart_studyid, unlabeledpart_diagnosis_label)
                
            current_studyid = labeledpart_studyid

    
        
        labeledpart_view_label = suggested_split_csv_labeledpart.iloc[i].view_label
        labeledpart_diagnosis_label = suggested_split_csv_labeledpart.iloc[i].diagnosis_label
        labeledpart_source_folder = suggested_split_csv_labeledpart.iloc[i].SourceFolder

        im = LoadImageFeature(os.path.join(raw_data_rootdir, labeledpart_source_folder, labeledpart_query_key))

        image_list.append(im)
        viewlabel_list.append(view_class_to_integer_mapping[labeledpart_view_label])
        diagnosislabel_list.append(diagnosis_class_to_integer_mapping[labeledpart_diagnosis_label])
        
        study_order_list.append(labeledpart_studyid)
        count_dicts.update(labeledpart_studyid, labeledpart_diagnosis_label)
        
    
     
    image_list = np.array(image_list)
    n_images = image_list.shape[0]
    image_list = _encode_png(image_list)
    viewlabel_list = np.array(viewlabel_list)
    diagnosislabel_list = np.array(diagnosislabel_list)
    
    
    #save the study_level_count_dicts and study_order_list
    make_dir_if_not_exists(os.path.join(result_save_dir, 'study_order_info'))
    
    with open(os.path.join(result_save_dir, 'study_order_info', 'PatientLevel_{}_study_level_count_dicts.json'.format(split)), 'w') as study_level_count_dicts_json_file:
        json.dump(count_dicts(), study_level_count_dicts_json_file)
    
    with open(os.path.join(result_save_dir, 'study_order_info', 'PatientLevel_{}_study_order_list.json'.format(split)), 'w') as study_order_list_json_file:
        json.dump(study_order_list, study_order_list_json_file)
    
    ######test
    with tf.python_io.TFRecordWriter(os.path.join(result_save_dir, 'PatientLevel_{}.tfrecord'.format(split))) as writer:
        for i in trange(n_images, desc='Writing val tfrecrods'):
            feat = dict(image = _bytes_feature(image_list[i]),
                        diagnosis_label = _int64_feature(diagnosislabel_list[i]),
                        view_label = _int64_feature(viewlabel_list[i]))
            
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
        
    print('Finished saving: ', os.path.join(result_save_dir, 'PatientLevel_{}.tfrecord'.format(split)))
    
    
    
    
    
if __name__=="__main__":
    
    args = parser.parse_args()
    
    split = args.split
    result_save_dir = args.result_save_dir
    class_to_integer_mapping_dir = args.class_to_integer_mapping_dir
    suggested_split_file_path_labeledpart = args.suggested_split_file_path_labeledpart
    suggested_split_file_path_unlabeledpart = args.suggested_split_file_path_unlabeledpart    
    raw_data_rootdir = args.raw_data_rootdir
    
    main(split, result_save_dir, class_to_integer_mapping_dir, suggested_split_file_path_labeledpart, suggested_split_file_path_unlabeledpart, raw_data_rootdir)


    
    
        
