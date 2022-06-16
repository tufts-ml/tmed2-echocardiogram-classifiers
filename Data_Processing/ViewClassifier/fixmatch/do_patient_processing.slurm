#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
python $script\
    --split $split\
    --result_save_dir $result_save_dir\
    --class_to_integer_mapping_dir $class_to_integer_mapping_dir\
    --suggested_split_file_path_labeledpart $suggested_split_file_path_labeledpart\
    --suggested_split_file_path_unlabeledpart $suggested_split_file_path_unlabeledpart\
    --raw_data_rootdir $raw_data_rootdir\
    