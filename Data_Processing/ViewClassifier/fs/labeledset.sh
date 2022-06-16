#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export script='/cluster/tufts/hugheslab/zhuang12/JACC_CodeRelease/Echo_TMEDV2/Data_Processing/ViewClassifier/fs/labeledset_data_processing.py'
export class_to_integer_mapping_dir="/cluster/tufts/hugheslab/zhuang12/JACC_CodeRelease/Echo_TMEDV2/Data_Processing/"
export raw_data_rootdir="/cluster/tufts/hugheslab/zhuang12/JACC_DataRelease/20220412version/generated_images"

for seed in 0 1 2
do
    export seed=$seed

for version in DEV479 DEV165 DEV56
do
    export version=$version
    
    export result_save_dir="/cluster/tufts/hugheslab/zhuang12/JACC_CodeRelease/Echo_TMEDV2/ML_DATA/ViewClassifier/fs/fold$seed/$version" 
    export suggested_split_file_path="/cluster/tufts/hugheslab/zhuang12/JACC_DataRelease/20220412version/SplitImageLabelMapping/release/fold$seed/$version/labeledset_labeledpart.csv"


    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < do_processing.slurm

    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash do_processing.slurm
    fi

done
done