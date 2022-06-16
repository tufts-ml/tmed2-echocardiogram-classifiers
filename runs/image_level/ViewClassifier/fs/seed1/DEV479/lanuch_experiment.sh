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


export script="$YOUR_PATH/src/image_level/ViewClassifier/fs/fs.py"
export ML_DATA="$YOUR_PATH/ML_DATA"
export PYTHONPATH=$PYTHONPATH:.
export train_epoch=400
export nimg_per_epoch=18025
export report_nimg=18025
export save_nimg=18025

export train_dir="$YOUR_PATH/experiments/ViewClassifier/fs/fold1/DEV479"
export task_name="ViewClassification"
export report_type="EMA_BalancedAccuracy"

export train_PLAX_labeled_files='ViewClassifier/fs/fold1/DEV479/train_PLAX.tfrecord'
export train_PSAX_labeled_files='ViewClassifier/fs/fold1/DEV479/train_PSAX.tfrecord'
export train_A4C_labeled_files='ViewClassifier/fs/fold1/DEV479/train_A4C.tfrecord'
export train_A2C_labeled_files='ViewClassifier/fs/fold1/DEV479/train_A2C.tfrecord'
export train_UsefulUnlabeled_labeled_files='ViewClassifier/fs/fold1/DEV479/train_A4CorA2CorOther.tfrecord'

export valid_files='ViewClassifier/fs/fold1/DEV479/val.tfrecord'
export test_files='ViewClassifier/fs/fold1/DEV479/test.tfrecord'
export stanford_test_files='stanford_A4C.tfrecord'

export class_weights="1.17,3.21,1.72,2.28,1.61"
export PLAX_batch=60
export PSAX_batch=22
export A4C_batch=41
export A2C_batch=31
export UsefulUnlabeled_batch=44

mkdir -p $train_dir


for lr in 0.0007
do
    export lr=$lr
    
for wd in 0.002 
do
    export wd=$wd

for dropout_rate in 0.4
do
    export dropout_rate=$dropout_rate
    
for PLAX_PSAX_upweight_factor in 3
do
    export PLAX_PSAX_upweight_factor=$PLAX_PSAX_upweight_factor



    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < $YOUR_PATH/runs/image_level/ViewClassifier/fs/do_experiment.slurm

    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash $YOUR_PATH/runs/image_level/ViewClassifier/fs/do_experiment.slurm
    fi

done
done
done
done
