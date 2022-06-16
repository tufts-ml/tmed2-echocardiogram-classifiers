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


export script="$YOUR_PATH/src/image_level/ViewClassifier/fs.py"
export ML_DATA="$YOUR_PATH/ML_DATA"
export PYTHONPATH=$PYTHONPATH:.
export train_epoch=550
export nimg_per_epoch=2123
export report_nimg=10615
export save_nimg=10615

export train_dir="$YOUR_PATH/experiments/ViewClassifier/fs/seed1/DEV56"
export task_name="ViewClassification"
export report_type="EMA_BalancedAccuracy"

export train_PLAX_labeled_files='ViewClassifier/fs/seed1/DEV56/train_PLAX_SingleLabel.tfrecord'
export train_PSAX_labeled_files='ViewClassifier/fs/seed1/DEV56/train_PSAX_SingleLabel.tfrecord'
export train_A4C_labeled_files='ViewClassifier/fs/seed1/DEV56/train_A4C_SingleLabel.tfrecord'
export train_A2C_labeled_files='ViewClassifier/fs/seed1/DEV56/train_A2C_SingleLabel.tfrecord'
export train_UsefulUnlabeled_labeled_files='ViewClassifier/fs/seed1/DEV56/train_UsefulUnlabeled_SingleLabel.tfrecord'

export valid_files='ViewClassifier/fs/seed1/DEV56/val.tfrecord'
export test_files='ViewClassifier/fs/seed1/shared_test_this_seed/test.tfrecord'
export stanford_test_files='stanford_A4C.tfrecord'

export class_weights="1.11,3.42,1.60,2.28,1.60"
export PLAX_batch=43
export PSAX_batch=14
export A4C_batch=30
export A2C_batch=21
export UsefulUnlabeled_batch=30

mkdir -p $train_dir


for lr in 0.002
do
    export lr=$lr
    
for wd in 0.002 
do
    export wd=$wd

for dropout_rate in 0.4
do
    export dropout_rate=$dropout_rate
    
for PLAX_PSAX_upweight_factor in 5
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
