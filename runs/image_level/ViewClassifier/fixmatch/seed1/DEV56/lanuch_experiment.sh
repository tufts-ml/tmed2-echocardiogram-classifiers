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

export resume='last_checkpoint.pth.tar'

export arch='wideresnet_scale4'

export script="$YOUR_PATH/src/image_level/ViewClassifier/fixmatch/fixmatch.py"

export train_PLAX_image_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/DEV56/train_PLAX_image.npy"
export train_PLAX_label_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/DEV56/train_PLAX_label.npy"
export train_PSAX_image_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/DEV56/train_PSAX_image.npy"
export train_PSAX_label_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/DEV56/train_PSAX_label.npy"
export train_A4C_image_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/DEV56/train_A4C_image.npy"
export train_A4C_label_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/DEV56/train_A4C_label.npy"
export train_A2C_image_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/DEV56/train_A2C_image.npy"
export train_A2C_label_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/DEV56/train_A2C_label.npy"
export train_UsefulUnlabeled_image_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/DEV56/train_UsefulUnlabeled_image.npy"
export train_UsefulUnlabeled_label_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/DEV56/train_UsefulUnlabeled_label.npy"

export val_image_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/DEV56/val_image.npy"
export val_label_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/DEV56/val_label.npy"
export test_image_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/shared_test_this_seed/test_image.npy"
export test_label_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/shared_test_this_seed/test_label.npy"  
export unlabeled_image_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_shared_Unlabeled/Unlabeled_image.npy"
export unlabeled_label_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_shared_Unlabeled/Unlabeled_label.npy"

export stanford_test_image_path="$YOUR_PATH/ML_DATA/stanford_A4C_image.npy"
export stanford_test_label_path="$YOUR_PATH/ML_DATA/stanford_A4C_label.npy"

export ForPatientTestSet_test_image_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/shared_test_this_seed/ForPatientTestSet_test_image.npy"
export ForPatientTestSet_test_label_path="$YOUR_PATH/ML_DATA/ViewClassifier/fixmatch/npy_seed1/shared_test_this_seed/ForPatientTestSet_test_label.npy"


export train_dir="$YOUR_PATH/experiments/ViewClassifier/fixmatch/seed1/DEV56"

mkdir -p $train_dir

export train_epoch=250
export nimg_per_epoch=2123
export num_workers=4

export class_weights="1.08,3.24,1.62,2.43,1.62"
export PLAX_batch=9
export PSAX_batch=3
export A4C_batch=6
export A2C_batch=4
export UsefulUnlabeled_batch=6




for lr in 0.003 
do 
    export lr=$lr

for wd in 5e-3
do 
    export wd=$wd
    

for dropout_rate in 0.4
do
    export dropout_rate=$dropout_rate


for PLAX_PSAX_upweight_factor in 1

do
    export PLAX_PSAX_upweight_factor=$PLAX_PSAX_upweight_factor


for lambda_u in 1.0
do
    export lambda_u=$lambda_u
    
    
for warmup_img in 0
do
    export warmup_img=$warmup_img

for mu in 2
do
    export mu=$mu

for T in 1.0
do
    export T=$T

for threshold in 0.95 
do
    export threshold=$threshold



if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < $YOUR_PATH/runs/image_level/ViewClassifier/fixmatch/do_experiment.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash $YOUR_PATH/runs/image_level/ViewClassifier/fixmatch/do_experiment.slurm
fi

done
done
done
done
done
done
done
done
done

