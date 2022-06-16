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


export script="$YOUR_PATH/src/image_level/DiagnosisClassifier/image_diagnosis.py"
export ML_DATA="$YOUR_PATH/ML_DATA"
export PYTHONPATH=$PYTHONPATH:.
export arch='resnet_multitask'
export train_epoch=250
export nimg_per_epoch=10331
export report_nimg=10331
export save_nimg=10331

export train_dir="$YOUR_PATH/experiments/DiagnosisClassifier/fold1/DEV479"
export report_type="EMA_BalancedAccuracy"

export train_labeled_files='DiagnosisClassifier/fold1/DEV479/train.tfrecord'
export valid_files='DiagnosisClassifier/fold1/DEV479/val.tfrecord'
export test_files='DiagnosisClassifier/fold1/DEV479/test.tfrecord'
export stanford_test_files='stanford_A4C.tfrecord'

export diagnosis_class_weights="0.524,0.409,0.066"
export view_class_weights="0.105,0.301,0.223,0.295,0.075"

export batch=120

mkdir -p $train_dir

for lr in 0.018
do
    export lr=$lr
    
for wd in 0.02

do
    export wd=$wd

for dropout_rate in 0.0 

do
    export dropout_rate=$dropout_rate
    
for PLAX_PSAX_upweight_factor in 5

do
    export PLAX_PSAX_upweight_factor=$PLAX_PSAX_upweight_factor

for auxiliary_task_weight in 1.0

do
    export auxiliary_task_weight=$auxiliary_task_weight
    
    
#     echo $lr $wd $dropout_rate 

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < $YOUR_PATH/runs/image_level/DiagnosisClassifier/do_experiment.slurm

    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash $YOUR_PATH/runs/image_level/DiagnosisClassifier/do_experiment.slurm
    fi

done
done
done
done
done

