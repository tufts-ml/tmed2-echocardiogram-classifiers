#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
python $script \
    --train_epoch $train_epoch \
    --nimg_per_epoch $nimg_per_epoch\
    --report_nimg $report_nimg\
    --save_nimg $save_nimg\
    --train_dir $train_dir \
    --task_name $task_name \
    --report_type $report_type\
    --train_PLAX_labeled_files $train_PLAX_labeled_files \
    --train_PSAX_labeled_files $train_PSAX_labeled_files \
    --train_A4C_labeled_files $train_A4C_labeled_files \
    --train_A2C_labeled_files $train_A2C_labeled_files \
    --train_UsefulUnlabeled_labeled_files $train_UsefulUnlabeled_labeled_files \
    --valid_files $valid_files \
    --test_files $test_files \
    --stanford_test_files $stanford_test_files \
    --class_weights $class_weights \
    --PLAX_batch $PLAX_batch\
    --PSAX_batch $PSAX_batch\
    --A4C_batch $A4C_batch\
    --A2C_batch $A2C_batch\
    --UsefulUnlabeled_batch $UsefulUnlabeled_batch\
    --lr $lr \
    --wd $wd \
    --dropout_rate $dropout_rate\
    --PLAX_PSAX_upweight_factor $PLAX_PSAX_upweight_factor
