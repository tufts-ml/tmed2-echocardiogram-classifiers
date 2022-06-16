#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
python $script \
    --arch $arch\
    --train_epoch $train_epoch \
    --nimg_per_epoch $nimg_per_epoch\
    --report_nimg $report_nimg\
    --save_nimg $report_nimg\
    --train_dir $train_dir \
    --report_type $report_type\
    --train_labeled_files $train_labeled_files \
    --valid_files $valid_files \
    --test_files $test_files \
    --stanford_test_files $stanford_test_files \
    --diagnosis_class_weights $diagnosis_class_weights \
    --view_class_weights $view_class_weights \
    --batch $batch\
    --lr $lr \
    --wd $wd \
    --dropout_rate $dropout_rate\
    --PLAX_PSAX_upweight_factor $PLAX_PSAX_upweight_factor\
    --auxiliary_task_weight $auxiliary_task_weight \

