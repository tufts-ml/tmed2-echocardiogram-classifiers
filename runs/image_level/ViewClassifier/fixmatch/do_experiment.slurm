#!/usr/bin/env bash

# Pass along all ENV variables as arguments to my Python script
python $script \
    --resume $resume\
    --arch $arch\
    --train_PLAX_image_path $train_PLAX_image_path\
    --train_PLAX_label_path $train_PLAX_label_path\
    --train_PSAX_image_path $train_PSAX_image_path\
    --train_PSAX_label_path $train_PSAX_label_path\
    --train_A4C_image_path $train_A4C_image_path\
    --train_A4C_label_path $train_A4C_label_path\
    --train_A2C_image_path $train_A2C_image_path\
    --train_A2C_label_path $train_A2C_label_path\
    --train_UsefulUnlabeled_image_path $train_UsefulUnlabeled_image_path\
    --train_UsefulUnlabeled_label_path $train_UsefulUnlabeled_label_path\
    --val_image_path $val_image_path\
    --val_label_path $val_label_path\
    --test_image_path $test_image_path\
    --test_label_path $test_label_path\
    --unlabeled_image_path $unlabeled_image_path\
    --unlabeled_label_path $unlabeled_label_path\
    --stanford_test_image_path $stanford_test_image_path\
    --stanford_test_label_path $stanford_test_label_path\
    --ForPatientTestSet_test_image_path $ForPatientTestSet_test_image_path\
    --ForPatientTestSet_test_label_path $ForPatientTestSet_test_label_path\
    --train_dir $train_dir \
    --train_epoch $train_epoch\
    --nimg_per_epoch $nimg_per_epoch\
    --num_workers $num_workers\
    --class_weights $class_weights\
    --PLAX_batch $PLAX_batch\
    --PSAX_batch $PSAX_batch\
    --A4C_batch $A4C_batch\
    --A2C_batch $A2C_batch\
    --UsefulUnlabeled_batch $UsefulUnlabeled_batch\
    --lr $lr\
    --wd $wd\
    --dropout_rate $dropout_rate\
    --PLAX_PSAX_upweight_factor $PLAX_PSAX_upweight_factor\
    --lambda_u $lambda_u\
    --warmup_img $warmup_img\
    --mu $mu\
    --T $T\
    --threshold $threshold\
    
    