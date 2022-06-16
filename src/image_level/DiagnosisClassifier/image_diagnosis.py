# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fully supervised training.
"""

import functools
import os
import numpy as np

from absl import app
from absl import flags
from easydict import EasyDict

import libfs.data as datafs
from libfs.train import ClassifyFullySupervised
from libml import utils
from libml.models import MultiModel
# from libml.result_analysis_view import perform_analysis_view
# from libml.result_analysis_diagnosis import perform_analysis_diagnosis
# from libml.checkpoint_ensemble_view import perform_ensemble_view
# from libml.checkpoint_ensemble_diagnosis import perform_ensemble_diagnosis


import tensorflow as tf

FLAGS = flags.FLAGS


class FSBaseline(ClassifyFullySupervised, MultiModel):

    def model(self, lr, wd, ema, diagnosis_class_weights, view_class_weights, **kwargs):
        
        print('Insie FSBaseline, inside model function, passed in diagnosis_class_weights is {}'.format(diagnosis_class_weights), flush=True)
        print('Insie FSBaseline, inside model function, passed in view_class_weights is {}'.format(view_class_weights), flush=True)

        
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x') 
        l_in_diagnosis = tf.placeholder(tf.int32, [None], 'diagnosis_labels') 
        l_in_view = tf.placeholder(tf.int32, [None], 'view_labels')
        
        wd *= lr

        #turn into one-hot
        l_diagnosis = tf.one_hot(l_in_diagnosis, self.diagnosis_nclass)
        l_view = tf.one_hot(l_in_view, self.view_nclass)
        
        
        diagnosis_class_weights = diagnosis_class_weights.split(',')
        diagnosis_class_weights = [float(i) for i in diagnosis_class_weights]
        diagnosis_class_weights = tf.constant(diagnosis_class_weights) #passed in class_weights is a list of floats
        diagnosis_weights = tf.reduce_sum(diagnosis_class_weights * l_diagnosis, axis=1)

        view_class_weights = view_class_weights.split(',')
        view_class_weights = [float(i) for i in view_class_weights]
        view_class_weights[0] = round(view_class_weights[0]*FLAGS.PLAX_PSAX_upweight_factor, 3)
        view_class_weights[1] = round(view_class_weights[1]*FLAGS.PLAX_PSAX_upweight_factor, 3)
        view_class_weights = tf.constant(view_class_weights) #passed in class_weights is a list of floats
        view_weights = tf.reduce_sum(view_class_weights * l_view, axis=1)
        
        #label smoothing
        smoothing = kwargs['smoothing']
        l_diagnosis = l_diagnosis - smoothing * (l_diagnosis-1./self.diagnosis_nclass)
#         l_view = l_view - smoothing * (l_view - 1./self.view_nclass)
        
                
        classifier = functools.partial(self.classifier, **kwargs)
        
        diagnosis_logits, view_logits = classifier(x_in, training=True) 
        
        #diagnosis loss
#         diagnosis_loss = utils.customized_diagnosis_loss(l_diagnosis, diagnosis_logits, diagnosis_weights)
        unweighted_diagnosis_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l_diagnosis, logits=diagnosis_logits)
        diagnosis_loss = unweighted_diagnosis_loss * diagnosis_weights 
        diagnosis_loss = tf.reduce_mean(diagnosis_loss)
        tf.summary.scalar('losses/diagnosis_loss', diagnosis_loss)
        
        
        #view loss
        unweighted_view_loss = utils.customized_CE_using_LabelMask_withoutlabelsmoothing(l_in_view, view_logits)
        view_loss = unweighted_view_loss * view_weights 
        view_loss = tf.reduce_mean(view_loss)

        tf.summary.scalar('losses/view_loss', view_loss)
        
        #scaled view loss
        auxiliary_task_weight = kwargs['auxiliary_task_weight']
        scaled_view_loss = auxiliary_task_weight * view_loss
        tf.summary.scalar('losses/scaled_view_loss', scaled_view_loss)
        
        
        #total loss:
        loss = diagnosis_loss + scaled_view_loss
        tf.summary.scalar('losses/total_loss', loss)
        
        
        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) + [ema_op]
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        train_op = tf.train.AdamOptimizer(lr).minimize(loss, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        diagnosis_logits_raw_inferencetime, view_logits_raw_inferencetime = classifier(x_in, training=False)
        diagnosis_logits_ema_inferencetime, view_logits_ema_inferencetime = classifier(x_in, getter=ema_getter, training=False)

        return EasyDict(
            x=x_in, diagnosis_label=l_in_diagnosis, view_label=l_in_view, train_op=train_op, 
            diagnosis_inference_raw=tf.nn.softmax(diagnosis_logits_raw_inferencetime),
            diagnosis_inference_ema=tf.nn.softmax(diagnosis_logits_ema_inferencetime),
# No EMA, for debugging.
            view_inference_raw=tf.nn.softmax(view_logits_raw_inferencetime),# No EMA, for debugging.
            view_inference_ema=tf.nn.softmax(view_logits_ema_inferencetime),
            total_losses = loss, diagnosis_loss=diagnosis_loss, unscaled_view_loss=view_loss, scaled_view_loss=scaled_view_loss)


def main(argv):
    del argv  # Unused.
    
    ######################################################################################
    #experiment settings:
    view_nclass=5 #to manually define in the script
    diagnosis_nclass=3
    height=112
    width=112
    colors=1
    figure_title = 'FS'
    num_bootstrap_samples = 10 #how many bootstrap samples to use
    bootstrap_upper_percentile = 90 #what upper percentile of the bootstrap result to show
    bootstrap_lower_percentile = 10 #what lower percentile of the bootstrap result to show
    num_selection_step = 80 #number of forward stepwise selection step to perform
    ensemble_last_checkpoints = 25 #use last 100 checkpoints as source for ensemble
    ylim_lower=30
    ylim_upper=100
    

    train_labeled_files = FLAGS.train_labeled_files.split(',')
    valid_files = FLAGS.valid_files.split(',')
    test_files = FLAGS.test_files.split(',')
    stanford_test_files = FLAGS.stanford_test_files.split(',') #only look at view prediction performance for stanford_test_files

    print('train_labeled_files is {}'.format(train_labeled_files), flush=True)
    print('valid_files is {}'.format(valid_files), flush=True)
    print('test_files is {}'.format(test_files), flush=True)
    print('stanford_test_files is {}'.format(stanford_test_files), flush=True)
    
    
    ######################################################################################
    DATASETS = {}
    DATASETS.update([datafs.DataSetFS.creator('echo', train_labeled_files, valid_files, test_files, stanford_test_files,
                                   datafs.data.augment_echo, diagnosis_nclass=diagnosis_nclass, view_nclass=view_nclass, height=height, width=width, colors=colors)])

    dataset = DATASETS[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = FSBaseline(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        ema=FLAGS.ema,
        diagnosis_nclass=dataset.diagnosis_nclass,
        view_nclass=dataset.view_nclass,
        PLAX_PSAX_upweight_factor=FLAGS.PLAX_PSAX_upweight_factor,
        diagnosis_class_weights=FLAGS.diagnosis_class_weights,
        view_class_weights=FLAGS.view_class_weights,
        continued_training=FLAGS.continued_training,
        auxiliary_task_weight=FLAGS.auxiliary_task_weight,
        smoothing=FLAGS.smoothing,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat,
        dropout_rate=FLAGS.dropout_rate)
    
    experiment_dir = model.train_dir
    
    model.train(FLAGS.train_epoch * FLAGS.nimg_per_epoch, FLAGS.report_nimg)
    
    result_save_dir = os.path.join(experiment_dir, 'result_analysis', FLAGS.report_type)

    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

#     perform_analysis_view(figure_title, experiment_dir, result_save_dir, num_bootstrap_samples, bootstrap_upper_percentile, bootstrap_lower_percentile, ylim_lower, ylim_upper, FLAGS.report_type, 'ViewClassification')
    
#     perform_analysis_diagnosis(figure_title, experiment_dir, result_save_dir, num_bootstrap_samples, bootstrap_upper_percentile, bootstrap_lower_percentile, ylim_lower, ylim_upper, FLAGS.report_type, 'DiagnosisClassification')


#     perform_ensemble_view(experiment_dir, result_save_dir, num_selection_step, FLAGS.report_type, ensemble_last_checkpoints, 'ViewClassification')
    
#     perform_ensemble_diagnosis(experiment_dir, result_save_dir, num_selection_step, FLAGS.report_type, ensemble_last_checkpoints, 'DiagnosisClassification')

        
    
if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_string('diagnosis_class_weights', '0.516,0.418,0.065', 'the weights used for weighted cross entropy loss for diagnosis prediction')
    flags.DEFINE_integer('PLAX_PSAX_upweight_factor', 3, 'upweight the cost of missing PLAX and PSAX when calculating view loss')
    flags.DEFINE_string('view_class_weights', '0.101,0.262,0.233,0.337,0.067', 'the weights used for weighted cross entropy loss for view prediction')
    flags.DEFINE_float('auxiliary_task_weight', 0.3, 'control the strength of auxiliary task loss')
    flags.DEFINE_string('continued_training', '0_30000', 'the job is which step to which step') 
    flags.DEFINE_string('train_labeled_files', 'train.tfrecord', 'name of the train PLAX tfrecord')
    flags.DEFINE_string('valid_files', 'shared_val_SingleLabel.tfrecord', 'name of the valid tfrecord')
    flags.DEFINE_string('test_files', 'shared_test_SingleLabel.tfrecord', 'name of the test tfrecord') 
    flags.DEFINE_string('stanford_test_files', 'stanford_A4C.tfrecord', 'name of the test tfrecord')     
    flags.DEFINE_float('smoothing', 0.001, 'Label smoothing.')
    flags.DEFINE_integer('scales', 4, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_float('dropout_rate', 0.0, 'dropout rate for dropout layer in resnet.')
    flags.DEFINE_string('report_type','RAW_BalancedAccuracy', 'using raw or ema of the weights')
    FLAGS.set_default('dataset', 'echo')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_epoch', 10)
    app.run(main)
