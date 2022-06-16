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
# from libml.result_analysis import perform_analysis
# from libml.checkpoint_ensemble import perform_ensemble

import tensorflow as tf

FLAGS = flags.FLAGS


class FSBaseline(ClassifyFullySupervised, MultiModel):

    def model(self, lr, wd, ema, class_weights, **kwargs):
        
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x') 
        l_in = tf.placeholder(tf.int32, [None], 'labels') 
        
        wd *= lr

        l = tf.one_hot(l_in, self.nclass)

        
        class_weights = class_weights.split(',')
        class_weights = [float(i) for i in class_weights]
        class_weights[0] = round(class_weights[0]*FLAGS.PLAX_PSAX_upweight_factor,3)
        class_weights[1] = round(class_weights[1]*FLAGS.PLAX_PSAX_upweight_factor,3)
        
        class_weights = tf.constant(class_weights) #passed in class_weights is a list of floats
        weights = tf.reduce_sum(class_weights * l, axis=1)

        x, l = self.augment(x_in, l, **kwargs)
        
        classifier = functools.partial(self.classifier, **kwargs)
        
        logits = classifier(x, training=True) 

        unweighted_loss = utils.customized_CE(l, logits, FLAGS.UsefulUnlabeled_batch)

        loss = unweighted_loss * weights
    
        loss = tf.reduce_mean(loss)
        
        tf.summary.scalar('losses/total_loss', loss)
        tf.summary.scalar('losses/unweighted_loss', tf.py_func(np.mean, [unweighted_loss], tf.float32))


        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) + [ema_op]
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        train_op = tf.train.AdamOptimizer(lr).minimize(loss, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)


        return EasyDict(
            x=x_in, label=l_in, train_op=train_op, 
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)),
            labeled_losses = loss)


def main(argv):
    del argv  # Unused.
    
    ######################################################################################
    #experiment settings:
    nclass=5 #to manually define in the script
    height=112
    width=112
    colors=1
    figure_title = 'FS'
    num_bootstrap_samples = 10 #how many bootstrap samples to use
    bootstrap_upper_percentile = 90 #what upper percentile of the bootstrap result to show
    bootstrap_lower_percentile = 10 #what lower percentile of the bootstrap result to show
    num_selection_step = 80 #number of forward stepwise selection step to perform
    ensemble_last_checkpoints = 200 #use last 100 checkpoints as source for ensemble
    ylim_lower=40
    ylim_upper=100
    
    train_PLAX_labeled_files = FLAGS.train_PLAX_labeled_files.split(',')
    train_PSAX_labeled_files = FLAGS.train_PSAX_labeled_files.split(',')
    train_A4C_labeled_files = FLAGS.train_A4C_labeled_files.split(',')
    train_A2C_labeled_files = FLAGS.train_A2C_labeled_files.split(',')
    train_UsefulUnlabeled_labeled_files = FLAGS.train_UsefulUnlabeled_labeled_files.split(',')

    valid_files = FLAGS.valid_files.split(',')
    test_files = FLAGS.test_files.split(',')
    stanford_test_files = FLAGS.stanford_test_files.split(',')
    
    print('train_PLAX_labeled_files is {}'.format(train_PLAX_labeled_files), flush=True)
    print('train_PSAX_labeled_files is {}'.format(train_PSAX_labeled_files), flush=True)
    print('train_A4C_labeled_files is {}'.format(train_A4C_labeled_files), flush=True)
    print('train_A2C_labeled_files is {}'.format(train_A2C_labeled_files), flush=True)
    print('train_UsefulUnlabeled_labeled_files is {}'.format(train_UsefulUnlabeled_labeled_files), flush=True)

    print('valid_files is {}'.format(valid_files), flush=True)
    print('test_files is {}'.format(test_files), flush=True)
    print('stanford_test_files is {}'.format(stanford_test_files), flush=True)
    
    
    ######################################################################################
    DATASETS = {}
    DATASETS.update([datafs.DataSetFS.creator('echo', train_PLAX_labeled_files, train_PSAX_labeled_files, train_A4C_labeled_files, train_A2C_labeled_files, train_UsefulUnlabeled_labeled_files, valid_files, test_files, stanford_test_files,
                                   datafs.data.augment_echo, nclass=nclass, height=height, width=width, colors=colors)])

    dataset = DATASETS[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = FSBaseline(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        PLAX_batch=FLAGS.PLAX_batch,
        PSAX_batch=FLAGS.PSAX_batch,
        A4C_batch=FLAGS.A4C_batch,
        A2C_batch=FLAGS.A2C_batch,
        UsefulUnlabeled_batch=FLAGS.UsefulUnlabeled_batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        PLAX_PSAX_upweight_factor=FLAGS.PLAX_PSAX_upweight_factor,
        class_weights=FLAGS.class_weights,
        continued_training=FLAGS.continued_training,
        smoothing=FLAGS.smoothing,
        scales=FLAGS.scales,
        filters=FLAGS.filters,
        repeat=FLAGS.repeat,
        dropout_rate=FLAGS.dropout_rate)
    
    experiment_dir = model.train_dir
    
    model.train(FLAGS.train_epoch * FLAGS.nimg_per_epoch, FLAGS.report_nimg)
    
    
#     for report_type in ['RAW_BalancedAccuracy', 'EMA_BalancedAccuracy']:
    result_save_dir = os.path.join(experiment_dir, 'result_analysis', FLAGS.report_type)

    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

#     perform_analysis(figure_title, experiment_dir, result_save_dir, num_bootstrap_samples, bootstrap_upper_percentile, bootstrap_lower_percentile, ylim_lower, ylim_upper, FLAGS.report_type, FLAGS.task_name)

#     perform_ensemble(experiment_dir, result_save_dir, num_selection_step, FLAGS.report_type, ensemble_last_checkpoints)

        
    
if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_integer('PLAX_PSAX_upweight_factor', 3, 'upweight the cost of missing PLAX and PSAX')
    flags.DEFINE_string('class_weights', '1.01,2.58,2.39,3.35,0.67', 'the weights used for weighted cross entropy loss')
    flags.DEFINE_string('continued_training', '0_30000', 'the job is which step to which step') 
    flags.DEFINE_string('task_name', 'ViewClassification', 'either ViewClassification or DiagnosisClassification')
    
    flags.DEFINE_string('train_PLAX_labeled_files', 'train_PLAX_SingleLabel.tfrecord', 'name of the train PLAX tfrecord')
    flags.DEFINE_string('train_PSAX_labeled_files', 'train_PSAX_SingleLabel.tfrecord', 'name of the train PSAX tfrecord')
    flags.DEFINE_string('train_A4C_labeled_files', 'train_A4C_SingleLabel.tfrecord', 'name of the train A4C tfrecord')
    flags.DEFINE_string('train_A2C_labeled_files', 'train_A2C_SingleLabel.tfrecord', 'name of the train A2C tfrecord')
    flags.DEFINE_string('train_UsefulUnlabeled_labeled_files', 'train_UsefulUnlabeled_SingleLabel.tfrecord', 'name of the train UsefulUnlabeled tfrecord')

    flags.DEFINE_string('valid_files', 'shared_val_SingleLabel.tfrecord', 'name of the valid tfrecord')
    flags.DEFINE_string('test_files', 'shared_test_SingleLabel.tfrecord', 'name of the test tfrecord') 
    flags.DEFINE_string('stanford_test_files', 'stanford_A4C.tfrecord', 'name of the echonet data files')
    flags.DEFINE_float('smoothing', 0.001, 'Label smoothing.')
    flags.DEFINE_integer('scales', 4, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_float('dropout_rate', 0.0, 'dropout rate for dropout layer in resnet.')
    flags.DEFINE_string('report_type','EMA_BalancedAccuracy', 'using raw or ema of the weights')
    FLAGS.set_default('dataset', 'echo')
    FLAGS.set_default('PLAX_batch', 66)
    FLAGS.set_default('PSAX_batch', 26)
    FLAGS.set_default('A4C_batch', 28)
    FLAGS.set_default('A2C_batch', 20)
    FLAGS.set_default('UsefulUnlabeled_batch', 100)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_epoch', 10)
    app.run(main)
