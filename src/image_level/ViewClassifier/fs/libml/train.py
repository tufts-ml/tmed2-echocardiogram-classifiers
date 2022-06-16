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
"""Training loop, checkpoint saving and loading, evaluation code."""

import json
#import os.path
import os
import shutil

import numpy as np
import tensorflow as tf
from absl import flags
from easydict import EasyDict
from tqdm import trange

from libml import utils
from libml import data as data

import time

FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', './experiments',
                    'Folder where to save training data.')
flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
flags.DEFINE_integer('PLAX_batch', 33, 'PLAX Batch size.')
flags.DEFINE_integer('PSAX_batch', 13, 'PSAX Batch size.')
flags.DEFINE_integer('A4C_batch', 14, 'A4C Batch size.')
flags.DEFINE_integer('A2C_batch', 10, 'A2C Batch size.')
flags.DEFINE_integer('UsefulUnlabeled_batch', 50, 'UsefulUnlabeled Batch size.')

flags.DEFINE_integer('train_epoch', 500, 'How many epoch to train.')
flags.DEFINE_integer('nimg_per_epoch', 25000, 'Training duration in number of samples.')
flags.DEFINE_integer('report_nimg', 25000, 'Report summary period in number of samples.')
flags.DEFINE_integer('save_nimg', 25000, 'Save checkpoint period in number of samples.')

flags.DEFINE_integer('keep_ckpt', 1, 'Number of checkpoints to keep.')
flags.DEFINE_bool('reset_global_step', False, 'initialized from pretrained weights')
flags.DEFINE_string('load_ckpt', "None", 'Checkpoint to initialize from')
flags.DEFINE_string('checkpoint_exclude_scopes', "None", 'Comma-separated list of scopes of variables to exclude when restoring')
flags.DEFINE_string('trainable_scopes', "None", 'Comma-separated list of scopes of variables to train')

class Model:
    def __init__(self, train_dir: str, dataset: data.DataSet, **kwargs):
        self.train_dir = os.path.join(train_dir, self.experiment_name(**kwargs))
        self.params = EasyDict(kwargs)
        self.dataset = dataset
        self.session = None
        self.tmp = EasyDict(print_queue=[], cache=EasyDict())
        self.step = tf.train.get_or_create_global_step()
        self.ops = self.model(**kwargs)
        self.batch_total = FLAGS.PLAX_batch + FLAGS.PSAX_batch + FLAGS.A4C_batch + FLAGS.A2C_batch + FLAGS.UsefulUnlabeled_batch  
        self.ops.update_step = tf.assign_add(self.step, self.batch_total)
        self.add_summaries(**kwargs)

#         self.losses_dict = {'labeled_losses':[], 'unlabeled_losses_unscaled':[], 'unlabeled_losses_scaled':[], 'unlabeled_losses_multiplier':[]}
        self.best_balanced_validation_accuracy_raw = 0 #initialize to 0
        self.best_balanced_validation_accuracy_ema = 0 #initialize to 0
        self.best_balanced_test_accuracy_raw_at_max_val = 0
        self.best_balanced_test_accuracy_ema_at_max_val = 0
        
        self.saver = tf.train.Saver()
        self.init_op = tf.global_variables_initializer()


        #if there is already checkpoint in the Model/tf folder, continue training from the latest checkpoint, set FLAGS.load_ckpt to None, FLAGS.reset_global_step to False
        try:
            continue_training_ckpt = utils.find_latest_checkpoint(self.checkpoint_dir)
        except:
            continue_training_ckpt = None
        
        if continue_training_ckpt is not None:
            FLAGS.load_ckpt = "None"
            FLAGS.reset_global_step = False
            
        
        
        if FLAGS.load_ckpt != "None":
            vars_to_exclude = []
            scopes_to_exclude = []
            if FLAGS.reset_global_step:
                scopes_to_exclude.append('global_step:0')

            if FLAGS.checkpoint_exclude_scopes != "None":
                scopes_to_exclude.extend([scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')])
                
            for v in tf.all_variables():
                for scope in scopes_to_exclude:
                    if scope in v.name.split('/'):
                        vars_to_exclude.append(v)

              
            vars_to_load = [v for v in tf.all_variables() if v not in vars_to_exclude]
#             vars_to_load = [v for v in tf.all_variables()]
            
            self.finetuning_saver = tf.train.Saver(var_list=vars_to_load)
        
        
        print(' Config '.center(80, '-'))
        print('train_dir', self.train_dir)
        print('%-32s %s' % ('Model', self.__class__.__name__))
        print('%-32s %s' % ('Dataset', dataset.name))
        for k, v in sorted(kwargs.items()):
            print('%-32s %s' % (k, v))
        print(' Model '.center(80, '-'))
        to_print = [tuple(['%s' % x for x in (v.name, np.prod(v.shape), v.shape)]) for v in utils.model_vars(None)]
        to_print.append(('Total', str(sum(int(x[1]) for x in to_print)), ''))
        sizes = [max([len(x[i]) for x in to_print]) for i in range(3)]
        fmt = '%%-%ds  %%%ds  %%%ds' % tuple(sizes)
        for x in to_print[:-1]:
            print(fmt % x)
        print()
        print(fmt % to_print[-1])
        print('-' * 80)
        self._create_initial_files()
        
        
    
    def get_variables_to_train(self):
        '''
        Return a list of variables to train, to be passed to optimizer
        '''
        
        if FLAGS.trainable_scopes == "None":
            return tf.trainable_variables()
        else:
            scopes_to_train=[scope.strip() for scope in FLAGS.trainable_scopes.split(',')]
                
            variables_to_train = []
            for v in tf.trainable_variables():
                for scope in scopes_to_train:
                    if scope in v.name.split('/'):
                        variables_to_train.append(v)

            return variables_to_train
        
    
    def init_fn(self, _, sess):
        sess.run(self.init_op)   
        if FLAGS.load_ckpt != "None":
            self.finetuning_saver.restore(sess, FLAGS.load_ckpt)

            
    @property
    def arg_dir(self):
        return os.path.join(self.train_dir, 'args')

    @property
    def checkpoint_dir(self):
        return os.path.join(self.train_dir, 'tf')

    def train_print(self, text):
        self.tmp.print_queue.append(text)

    def _create_initial_files(self):
        for dir in (self.checkpoint_dir, self.arg_dir):
            if not os.path.exists(dir):
                os.makedirs(dir)
        self.save_args()

    def _reset_files(self):
        shutil.rmtree(self.train_dir)
        self._create_initial_files()

    def save_args(self, **extra_params):
        with open(os.path.join(self.arg_dir, 'args.json'), 'w') as f:
            json.dump({**self.params, **extra_params}, f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, train_dir):
        with open(os.path.join(train_dir, 'args/args.json'), 'r') as f:
            params = json.load(f)
        instance = cls(train_dir=train_dir, **params)
        instance.train_dir = train_dir
        return instance

    def experiment_name(self, **kwargs):
        args = [x + str(y) for x, y in sorted(kwargs.items()) if x != 'continued_training']
        return '_'.join([self.__class__.__name__] + args)

    def eval_mode(self, ckpt=None):
        self.session = tf.Session(config=utils.get_config())
        saver = tf.train.Saver()
        if ckpt is None:
            ckpt = utils.find_latest_checkpoint(self.checkpoint_dir)
        else:
            ckpt = os.path.abspath(ckpt)
        saver.restore(self.session, ckpt)
        self.tmp.step = self.session.run(self.step)
        print('Eval model %s at global_step %d' % (self.__class__.__name__, self.tmp.step))
        return self

    def model(self, **kwargs):
        raise NotImplementedError()

    def add_summaries(self, **kwargs):
        raise NotImplementedError()


class ClassifySemi(Model):
    """Semi-supervised classification."""

    def __init__(self, train_dir: str, dataset: data.DataSet, nclass: int, **kwargs):
        self.nclass = nclass
        self.losses_dict = {'labeled_losses':[], 'unlabeled_losses_unscaled':[], 'unlabeled_losses_scaled':[], 'unlabeled_losses_multiplier':[]}
        Model.__init__(self, train_dir, dataset, nclass=nclass, **kwargs)

    def train_step(self, train_session, data_PLAX_labeled, data_PSAX_labeled, data_A4C_labeled, data_A2C_labeled, data_UsefulUnlabeled_labeled, data_unlabeled):
        
        raise NotImplementedError('train_step() in libml/train.py should be overwritten by train_step() in libfs/train.py')

#         x_PLAX, x_PSAX, x_A4C, x_A2C, x_UsefulUnlabeled, y = self.session.run([data_PLAX_labeled, data_PSAX_labeled, data_A4C_labeled, data_A2C_labeled, data_UsefulUnlabeled_labeled, data_unlabeled])
        
#         image_batch = np.concatenate((x_PLAX['image'], x_PSAX['image'], x_A4C['image'], x_A2C['image'], x_UsefulUnlabeled['image']), axis=0)
#         label_batch = np.concatenate((x_PLAX['label'], x_PSAX['label'], x_A4C['label'], x_A2C['label'], x_UsefulUnlabeled['label']), axis=0)
        
        
#         #to record the losses and directly save to disk, instead of accessing through tensorboard
#         self.tmp.step, labeled_losses_this_step, unlabeled_losses_unscaled_this_step, unlabeled_losses_scaled_this_step, unlabeled_losses_multiplier_this_step = train_session.run([self.ops.train_op, self.ops.update_step, self.ops.labeled_losses, self.ops.unlabeled_losses_unscaled, self.ops.unlabeled_losses_scaled, self.ops.unlabeled_losses_multiplier],
#                                           feed_dict={self.ops.x: image_batch,
#                                                      self.ops.y: y['image'],
#                                                      self.ops.label: label_batch,
#                                                      self.ops.unlabeled_label: y['label']})[1:]
                
#         self.losses_dict['labeled_losses'].append(labeled_losses_this_step)
#         self.losses_dict['unlabeled_losses_unscaled'].append(unlabeled_losses_unscaled_this_step)
#         self.losses_dict['unlabeled_losses_scaled'].append(unlabeled_losses_scaled_this_step)
#         self.losses_dict['unlabeled_losses_multiplier'].append(unlabeled_losses_multiplier_this_step)

        
    def train(self, train_nimg, report_nimg):
        raise NotImplementedError('train() in libml/train.py should be overwritten by train() in libfs/train.py')
        
        
#         PLAX_batch = FLAGS.PLAX_batch
#         PSAX_batch = FLAGS.PSAX_batch
#         A4C_batch = FLAGS.A4C_batch
#         A2C_batch = FLAGS.A2C_batch
#         UsefulUnlabeled_batch = FLAGS.UsefulUnlabeled_batch
        

        
#         train_PLAX_labeled = self.dataset.train_PLAX_labeled.batch(PLAX_batch).prefetch(16)
#         train_PLAX_labeled = train_PLAX_labeled.make_one_shot_iterator().get_next()

#         train_PSAX_labeled = self.dataset.train_PSAX_labeled.batch(PSAX_batch).prefetch(16)
#         train_PSAX_labeled = train_PSAX_labeled.make_one_shot_iterator().get_next()

#         train_A4C_labeled = self.dataset.train_A4C_labeled.batch(A4C_batch).prefetch(16)
#         train_A4C_labeled = train_A4C_labeled.make_one_shot_iterator().get_next()

#         train_A2C_labeled = self.dataset.train_A2C_labeled.batch(A2C_batch).prefetch(16)
#         train_A2C_labeled = train_A2C_labeled.make_one_shot_iterator().get_next()

#         train_UsefulUnlabeled_labeled = self.dataset.train_UsefulUnlabeled_labeled.batch(UsefulUnlabeled_batch).prefetch(16)
#         train_UsefulUnlabeled_labeled = train_UsefulUnlabeled_labeled.make_one_shot_iterator().get_next()

#         train_unlabeled = self.dataset.train_unlabeled.batch(self.batch_total).prefetch(16)
#         train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
                
                
              
#         scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
#                                                           pad_step_number=10), init_fn=self.init_fn)
            

#         with tf.Session(config=utils.get_config()) as sess:
#             self.session = sess
#             self.cache_eval()
#         with tf.train.MonitoredTrainingSession(
#                 scaffold=scaffold,
#                 checkpoint_dir=self.checkpoint_dir,
#                 config=utils.get_config(),
#                 save_checkpoint_steps=FLAGS.save_nimg,
#                 save_summaries_steps=report_nimg) as train_session:
#             self.session = train_session._tf_sess()
            

            
#             self.tmp.step = self.session.run(self.step)
#             while self.tmp.step < train_nimg:
#                 loop = trange(self.tmp.step % report_nimg, report_nimg, self.batch_total,
#                               leave=False, unit='img', unit_scale=self.batch_total,
#                               desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), FLAGS.train_epoch))
                
#                 for _ in loop:
#                     self.train_step(train_session, train_PLAX_labeled, train_PSAX_labeled, train_A4C_labeled, train_A2C_labeled, train_UsefulUnlabeled_labeled, train_unlabeled)
#                     while self.tmp.print_queue:
#                         loop.write(self.tmp.print_queue.pop(0))
                                            
                    
#             while self.tmp.print_queue:
#                 print(self.tmp.print_queue.pop(0))


    def cache_eval(self):
        """Cache datasets for computing eval stats."""

        def collect_samples(dataset):
            """Return numpy arrays of all the samples from a dataset."""
            it = dataset.batch(1).prefetch(16).make_one_shot_iterator().get_next()
            images, labels = [], []
            while 1:
                try:
                    v = self.session.run(it)
                except tf.errors.OutOfRangeError:
                    break
                images.append(v['image'])
                labels.append(v['label'])

            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            return images, labels
        

        if 'test' not in self.tmp.cache:
            self.tmp.cache.test = collect_samples(self.dataset.test)
            self.tmp.cache.stanford_test = collect_samples(self.dataset.stanford_test)
            self.tmp.cache.valid = collect_samples(self.dataset.valid)
            self.tmp.cache.train_PLAX_labeled = collect_samples(self.dataset.train_PLAX_labeled.take(300))
            self.tmp.cache.train_PSAX_labeled = collect_samples(self.dataset.train_PSAX_labeled.take(300))
            self.tmp.cache.train_A4C_labeled = collect_samples(self.dataset.train_A4C_labeled.take(300))
            self.tmp.cache.train_A2C_labeled = collect_samples(self.dataset.train_A2C_labeled.take(300))
            self.tmp.cache.train_UsefulUnlabeled_labeled = collect_samples(self.dataset.train_UsefulUnlabeled_labeled.take(300))

        

            
    def eval_stats(self, batch=None, feed_extra=None, classify_op=None):
        """Evaluate model on train, valid and test."""
#         batch = batch or FLAGS.batch
        batch = self.batch_total

        classify_op = self.ops.classify_op 
        classify_raw = self.ops.classify_raw
        
        accuracies = []

        subsets = ('train_PLAX_labeled', 'train_PSAX_labeled', 'train_A4C_labeled', 'train_A2C_labeled', 'train_UsefulUnlabeled_labeled', 'valid', 'test', 'stanford_test')

        for subset in subsets:
            print()            
            inference_start_time = time.time()
            
            images, labels = self.tmp.cache[subset]
            predicted = []
            predicted_raw = []
            
            #save predictions:
            predictions_save_dict = dict()
            
            for x in range(0, images.shape[0], batch):
                p, p_raw = self.session.run(
                    [classify_op, classify_raw],
                    feed_dict={
                        self.ops.x: images[x:x + batch],
                        **(feed_extra or {})
                    })
                
                predicted.append(p)
                predicted_raw.append(p_raw)
                
            predicted = np.concatenate(predicted, axis=0)
            predicted_raw = np.concatenate(predicted_raw, axis=0)
            
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            print('current subset: {}, total {} images, inference time total: {}'.format(subset, predicted.shape, inference_time))

                
            
            if subset == 'train_PLAX_labeled' or subset == 'train_PSAX_labeled' or subset == 'train_A4C_labeled' or subset == 'train_A2C_labeled' or subset == 'train_UsefulUnlabeled_labeled' :
                print('Current subset:{}, using PLAIN_accuracy'.format(subset), flush=True)
                print('ema predicted classes are: {}'.format(predicted.argmax(1)), flush=True)
                print('raw predicted classes are: {}'.format(predicted_raw.argmax(1)), flush=True)
                
                ema_accuracy_this_step = utils.calculate_accuracy(labels, predicted.argmax(1))
                raw_accuracy_this_step = utils.calculate_accuracy(labels, predicted_raw.argmax(1))

                accuracies.append(ema_accuracy_this_step)

                accuracies.append(raw_accuracy_this_step)
            
            elif subset == 'valid' :
                print('Current subset:{}, using balanced_accuracy'.format(subset), flush=True)

                val_ema_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(labels, predicted.argmax(1), 'all')
                val_raw_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(labels, predicted_raw.argmax(1), 'all')

                accuracies.append(val_ema_balanced_accuracy_this_step)

                accuracies.append(val_raw_balanced_accuracy_this_step)
                
                #save predictions to disk for every evaluation
                predictions_save_dict['ema_predictions'] = predicted
                predictions_save_dict['raw_predictions'] = predicted_raw
                predictions_save_dict['true_labels'] = labels

                predictions_save_dict['ema_balanced_accuracy'] = val_ema_balanced_accuracy_this_step

                predictions_save_dict['raw_balanced_accuracy'] = val_raw_balanced_accuracy_this_step

                utils.save_pickle(os.path.join(self.train_dir,'predictions'), subset + '_epoch_' + str(1 + (self.tmp.step // FLAGS.report_nimg)) + '_predictions.pkl', predictions_save_dict)
                
            elif subset == 'test':
                print('Current subset:{}, using balanced_accuracy'.format(subset), flush=True)

                test_ema_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(labels, predicted.argmax(1), 'all')
                test_raw_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(labels, predicted_raw.argmax(1), 'all')

                accuracies.append(test_ema_balanced_accuracy_this_step)

                accuracies.append(test_raw_balanced_accuracy_this_step)
                
                #save predictions to disk for every evaluation
                predictions_save_dict['ema_predictions'] = predicted
                predictions_save_dict['raw_predictions'] = predicted_raw
                predictions_save_dict['true_labels'] = labels

                predictions_save_dict['ema_balanced_accuracy'] = test_ema_balanced_accuracy_this_step

                predictions_save_dict['raw_balanced_accuracy'] = test_raw_balanced_accuracy_this_step

                utils.save_pickle(os.path.join(self.train_dir,'predictions'), subset + '_epoch_' + str(1 + (self.tmp.step // FLAGS.report_nimg)) + '_predictions.pkl', predictions_save_dict)
                
            
            elif subset == 'stanford_test':
                print('Current subset:{}, using PLAIN_accuracy'.format(subset), flush=True)
                print('ema predicted classes are: {}'.format(predicted.argmax(1)), flush=True)
                print('raw predicted classes are: {}'.format(predicted_raw.argmax(1)), flush=True)
                
                ema_accuracy_this_step = utils.calculate_accuracy(labels, predicted.argmax(1))
                raw_accuracy_this_step = utils.calculate_accuracy(labels, predicted_raw.argmax(1))

                accuracies.append(ema_accuracy_this_step)
                accuracies.append(raw_accuracy_this_step)
                
                
                predictions_save_dict['ema_predictions'] = predicted
                predictions_save_dict['raw_predictions'] = predicted_raw
                predictions_save_dict['true_labels'] = labels

                predictions_save_dict['ema_accuracy'] = ema_accuracy_this_step

                predictions_save_dict['raw_accuracy'] = raw_accuracy_this_step

                utils.save_pickle(os.path.join(self.train_dir,'predictions'), subset + '_epoch_' + str(1 + (self.tmp.step // FLAGS.report_nimg)) + '_predictions.pkl', predictions_save_dict)
            

                
            else:
                raise NameError('invalid subset name')
                
            
#             if subset == 'valid' and ema_balanced_accuracy_this_step > self.best_balanced_validation_accuracy_ema:
#                 self.best_balanced_validation_accuracy_ema = ema_balanced_accuracy_this_step
#                 #save checkpoint
#                 print('Found new record validation_ema!', flush = True)
#                 self.saver.save(self.session, '{}/best_balanced_validation_accuracy_ema.ckpt'.format(self.train_dir))

#             if subset == 'valid' and raw_balanced_accuracy_this_step > self.best_balanced_validation_accuracy_raw:
#                 self.best_balanced_validation_accuracy_raw = raw_balanced_accuracy_this_step
#                 #save checkpoint
#                 print('Found new record validation_raw!', flush = True)
#                 self.saver.save(self.session, '{}/best_balanced_validation_accuracy_raw.ckpt'.format(self.train_dir))

        
        if val_ema_balanced_accuracy_this_step > self.best_balanced_validation_accuracy_ema:
            self.best_balanced_validation_accuracy_ema = val_ema_balanced_accuracy_this_step
            self.best_balanced_test_accuracy_ema_at_max_val = test_ema_balanced_accuracy_this_step
            #save checkpoint
            print('Found new record validation_ema!', flush = True)
            self.saver.save(self.session, '{}/best_balanced_validation_accuracy_ema.ckpt'.format(self.train_dir))
        
        if val_raw_balanced_accuracy_this_step > self.best_balanced_validation_accuracy_raw:
            self.best_balanced_validation_accuracy_raw = val_raw_balanced_accuracy_this_step
            self.best_balanced_test_accuracy_raw_at_max_val = test_raw_balanced_accuracy_this_step
            print('Found new record validation_raw!', flush = True)
            self.saver.save(self.session, '{}/best_balanced_validation_accuracy_raw.ckpt'.format(self.train_dir))
        
        
        accuracies.append(self.best_balanced_validation_accuracy_ema)
        accuracies.append(self.best_balanced_test_accuracy_ema_at_max_val)
        
        accuracies.append(self.best_balanced_validation_accuracy_raw)
        accuracies.append(self.best_balanced_test_accuracy_raw_at_max_val)
        
        #save the losses of each batch until current epoch
        utils.save_pickle(os.path.join(self.train_dir, 'losses'), 'losses_dict.pkl', self.losses_dict)
        

        self.train_print('train nimg %-5d EMA Best Balanced Accuracy,  validation/test  %.2f  %.2f  ' % tuple([self.tmp.step] + [self.best_balanced_validation_accuracy_ema, self.best_balanced_test_accuracy_ema_at_max_val]))

#         self.train_print('train nimg %-5d RAW Best Balanced Accuracy,  validation/test  %.2f  %.2f  ' % tuple([self.tmp.step] + [self.best_balanced_validation_accuracy_raw, self.best_balanced_test_accuracy_raw_at_max_val]))
        
        
        self.train_print('train nimg %-5d  accuracy train_A4C_ema/valid_balanced_ema/test_balanced_ema/stanford_test_ema  %.2f  %.2f  %.2f  %.2f' % tuple([self.tmp.step] + [accuracies[4], accuracies[10], accuracies[12], accuracies[14]]))
        
        return np.array(accuracies, 'f')

    def add_summaries(self, feed_extra=None, **kwargs):
        del kwargs

        def gen_stats():
            return self.eval_stats(feed_extra=feed_extra)
        
        accuracies = tf.py_func(gen_stats, [], tf.float32)
        
        
        tf.summary.scalar('ema/train_PLAX_labeled/accuarcy', accuracies[0])
        tf.summary.scalar('raw/train_PLAX_labeled/accuarcy', accuracies[1])
        
        tf.summary.scalar('ema/train_PSAX_labeled/accuarcy', accuracies[2])
        tf.summary.scalar('raw/train_PSAX_labeled/accuarcy', accuracies[3])
        
        tf.summary.scalar('ema/train_A4C_labeled/accuarcy', accuracies[4])
        tf.summary.scalar('raw/train_A4C_labeled/accuarcy', accuracies[5])
        
        tf.summary.scalar('ema/train_A2C_labeled/accuarcy', accuracies[6])
        tf.summary.scalar('raw/train_A2C_labeled/accuarcy', accuracies[7])
        
        tf.summary.scalar('ema/train_UsefulUnlabeled_labeled/accuarcy', accuracies[8])
        tf.summary.scalar('raw/train_UsefulUnlabeled_labeled/accuarcy', accuracies[9])
        
        tf.summary.scalar('ema/valid/balanced_accuarcy', accuracies[10])
        tf.summary.scalar('raw/valid/balanced_accuarcy', accuracies[11])
    
        tf.summary.scalar('ema/test/balanced_accuarcy', accuracies[12])
        tf.summary.scalar('raw/test/balanced_accuarcy', accuracies[13])
        
        tf.summary.scalar('ema/stanford_test/accuarcy', accuracies[14])
        tf.summary.scalar('raw/stanford_test/accuarcy', accuracies[15])

        tf.summary.scalar('ema/val_max/balanced_accuarcy', accuracies[16])
        tf.summary.scalar('ema/test_at_val_max/balanced_accuarcy', accuracies[17])
        
        tf.summary.scalar('raw/val_max/balanced_accuarcy', accuracies[18])
        tf.summary.scalar('raw/test_at_val_max/balanced_accuarcy', accuracies[19])
        
            
            
            
            

