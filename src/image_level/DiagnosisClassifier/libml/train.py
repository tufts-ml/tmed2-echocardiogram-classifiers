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


FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', './experiments',
                    'Folder where to save training data.')
flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
flags.DEFINE_integer('batch', 240, 'total batch size.')
flags.DEFINE_integer('train_epoch', 500, 'How many epoch to train.')
flags.DEFINE_integer('nimg_per_epoch', 10066, 'Training duration in number of samples.')
flags.DEFINE_integer('report_nimg', 10066, 'Report summary period in number of samples.')
flags.DEFINE_integer('save_nimg', 10066, 'Save checkpoint period in number of samples.')

flags.DEFINE_integer('keep_ckpt', 1, 'Number of checkpoints to keep.')
flags.DEFINE_bool('reset_global_step', False, 'initialized from pretrained weights')
flags.DEFINE_string('load_ckpt', "None", 'Checkpoint to initialize from')
flags.DEFINE_string('checkpoint_exclude_scopes', "None", 'Comma-separated list of scopes of variables to exclude when restoring')
flags.DEFINE_string('trainable_scopes', "None", 'Comma-separated list of scopes of variables to train')

class Model:
    def __init__(self, train_dir: str, dataset: data.DataSet, **kwargs):
        print('train_dir: {}'.format(train_dir), flush=True)
        self.train_dir = os.path.join(train_dir, self.experiment_name(**kwargs))
        self.params = EasyDict(kwargs)
        self.dataset = dataset
        self.session = None
        self.tmp = EasyDict(print_queue=[], cache=EasyDict())
        self.step = tf.train.get_or_create_global_step()
        self.ops = self.model(**kwargs)
        self.ops.update_step = tf.assign_add(self.step, FLAGS.batch)
        self.add_summaries(**kwargs)

#         self.losses_dict = {'labeled_losses':[], 'unlabeled_losses_unscaled':[], 'unlabeled_losses_scaled':[], 'unlabeled_losses_multiplier':[]}
        self.diagnosis_best_balanced_validation_accuracy_raw = 0 #initialize to 0
        self.diagnosis_best_balanced_validation_accuracy_ema = 0 #initialize to 0
        self.diagnosis_best_balanced_test_accuracy_raw_at_max_val = 0
        self.diagnosis_best_balanced_test_accuracy_ema_at_max_val = 0
        
        self.view_best_balanced_validation_accuracy_raw = 0 #initialize to 0
        self.view_best_balanced_validation_accuracy_ema = 0 #initialize to 0
        self.view_best_balanced_test_accuracy_raw_at_max_val = 0
        self.view_best_balanced_test_accuracy_ema_at_max_val = 0
        
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
        args = [x + str(y) for x, y in sorted(kwargs.items()) if x != 'continued_training' if x != 'arch' if x!='smoothing']
        print('args: {}'.format(args), flush=True)
        print('self.__class__.__name__: {}'.format(self.__class__.__name__), flush=True)
#         return '_'.join([self.__class__.__name__] + args)
        return '_'.join(args)

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

    def __init__(self, train_dir: str, dataset: data.DataSet, diagnosis_nclass:int, view_nclass:int, **kwargs):

        self.diagnosis_nclass = diagnosis_nclass
        self.view_nclass = view_nclass
        self.losses_dict = {'total_losses':[], 'diagnosis_loss':[], 'unscaled_view_loss':[], 'scaled_view_loss':[]}
        Model.__init__(self, train_dir, dataset, diagnosis_nclass=diagnosis_nclass, view_nclass=view_nclass, **kwargs)

    def train_step(self, train_session, data_labeled, data_unlabeled):
        raise NotImplementedError('train_step() in libml/train.py should be overwritten by train_step() in libfs/train.py')


        
    def train(self, train_nimg, report_nimg):
        raise NotImplementedError('train() in libml/train.py should be overwritten by train() in libfs/train.py')

        
        

    def cache_eval(self):
        """Cache datasets for computing eval stats."""

        def collect_samples(dataset):
            """Return numpy arrays of all the samples from a dataset."""
            it = dataset.batch(1).prefetch(16).make_one_shot_iterator().get_next()
            images, diagnosis_labels, view_labels = [], [], []
            while 1:
                try:
                    v = self.session.run(it)
                except tf.errors.OutOfRangeError:
                    break
                images.append(v['image'])
                diagnosis_labels.append(v['diagnosis_label'])
                view_labels.append(v['view_label'])

            images = np.concatenate(images, axis=0)
            diagnosis_labels = np.concatenate(diagnosis_labels, axis=0)
            view_labels = np.concatenate(view_labels, axis=0)
            
            return images, diagnosis_labels, view_labels
        
        def collect_samples_stanford_test(dataset):
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
            self.tmp.cache.train_labeled = collect_samples(self.dataset.train_labeled.take(300))
            self.tmp.cache.valid = collect_samples(self.dataset.valid)
            self.tmp.cache.test = collect_samples(self.dataset.test)
            self.tmp.cache.stanford_test = collect_samples_stanford_test(self.dataset.stanford_test)

            
    def eval_stats(self, batch=None, feed_extra=None, classify_op=None):
        """Evaluate model on train, valid and test."""
        batch = FLAGS.batch

        diagnosis_inference_raw = self.ops.diagnosis_inference_raw
        diagnosis_inference_ema = self.ops.diagnosis_inference_ema
        view_inference_raw = self.ops.view_inference_raw
        view_inference_ema = self.ops.view_inference_ema
        
        accuracies = []
        
        subsets = ('train_labeled', 'valid', 'test', 'stanford_test')

        for subset in subsets:
            if subset in ['train_labeled', 'valid', 'test']:
                images, diagnosis_labels, view_labels = self.tmp.cache[subset]
                diagnosis_predicted_ema = []
                diagnosis_predicted_raw = []

                view_predicted_ema = []
                view_predicted_raw = []

                #save predictions:
                diagnosis_predictions_save_dict = dict()
                view_predictions_save_dict = dict()

                for x in range(0, images.shape[0], batch):
                    diagnosis_p_ema, diagnosis_p_raw, view_p_ema, view_p_raw  = self.session.run(
                        [diagnosis_inference_ema, diagnosis_inference_raw, view_inference_ema, view_inference_raw],
                        feed_dict={
                            self.ops.x: images[x:x + batch],
                            **(feed_extra or {})
                        })

                    diagnosis_predicted_ema.append(diagnosis_p_ema)
                    diagnosis_predicted_raw.append(diagnosis_p_raw)
                    view_predicted_ema.append(view_p_ema)
                    view_predicted_raw.append(view_p_raw)


                diagnosis_predicted_ema = np.concatenate(diagnosis_predicted_ema, axis=0)
                diagnosis_predicted_raw = np.concatenate(diagnosis_predicted_raw, axis=0)

                view_predicted_ema = np.concatenate(view_predicted_ema, axis=0)
                view_predicted_raw = np.concatenate(view_predicted_raw, axis=0)     

                
                if subset == 'train_labeled':
                    print('Current subset:{}, using balanced_accuracy'.format(subset), flush=True)
                    diagnosis_ema_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(diagnosis_labels, diagnosis_predicted_ema.argmax(1), 'DiagnosisClassification', 'all')
                    diagnosis_raw_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(diagnosis_labels, diagnosis_predicted_raw.argmax(1), 'DiagnosisClassification', 'all')

                    view_ema_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(view_labels, view_predicted_ema.argmax(1), 'ViewClassification', 'all')
                    view_raw_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(view_labels, view_predicted_raw.argmax(1), 'ViewClassification', 'all')


                    accuracies.append(diagnosis_ema_balanced_accuracy_this_step)
                    accuracies.append(diagnosis_raw_balanced_accuracy_this_step)
                    accuracies.append(view_ema_balanced_accuracy_this_step)
                    accuracies.append(view_raw_balanced_accuracy_this_step)
                    
                
                elif subset == 'valid':
                    print('Current subset:{}, using balanced_accuracy'.format(subset), flush=True)
                    val_diagnosis_ema_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(diagnosis_labels, diagnosis_predicted_ema.argmax(1), 'DiagnosisClassification', 'all')
                    val_diagnosis_raw_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(diagnosis_labels, diagnosis_predicted_raw.argmax(1), 'DiagnosisClassification', 'all')

                    val_view_ema_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(view_labels, view_predicted_ema.argmax(1), 'ViewClassification', 'all')
                    val_view_raw_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(view_labels, view_predicted_raw.argmax(1), 'ViewClassification', 'all')


                    accuracies.append(val_diagnosis_ema_balanced_accuracy_this_step)
                    accuracies.append(val_diagnosis_raw_balanced_accuracy_this_step)
                    accuracies.append(val_view_ema_balanced_accuracy_this_step)
                    accuracies.append(val_view_raw_balanced_accuracy_this_step)


                    #save diagnosis predictions to disk for every evaluation
                    diagnosis_predictions_save_dict['ema_predictions'] = diagnosis_predicted_ema
                    diagnosis_predictions_save_dict['raw_predictions'] = diagnosis_predicted_raw
                    diagnosis_predictions_save_dict['true_labels'] = diagnosis_labels

                    diagnosis_predictions_save_dict['ema_balanced_accuracy'] = val_diagnosis_ema_balanced_accuracy_this_step

                    diagnosis_predictions_save_dict['raw_balanced_accuracy'] = val_diagnosis_raw_balanced_accuracy_this_step

                    utils.save_pickle(os.path.join(self.train_dir,'DiagnosisClassification_predictions'), subset + '_epoch_' + str(1 + (self.tmp.step // FLAGS.report_nimg)) + '_predictions.pkl', diagnosis_predictions_save_dict)

                    #save diagnosis predictions to disk for every evaluation
                    view_predictions_save_dict['ema_predictions'] = view_predicted_ema
                    view_predictions_save_dict['raw_predictions'] = view_predicted_raw
                    view_predictions_save_dict['true_labels'] = view_labels

                    view_predictions_save_dict['ema_balanced_accuracy'] = val_view_ema_balanced_accuracy_this_step

                    view_predictions_save_dict['raw_balanced_accuracy'] = val_view_raw_balanced_accuracy_this_step

                    utils.save_pickle(os.path.join(self.train_dir,'ViewClassification_predictions'), subset + '_epoch_' + str(1 + (self.tmp.step // FLAGS.report_nimg)) + '_predictions.pkl', view_predictions_save_dict)

                    
                elif subset == 'test':
                    print('Current subset:{}, using balanced_accuracy'.format(subset), flush=True)
                    test_diagnosis_ema_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(diagnosis_labels, diagnosis_predicted_ema.argmax(1), 'DiagnosisClassification', 'all')
                    test_diagnosis_raw_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(diagnosis_labels, diagnosis_predicted_raw.argmax(1), 'DiagnosisClassification', 'all')

                    test_view_ema_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(view_labels, view_predicted_ema.argmax(1), 'ViewClassification', 'all')
                    test_view_raw_balanced_accuracy_this_step, _ = utils.calculate_balanced_accuracy(view_labels, view_predicted_raw.argmax(1), 'ViewClassification', 'all')


                    accuracies.append(test_diagnosis_ema_balanced_accuracy_this_step)
                    accuracies.append(test_diagnosis_raw_balanced_accuracy_this_step)
                    accuracies.append(test_view_ema_balanced_accuracy_this_step)
                    accuracies.append(test_view_raw_balanced_accuracy_this_step)


                    #save diagnosis predictions to disk for every evaluation
                    diagnosis_predictions_save_dict['ema_predictions'] = diagnosis_predicted_ema
                    diagnosis_predictions_save_dict['raw_predictions'] = diagnosis_predicted_raw
                    diagnosis_predictions_save_dict['true_labels'] = diagnosis_labels

                    diagnosis_predictions_save_dict['ema_balanced_accuracy'] = test_diagnosis_ema_balanced_accuracy_this_step

                    diagnosis_predictions_save_dict['raw_balanced_accuracy'] = test_diagnosis_raw_balanced_accuracy_this_step

                    utils.save_pickle(os.path.join(self.train_dir,'DiagnosisClassification_predictions'), subset + '_epoch_' + str(1 + (self.tmp.step // FLAGS.report_nimg)) + '_predictions.pkl', diagnosis_predictions_save_dict)

                    #save diagnosis predictions to disk for every evaluation
                    view_predictions_save_dict['ema_predictions'] = view_predicted_ema
                    view_predictions_save_dict['raw_predictions'] = view_predicted_raw
                    view_predictions_save_dict['true_labels'] = view_labels

                    view_predictions_save_dict['ema_balanced_accuracy'] = test_view_ema_balanced_accuracy_this_step

                    view_predictions_save_dict['raw_balanced_accuracy'] = test_view_raw_balanced_accuracy_this_step

                    utils.save_pickle(os.path.join(self.train_dir,'ViewClassification_predictions'), subset + '_epoch_' + str(1 + (self.tmp.step // FLAGS.report_nimg)) + '_predictions.pkl', view_predictions_save_dict)
                
                
                
            elif subset in ['stanford_test']:
                print('Current subset:{}, using PLAIN_accuracy'.format(subset), flush=True)
                images, labels = self.tmp.cache[subset]
                predicted = []
                predicted_raw = []

                #save predictions:
                predictions_save_dict = dict()

                for x in range(0, images.shape[0], batch):
                    p, p_raw = self.session.run(
                        [view_inference_ema, view_inference_raw],
                        feed_dict={
                            self.ops.x: images[x:x + batch],
                            **(feed_extra or {})
                        })

                    predicted.append(p)
                    predicted_raw.append(p_raw)

                predicted = np.concatenate(predicted, axis=0)
                predicted_raw = np.concatenate(predicted_raw, axis=0)

                print('stanford test ema predicted classes are: {}'.format(predicted.argmax(1)), flush=True)
                print('stanford test raw predicted classes are: {}'.format(predicted_raw.argmax(1)), flush=True)

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
        
        
        

        #save best epoch for diagnosis
        if val_diagnosis_ema_balanced_accuracy_this_step > self.diagnosis_best_balanced_validation_accuracy_ema:
            self.diagnosis_best_balanced_validation_accuracy_ema = val_diagnosis_ema_balanced_accuracy_this_step
            self.diagnosis_best_balanced_test_accuracy_ema_at_max_val = test_diagnosis_ema_balanced_accuracy_this_step
            #save checkpoint
            print('Found new record diagnosis validation_ema!', flush = True)
            self.saver.save(self.session, '{}/diagnosis_best_balanced_validation_accuracy_ema.ckpt'.format(self.train_dir))
                        
            
        if val_diagnosis_raw_balanced_accuracy_this_step > self.diagnosis_best_balanced_validation_accuracy_raw:
            self.diagnosis_best_balanced_validation_accuracy_raw = val_diagnosis_raw_balanced_accuracy_this_step
            self.diagnosis_best_balanced_test_accuracy_raw_at_max_val = test_diagnosis_raw_balanced_accuracy_this_step
            #save checkpoint
            print('Found new record validation_raw!', flush = True)
            self.saver.save(self.session, '{}/diagnosis_best_balanced_validation_accuracy_raw.ckpt'.format(self.train_dir))

        #save best epoch for view
        if val_view_ema_balanced_accuracy_this_step > self.view_best_balanced_validation_accuracy_ema:
            self.view_best_balanced_validation_accuracy_ema = val_view_ema_balanced_accuracy_this_step
            self.view_best_balanced_test_accuracy_ema_at_max_val = test_view_ema_balanced_accuracy_this_step
            #save checkpoint
            print('Found new record view validation_ema!', flush = True)
            self.saver.save(self.session, '{}/view_best_balanced_validation_accuracy_ema.ckpt'.format(self.train_dir))

        if val_view_raw_balanced_accuracy_this_step > self.view_best_balanced_validation_accuracy_raw:
            self.view_best_balanced_validation_accuracy_raw = val_view_raw_balanced_accuracy_this_step
            self.view_best_balanced_test_accuracy_raw_at_max_val = test_view_raw_balanced_accuracy_this_step
            #save checkpoint
            print('Found new record validation_raw!', flush = True)
            self.saver.save(self.session, '{}/view_best_balanced_validation_accuracy_raw.ckpt'.format(self.train_dir))

            
        accuracies.append(self.diagnosis_best_balanced_validation_accuracy_ema)
        accuracies.append(self.diagnosis_best_balanced_test_accuracy_ema_at_max_val)
        
        accuracies.append(self.diagnosis_best_balanced_validation_accuracy_raw)
        accuracies.append(self.diagnosis_best_balanced_test_accuracy_raw_at_max_val)
        
        accuracies.append(self.view_best_balanced_validation_accuracy_ema)
        accuracies.append(self.view_best_balanced_test_accuracy_ema_at_max_val)
        
        accuracies.append(self.view_best_balanced_validation_accuracy_raw)
        accuracies.append(self.view_best_balanced_test_accuracy_raw_at_max_val)
        
        
        #save the losses of each batch until current epoch
        utils.save_pickle(os.path.join(self.train_dir, 'losses'), 'losses_dict.pkl', self.losses_dict)

        self.train_print('train nimg %-5d Diagnosis EMA Best Balanced Accuracy,  validation/test  %.2f  %.2f  ' % tuple([self.tmp.step] + [self.diagnosis_best_balanced_validation_accuracy_ema, self.diagnosis_best_balanced_test_accuracy_ema_at_max_val]))
        
        self.train_print('train nimg %-5d VIEW EMA Best Balanced Accuracy,  validation/test  %.2f  %.2f  ' % tuple([self.tmp.step] + [self.view_best_balanced_validation_accuracy_ema, self.view_best_balanced_test_accuracy_ema_at_max_val]))

#         self.train_print('train nimg %-5d RAW Best Balanced Accuracy,  validation/test  %.2f  %.2f  ' % tuple([self.tmp.step] + [self.best_balanced_validation_accuracy_raw, self.best_balanced_test_accuracy_raw_at_max_val]))

        self.train_print('train nimg %-5d  Diagnosis EMA: train_labeled/valid/test/  %.2f  %.2f  %.2f ' % tuple([self.tmp.step] + [accuracies[0], accuracies[4], accuracies[8]]))
        
        self.train_print('train nimg %-5d  View EMA : train_labeled/valid/test/stanford_test  %.2f  %.2f  %.2f %.2f ' % tuple([self.tmp.step] + [accuracies[2], accuracies[6], accuracies[10], accuracies[12]]))
        
        
        return np.array(accuracies, 'f')



    def add_summaries(self, feed_extra=None, **kwargs):
        del kwargs

        def gen_stats():
            return self.eval_stats(feed_extra=feed_extra)
        
        accuracies = tf.py_func(gen_stats, [], tf.float32)
        
        tf.summary.scalar('diagnosis_ema/train_labeled', accuracies[0])
        tf.summary.scalar('diagnosis_ema/valid', accuracies[4])
        tf.summary.scalar('diagnosis_ema/test', accuracies[8])
        
        tf.summary.scalar('diagnosis_raw/train_labeled', accuracies[1])
        tf.summary.scalar('diagnosis_raw/valid', accuracies[5])
        tf.summary.scalar('diagnosis_raw/test', accuracies[9])
        
        tf.summary.scalar('view_ema/train_labeled', accuracies[2])
        tf.summary.scalar('view_ema/valid', accuracies[6])
        tf.summary.scalar('view_ema/test', accuracies[10])
        tf.summary.scalar('view_ema/stanford_test', accuracies[12])
        
        tf.summary.scalar('view_raw/train_labeled', accuracies[3])
        tf.summary.scalar('view_raw/valid', accuracies[7])
        tf.summary.scalar('view_raw/test', accuracies[11])
        tf.summary.scalar('view_raw/stanford_test', accuracies[13])

        tf.summary.scalar('diagnosis_ema/val_max/balanced_accuarcy', accuracies[14])
        tf.summary.scalar('diagnosis_ema/test_at_val_max/balanced_accuarcy', accuracies[15])
        
        tf.summary.scalar('diagnosis_raw/val_max/balanced_accuarcy', accuracies[16])
        tf.summary.scalar('diagnosis_raw/test_at_val_max/balanced_accuarcy', accuracies[17])
        
        tf.summary.scalar('view_ema/val_max/balanced_accuarcy', accuracies[18])
        tf.summary.scalar('view_ema/test_at_val_max/balanced_accuarcy', accuracies[19])
        
        tf.summary.scalar('view_raw/val_max/balanced_accuarcy', accuracies[20])
        tf.summary.scalar('view_raw/test_at_val_max/balanced_accuarcy', accuracies[21])
        
        

            
            
            
            
            

