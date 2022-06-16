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

import tensorflow as tf
from absl import flags
from tqdm import trange
import numpy as np

from libml import utils
from libml.train import ClassifySemi

FLAGS = flags.FLAGS


class ClassifyFullySupervised(ClassifySemi):
    
    """Fully supervised classification.
    """
    
    def train_step(self, train_session, data_PLAX_labeled, data_PSAX_labeled, data_A4C_labeled, data_A2C_labeled, data_UsefulUnlabeled_labeled):

        x_PLAX, x_PSAX, x_A4C, x_A2C, x_UsefulUnlabeled = self.session.run([data_PLAX_labeled, data_PSAX_labeled, data_A4C_labeled, data_A2C_labeled, data_UsefulUnlabeled_labeled])

        image_batch = np.concatenate((x_PLAX['image'], x_PSAX['image'], x_A4C['image'], x_A2C['image'], x_UsefulUnlabeled['image']), axis=0)
        label_batch = np.concatenate((x_PLAX['label'], x_PSAX['label'], x_A4C['label'], x_A2C['label'], x_UsefulUnlabeled['label']), axis=0)
        
        self.tmp.step, labeled_losses_this_step = train_session.run([self.ops.train_op, self.ops.update_step, self.ops.labeled_losses],
                                          feed_dict={self.ops.x: image_batch,
                                                     self.ops.label: label_batch})[1:]
        
        
        print('labeled_losses_this_step: {}'.format(labeled_losses_this_step), flush=True)
        self.losses_dict['labeled_losses'].append(labeled_losses_this_step)
        self.losses_dict['unlabeled_losses_unscaled'].append(0)
        self.losses_dict['unlabeled_losses_scaled'].append(0)
        self.losses_dict['unlabeled_losses_multiplier'].append(0)

    def train(self, train_nimg, report_nimg):
        
        
        PLAX_batch = FLAGS.PLAX_batch
        PSAX_batch = FLAGS.PSAX_batch
        A4C_batch = FLAGS.A4C_batch
        A2C_batch = FLAGS.A2C_batch
        UsefulUnlabeled_batch = FLAGS.UsefulUnlabeled_batch
        
        train_PLAX_labeled = self.dataset.train_PLAX_labeled.batch(PLAX_batch).prefetch(30)
        train_PLAX_labeled = train_PLAX_labeled.make_one_shot_iterator().get_next()
        
        train_PSAX_labeled = self.dataset.train_PSAX_labeled.batch(PSAX_batch).prefetch(30)
        train_PSAX_labeled = train_PSAX_labeled.make_one_shot_iterator().get_next()
        
        train_A4C_labeled = self.dataset.train_A4C_labeled.batch(A4C_batch).prefetch(30)
        train_A4C_labeled = train_A4C_labeled.make_one_shot_iterator().get_next()
        
        train_A2C_labeled = self.dataset.train_A2C_labeled.batch(A2C_batch).prefetch(30)
        train_A2C_labeled = train_A2C_labeled.make_one_shot_iterator().get_next()
        
        train_UsefulUnlabeled_labeled = self.dataset.train_UsefulUnlabeled_labeled.batch(UsefulUnlabeled_batch).prefetch(50)
        train_UsefulUnlabeled_labeled = train_UsefulUnlabeled_labeled.make_one_shot_iterator().get_next()
        
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                          pad_step_number=10))
        
       
        with tf.Session(config=utils.get_config()) as sess:
            self.session = sess
            self.cache_eval()

        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=self.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.save_nimg,
                save_summaries_steps=report_nimg) as train_session:
            self.session = train_session._tf_sess()
            self.tmp.step = self.session.run(self.step)
        
            while self.tmp.step < train_nimg:
                loop = trange(self.tmp.step % report_nimg, report_nimg, self.batch_total,
                              leave=False, unit='img', unit_scale=self.batch_total,
                              desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), FLAGS.train_epoch))
                
                
                for _ in loop:
                    self.train_step(train_session, train_PLAX_labeled, train_PSAX_labeled, train_A4C_labeled, train_A2C_labeled, train_UsefulUnlabeled_labeled,)
        
                    while self.tmp.print_queue:
                        loop.write(self.tmp.print_queue.pop(0))
                        
                        
            while self.tmp.print_queue:
                print(self.tmp.print_queue.pop(0))

    
    