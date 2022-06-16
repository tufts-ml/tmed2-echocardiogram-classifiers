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
"""Input data for image models.
"""

import glob
import itertools
import os

import numpy as np
import tensorflow as tf
from absl import flags
from tqdm import tqdm

from libml import utils


_DATA_CACHE = None
DATA_DIR = os.environ['ML_DATA']
flags.DEFINE_string('dataset', 'cifar10.1@4000-5000', 'Data to train on.')
flags.DEFINE_integer('para_parse', 4, 'Parallel parsing.')
flags.DEFINE_integer('para_augment', 4, 'Parallel augmentation.')
flags.DEFINE_integer('shuffle', 8192, 'Size of dataset shuffling.')
flags.DEFINE_string('p_unlabeled', '', 'Probability distribution of unlabeled.')
flags.DEFINE_bool('whiten', False, 'Whether to normalize images.')
FLAGS = flags.FLAGS


def record_parse(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)})
    image = tf.image.decode_image(features['image'])
    image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
    label = features['label']
    return dict(image=image, label=label)



def default_parse(dataset: tf.data.Dataset, parse_fn=record_parse) -> tf.data.Dataset:
    para = 4 * max(1, len(utils.get_available_gpus())) * FLAGS.para_parse
    return dataset.map(parse_fn, num_parallel_calls=para)


def dataset(filenames: list) -> tf.data.Dataset:
    filenames = sorted(sum([glob.glob(x) for x in filenames], []))
    if not filenames:
        raise ValueError('Empty dataset, did you mount gcsfuse bucket?')
    return tf.data.TFRecordDataset(filenames)


def memoize(dataset: tf.data.Dataset) -> tf.data.Dataset:
    data = []
    with tf.Session(config=utils.get_config()) as session:
        dataset = dataset.prefetch(16)
        it = dataset.make_one_shot_iterator().get_next()
        try:
            while 1:
                data.append(session.run(it))
        except tf.errors.OutOfRangeError:
            pass
    images = np.stack([x['image'] for x in data])
    labels = np.stack([x['label'] for x in data])

    def tf_get(index):
        def get(index):
            return images[index], labels[index]

        image, label = tf.py_func(get, [index], [tf.float32, tf.int64])
        return dict(image=image, label=label)

    dataset = tf.data.Dataset.range(len(data)).repeat()
    dataset = dataset.shuffle(len(data) if len(data) < FLAGS.shuffle else FLAGS.shuffle)
    return dataset.map(tf_get)


def augment_mirror(x):
    return tf.image.random_flip_left_right(x)


def augment_shift(x, w):
    y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
    return tf.random_crop(y, tf.shape(x))


def augment_noise(x, std):
    return x + std * tf.random_normal(tf.shape(x), dtype=x.dtype)


def compute_mean_std(data: tf.data.Dataset):
    print('Inside data.py, compute_mean_std is called', flush = True)
    
    data = data.map(lambda x: x['image']).batch(1024).prefetch(1)
    data = data.make_one_shot_iterator().get_next()
    count = 0
    stats = []
    with tf.Session(config=utils.get_config()) as sess:
        def iterator():
            while True:
                try:
                    yield sess.run(data)
                except tf.errors.OutOfRangeError:
                    break

        for batch in tqdm(iterator(), unit='kimg', desc='Computing dataset mean and std'):
            ratio = batch.shape[0] / 1024.
            count += ratio
            stats.append((batch.mean((0, 1, 2)) * ratio, (batch ** 2).mean((0, 1, 2)) * ratio))
    mean = sum(x[0] for x in stats) / count
    sigma = sum(x[1] for x in stats) / count - mean ** 2
    std = np.sqrt(sigma)
    print('Mean %s  Std: %s' % (mean, std))
    return mean, std


class DataSet:
    def __init__(self, name, train_PLAX_labeled, train_PSAX_labeled, train_A4C_labeled, train_A2C_labeled, train_UsefulUnlabeled_labeled, train_unlabeled, test, stanford_test, valid, eval_labeled, eval_unlabeled,
                 height=112, width=112, colors=3, nclass=5, mean=0, std=1, p_labeled=None, p_unlabeled=None):
        self.name = name

        self.train_PLAX_labeled = train_PLAX_labeled
        self.train_PSAX_labeled = train_PSAX_labeled
        self.train_A4C_labeled = train_A4C_labeled
        self.train_A2C_labeled = train_A2C_labeled
        self.train_UsefulUnlabeled_labeled = train_UsefulUnlabeled_labeled

        self.train_unlabeled = train_unlabeled
        self.eval_labeled = eval_labeled
        self.eval_unlabeled = eval_unlabeled
        self.test = test
        self.stanford_test = stanford_test
        self.valid = valid
        self.height = height
        self.width = width
        self.colors = colors
        self.nclass = nclass
        self.mean = mean
        self.std = std
        self.p_labeled = p_labeled
        self.p_unlabeled = p_unlabeled

    @classmethod
    def creator(cls, name, train_PLAX_labeled_files, train_PSAX_labeled_files, train_A4C_labeled_files, train_A2C_labeled_files, train_UsefulUnlabeled_labeled_files, train_unlabeled_files, valid_files, test_files, stanford_test_files, augment, parse_fn=default_parse, do_memoize=True, colors=1,
                nclass=5, height=112, width=112):
                
        if not isinstance(augment, list):
            augment = [augment] * 2
        
        train_PLAX_labeled_files = [os.path.join(DATA_DIR, x) for x in train_PLAX_labeled_files]
        train_PSAX_labeled_files = [os.path.join(DATA_DIR, x) for x in train_PSAX_labeled_files]
        train_A4C_labeled_files = [os.path.join(DATA_DIR, x) for x in train_A4C_labeled_files]
        train_A2C_labeled_files = [os.path.join(DATA_DIR, x) for x in train_A2C_labeled_files]
        train_UsefulUnlabeled_labeled_files = [os.path.join(DATA_DIR, x) for x in train_UsefulUnlabeled_labeled_files]

        train_unlabeled_files = [os.path.join(DATA_DIR, x) for x in train_unlabeled_files]
        valid_files = [os.path.join(DATA_DIR, x) for x in valid_files]
        test_files = [os.path.join(DATA_DIR, x) for x in test_files]
        stanford_test_files = [os.path.join(DATA_DIR, x) for x in stanford_test_files]

#         print('Using {} as labeled train set'.format(train_labeled_files), flush = True)
#         print('Using {} as unlabeled train set'.format(train_unlabeled_files), flush = True)
#         print('Using {} as validation set'.format(valid_files), flush = True)
#         print('Using {} as test set'.format(test_files), flush = True)


        fn = memoize if do_memoize else lambda x: x.repeat().shuffle(FLAGS.shuffle)

        def create():
            p_labeled = p_unlabeled = None
            para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment

            if FLAGS.p_unlabeled:
                sequence = FLAGS.p_unlabeled.split(',')
                p_unlabeled = np.array(list(map(float, sequence)), dtype=np.float32)
                p_unlabeled /= np.max(p_unlabeled)

            train_PLAX_labeled = parse_fn(dataset(train_PLAX_labeled_files))
            train_PSAX_labeled = parse_fn(dataset(train_PSAX_labeled_files))
            train_A4C_labeled = parse_fn(dataset(train_A4C_labeled_files))
            train_A2C_labeled = parse_fn(dataset(train_A2C_labeled_files))
            train_UsefulUnlabeled_labeled = parse_fn(dataset(train_UsefulUnlabeled_labeled_files))

            train_unlabeled = parse_fn(dataset(train_unlabeled_files))
            valid = parse_fn(dataset(valid_files))
            test = parse_fn(dataset(test_files))
            stanford_test = parse_fn(dataset(stanford_test_files))
            
            if FLAGS.whiten:
                raise NameError('TODO')
                mean, std = compute_mean_std(train_labeled.concatenate(train_unlabeled))
            else:
                mean, std = 0, 1

            return cls(name,
                       train_PLAX_labeled=fn(train_PLAX_labeled).map(augment[0], para),
                       train_PSAX_labeled=fn(train_PSAX_labeled).map(augment[0], para),
                       train_A4C_labeled=fn(train_A4C_labeled).map(augment[0], para),
                       train_A2C_labeled=fn(train_A2C_labeled).map(augment[0], para),
                       train_UsefulUnlabeled_labeled=fn(train_UsefulUnlabeled_labeled).map(augment[0], para),

                       train_unlabeled=fn(train_unlabeled).map(augment[1], para),
                       eval_labeled=None,
                       eval_unlabeled=None,
                       valid=valid,
                       test=test,
                       stanford_test=stanford_test,
                       nclass=nclass, colors=colors, p_labeled=p_labeled, p_unlabeled=p_unlabeled,
                       height=height, width=width, mean=mean, std=std)

        return name, create


augment_echo = lambda x: dict(image=augment_shift(augment_mirror(x['image']), 4), label=x['label'])
