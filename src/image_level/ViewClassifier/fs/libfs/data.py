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

import os

from absl import flags
from libml import utils
from libml import data

FLAGS = flags.FLAGS


class DataSetFS(data.DataSet):
    @classmethod
    def creator(cls, name, train_PLAX_labeled_files, train_PSAX_labeled_files, train_A4C_labeled_files, train_A2C_labeled_files, train_UsefulUnlabeled_labeled_files, valid_files, test_files, stanford_test_files, augment, parse_fn=data.default_parse, memoize_fn=data.memoize,
                colors=3, nclass=5, height=112, width=112):
        

        train_PLAX_labeled_files = [os.path.join(data.DATA_DIR, x) for x in train_PLAX_labeled_files]
        train_PSAX_labeled_files = [os.path.join(data.DATA_DIR, x) for x in train_PSAX_labeled_files]
        train_A4C_labeled_files = [os.path.join(data.DATA_DIR, x) for x in train_A4C_labeled_files]
        train_A2C_labeled_files = [os.path.join(data.DATA_DIR, x) for x in train_A2C_labeled_files]
        train_UsefulUnlabeled_labeled_files = [os.path.join(data.DATA_DIR, x) for x in train_UsefulUnlabeled_labeled_files]

        valid_files = [os.path.join(data.DATA_DIR, x) for x in valid_files]
        test_files = [os.path.join(data.DATA_DIR, x) for x in test_files]
        stanford_test_files = [os.path.join(data.DATA_DIR, x) for x in stanford_test_files]

#         fn = data.memoize if do_memoize else lambda x: x.repeat().shuffle(FLAGS.shuffle)
        fn = memoize_fn

        def create():
            para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment
            
            train_PLAX_labeled = parse_fn(data.dataset(train_PLAX_labeled_files))
            train_PSAX_labeled = parse_fn(data.dataset(train_PSAX_labeled_files))
            train_A4C_labeled = parse_fn(data.dataset(train_A4C_labeled_files))
            train_A2C_labeled = parse_fn(data.dataset(train_A2C_labeled_files))
            train_UsefulUnlabeled_labeled = parse_fn(data.dataset(train_UsefulUnlabeled_labeled_files))

            valid = parse_fn(data.dataset(valid_files))
            test = parse_fn(data.dataset(test_files))
            stanford_test = parse_fn(data.dataset(stanford_test_files))
            
                    
            if FLAGS.whiten:
                raise NameError('TODO')
                mean, std = data.compute_mean_std(train_labeled)
            else:
                mean, std = 0, 1

            return cls(name,
                       train_PLAX_labeled=fn(train_PLAX_labeled).map(augment, para),
                       train_PSAX_labeled=fn(train_PSAX_labeled).map(augment, para),
                       train_A4C_labeled=fn(train_A4C_labeled).map(augment, para),
                       train_A2C_labeled=fn(train_A2C_labeled).map(augment, para),
                       train_UsefulUnlabeled_labeled=fn(train_UsefulUnlabeled_labeled).map(augment, para),

                       train_unlabeled=None,
#                        eval_labeled=train_labeled.take(5000),  # No need to to eval on everything.
                       eval_labeled=None,
                       eval_unlabeled=None,
                       valid=valid,
                       test=test,
                       stanford_test=stanford_test,
                       nclass=nclass, colors=colors, height=height, width=width, mean=mean, std=std)

        return name, create



