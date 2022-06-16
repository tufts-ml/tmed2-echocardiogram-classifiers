#data set class

import logging
import math

# import pandas as pd
import numpy as np
import glob

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)


def read_npy(filepath):
    with open(filepath, 'rb') as f:
        data = np.load(f)
    return data


def normalizing_label(original_label):
    if original_label==0:
#         print('original_label is {}, return 0'.format(original_label))
        return 0
    elif original_label==1:
#         print('original_label is {}, return 1'.format(original_label))
        return 1
    elif original_label == 2 or original_label==3 or original_label==4:
#         print('original_label is {}, return 2'.format(original_label))
        return 2
    elif original_label == -1:
#         print('original_label is {}, return -1'.format(original_label))
        return -1
    
    
# class TransformFixMatch(object):
#     def __init__(self, mean=0, std=1):
#         self.weak = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=32,
#                                   padding=int(32*0.125),
#                                   padding_mode='reflect')])
#         self.strong = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=32,
#                                   padding=int(32*0.125),
#                                   padding_mode='reflect'),
#             RandAugmentMC(n=2, m=10)])
#         self.normalize = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std)])

#     def __call__(self, x):
#         weak = self.weak(Image.fromarray(x.squeeze()))
#         strong = self.strong(Image.fromarray(x.squeeze())h)
#         return self.normalize(weak), self.normalize(strong)
    
    
class Echo_LabeledDataset(Dataset):
    
    def __init__(self, image_npy_path, label_npy_path, transform):


        self.image_array = read_npy(image_npy_path)
        self.label_array = read_npy(label_npy_path)
        self.normalized_label_array = np.array(list(map(normalizing_label, self.label_array)))
        
        self.transform = transform
        
        
        
    def __getitem__(self, index):
        img, label, normalized_label = self.image_array[index], self.label_array[index], self.normalized_label_array[index]
#         print(img.shape)
        img = Image.fromarray(img)
        
        
        return self.transform(img), label, normalized_label
        
        
    def __len__(self):
        return len(self.label_array)

    
    
        
    
    
class Echo_UnlabeledDataset(Dataset):
    
    def __init__(self, image_npy_path, label_npy_path):


        self.image_array = read_npy(image_npy_path)
        self.label_array = read_npy(label_npy_path)
        self.normalized_label_array = np.array(list(map(normalizing_label, self.label_array)))
        self.to_tensor = transforms.ToTensor()

        self.weak = transforms.Compose([
#             transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=112,
                                  pad_if_needed=True),
            ])
        
        self.strong = transforms.Compose([
#             transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=112,
                                  pad_if_needed=True),
            
            RandAugmentMC(n=2, m=10)
            ])
        
        
        
    def __getitem__(self, index):
        img, label, normalized_label = self.image_array[index], self.label_array[index], self.normalized_label_array[index]
#         print(img.shape)
        img = Image.fromarray(img)
        
        
        weak_img = self.to_tensor(self.weak(img))
        strong_img = self.to_tensor(self.strong(img))
        
        return (weak_img, strong_img), label, normalized_label
        
        
        
        
    def __len__(self):
        return len(self.label_array)

    
    
    