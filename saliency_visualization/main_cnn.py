import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img

def read_npy(filepath):
    with open(filepath, 'rb') as f:
        data = np.load(f)
    
    return data


def create_model(model_depth, model_width, dropout_rate, num_classes):
    import models.wideresnet_ModifiedToBeSameAsTF as models
    model = models.build_wideresnet(depth=model_depth,
                                    widen_factor=model_width,
                                    dropout=dropout_rate,
                                    num_classes=num_classes)
    return model


cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')
    device = torch.device('cuda')
    device = device
else:
    raise ValueError('Not Using GPU?')

    
test_image_path = '/cluster/tufts/hugheslab/zhuang12/Echo_ClinicalManualScript_1112/ML_DATA/ViewClassifier/npy_seed0/shared_test_this_seed/test_image.npy'
test_label_path = '/cluster/tufts/hugheslab/zhuang12/Echo_ClinicalManualScript_1112/ML_DATA/ViewClassifier/npy_seed0/shared_test_this_seed/test_label.npy'

test_image_array = read_npy(test_image_path)
test_label_array = read_npy(test_label_path)


test_image0 = test_image_array[0]

model_depth = 28
model_width = 2
dropout_rate = 0.0
num_classes = 5

model = create_model(model_depth, model_width, dropout_rate, num_classes)
model.to(device)


checkpoint_path = '/cluster/tufts/hugheslab/zhuang12/Echo_ClinicalManualScript_torch/FixMatch-pytorch_1126/experiments/ViewClassifier/seed0/DEV479/echo/lr0.01_dropout_rate0.2_wd0.005_warmup_img0.0_lambda_u1.0_mu2_T1.0_threshold0.95_PLAX_batch14_PSAX_batch6_A4C_batch6_A2C_batch4_UsefulUnlabeled_batch20_PLAX_PSAX_upweight_factor5.0_class_weights1.01,2.36,2.36,3.55,0.71/model_best.pth.tar'

checkpoint = torch.load(checkpoint_path)

#load the ema weights
model.load_state_dict(checkpoint['ema_state_dict'])

target_layers = [model.block4]

data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])])

img_tensor = data_transform(test_image0)
img_tensor = img_tensor.to(device)

input_tensor = torch.unsqueeze(img_tensor, dim=0)
print('input_tensor.shape: {}'.format(input_tensor.shape))

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

grayscale_cam = cam(input_tensor=input_tensor)

print(grayscale_cam, grayscale_cam.shape, np.sum(grayscale_cam))

