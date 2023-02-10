import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image


class TaskSpecificBranch(nn.Module):
    def __init__(self):
        super(TaskSpecificBranch, self).__init__()
        self.segmentation = FCN()
        self.task_specific_subnet = TaskSpecificSubnet()
        
    def forward(self, x, task_label):
        x = self.segmentation(x)
        x = self.task_specific_subnet(x, task_label)
        return x  
    

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=100),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer_6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d()
        )
        
        # output
        self.score_fr = nn.Conv2d(4096, 5, 1)
        self.score_pool4 = nn.Conv2d(512, 5, 1)

        self.upscore2 = nn.ConvTranspose2d(5, 5, 4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(5, 5, 32, stride=16, bias=False)        

    
    def forward(self, x):
        input = x
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        l4_output = x
        x = self.layer_5(x)
        x = self.layer_6(x)
        
        x = self.score_fr(x)
        x = self.upscore2(x)
        upscore2 = x  # 1/16

        x = self.score_pool4(l4_output)
        x = x[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = x  # 1/16

        x = upscore2 + score_pool4c
        x = self.upscore16(x)
        x = x[:, :, 24:24 + input.size()[2], 24:24 + input.size()[3]].contiguous()
        return x

    
class TaskSpecificSubnet(nn.Module):
    def __init__(self):
        super(TaskSpecificSubnet, self).__init__()
        self.segmentation_encoder = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(3, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 784)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.task_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        
    def forward(self, x, task_label):
        x = self.segmentation_encoder(x)
        x = nn.functional.interpolate(x, size=torch.Size([28, 28]))
        task_encoding = self.task_encoder(task_label)
        
        task_encoding = task_encoding.view(-1, 1, 28, 28)
        task_encoding = torch.cat([task_encoding for _ in range(64)], dim=1)
        x = torch.cat((x, task_encoding), dim=1)
        
        x = self.conv(x)
        x = self.task_decoder(x)
        return x
    
    
LABELS = {'A': 0, 'B': 1, 'C': 2}

class Dataset(Dataset):
    def __init__(self, source_dir, target_dir, transformations):
        self.source_images = np.repeat([os.path.join(source_dir, x) for x in sorted(os.listdir(source_dir))], 3)
        self.target_images = [os.path.join(target_dir, x) for x in sorted(os.listdir(target_dir))]
        self.transformations = transformations
        
    
    def __len__(self):
        return len(self.target_images)
    
    
    def __getitem__(self, idx):
        source_img = Image.open(self.source_images[idx]).convert('RGB')
        target_img = Image.open(self.target_images[idx])
        task_label = target_img.filename.split('_')[-1].split('.')[0]
        task_label = nn.functional.one_hot(torch.tensor(LABELS[task_label]), 3).to(torch.float32)
        
        source_img = self.transformations(source_img)
        target_img = self.transformations(target_img)
        return (source_img, target_img, task_label)
    
    
base_path = '/Users/serhii/Desktop/mpl-ws2223_group7/dataset_taskvis'


def load_dataset():
    transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = Dataset(base_path + '/source', base_path + '/gt/heatmaps_accum/10000', transformations)
    return dataset