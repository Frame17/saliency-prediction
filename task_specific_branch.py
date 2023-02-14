import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from fcn import FCN


class TaskSpecificBranch(nn.Module):
    def __init__(self):
        super(TaskSpecificBranch, self).__init__()
        self.segmentation = FCN(5)
        self.task_specific_subnet = TaskSpecificSubnet()
        
    def forward(self, x, task_label):
        x = self.segmentation(x)
        x = self.task_specific_subnet(x, task_label)
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