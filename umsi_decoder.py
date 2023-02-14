import torch
import torch.nn as nn


class UMSI(nn.Module):
    def __init__(self):
        super(UMSI, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding='same'),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(1024, 512, 3, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)),
            nn.Dropout2d(p=0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 128, 3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)),
            nn.Dropout2d(p=0.3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)),
            nn.Dropout2d(p=0.3)
        )
        self.output_layer = nn.Conv2d(64, 1, 1, padding='same')
        
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x