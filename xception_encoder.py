import torch
import torch.nn as nn


class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.entry_flow = EntryFlow()
        self.middle_flow = MiddleFlow() 
        self.exit_flow = ExitFlow(num_classes)
        
    
    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)    
        return x


class EntryFlow(nn.Module):
    def __init__(self):
        super(EntryFlow, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128)
        )
        self.block2 = nn.Sequential(
            DepthwiseConv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            DepthwiseConv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2),
            nn.BatchNorm2d(256)
        )
        self.block3 = nn.Sequential(
            nn.ReLU(),
            DepthwiseConv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            DepthwiseConv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(256, 728, 1, stride=2),
            nn.BatchNorm2d(728)
        )
        self.block4 = nn.Sequential(
            nn.ReLU(),
            DepthwiseConv2d(256, 728, 3),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            DepthwiseConv2d(728, 728, 3),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
            
        
    def forward(self, x):
        x = self.block1(x)
        x_res = self.res1(x)
        x = self.block2(x)
        x = torch.add(x_res, x)
        x_res = self.res2(x)
        x = self.block3(x)
        x = torch.add(x_res, x)
        x_res = self.res3(x)
        x = self.block4(x)
        x = torch.add(x_res, x)
        return x
    

class MiddleFlow(nn.Module):
    def __init__(self):
        super(MiddleFlow, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(8):
            self.layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    DepthwiseConv2d(728, 728, 3),
                    nn.BatchNorm2d(728),
                    nn.ReLU(),
                    DepthwiseConv2d(728, 728, 3),
                    nn.BatchNorm2d(728),    
                    nn.ReLU(),
                    DepthwiseConv2d(728, 728, 3),
                    nn.BatchNorm2d(728)
                )
            )
            
        
    def forward(self, x):
        for layer in self.layers:
            x_res = x
            x = layer(x)
            x = torch.add(x_res, x)
        return x


class ExitFlow(nn.Module):
    def __init__(self, num_classes):
        super(ExitFlow, self).__init__()
        self.res1 = nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=2),
            nn.BatchNorm2d(1024)
        )
        self.block1 = nn.Sequential(
            nn.ReLU(),
            DepthwiseConv2d(728, 728, 3),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            DepthwiseConv2d(728, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            DepthwiseConv2d(1024, 1536, 3),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            DepthwiseConv2d(1536, 2048, 3),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
        self.softmax = nn.Softmax()
            
        
    def forward(self, x):
        x_res = self.res1(x)
        x = self.block1(x)
        x = torch.add(x_res, x)
        x = self.block2(x)
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         x = self.softmax(x)
        return x
        
        
class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same')
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x        
