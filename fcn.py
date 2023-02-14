import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, out_channels):
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
        self.score_fr = nn.Conv2d(4096, out_channels, 1)
        self.score_pool4 = nn.Conv2d(512, out_channels, 1)

        self.upscore2 = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(out_channels, out_channels, 32, stride=16, bias=False)        

    
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
        upscore2 = x

        x = self.score_pool4(l4_output)
        x = x[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = x

        x = upscore2 + score_pool4c
        x = self.upscore16(x)
        x = x[:, :, 24:24 + input.size()[2], 24:24 + input.size()[3]].contiguous()
        return x
    
    
     
class Dataset(Dataset):
    def __init__(self, source_dir, target_dir, transformations):
        self.source_images = [os.path.join(source_dir, x) for x in sorted(os.listdir(source_dir))]
        self.target_images = [os.path.join(target_dir, x) for x in sorted(os.listdir(target_dir))]
        self.transformations = transformations
        
    
    def __len__(self):
        return len(self.target_images)
    
    
    def __getitem__(self, idx):
        source_img = Image.open(self.source_images[idx]).convert('RGB')
        target_img = Image.open(self.target_images[idx])
        
        source_img = self.transformations(source_img)
        target_img = self.transformations(target_img)
        return (source_img, target_img)