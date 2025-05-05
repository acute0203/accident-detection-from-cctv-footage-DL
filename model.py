import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.4):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        
        if self.downsample:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        return out

class DeeperCCTVClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DeeperCCTVClassifier, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.layer1 = ResidualBlock(64, 128, stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2),
                nn.BatchNorm2d(128)
            ), dropout_rate=0.4)
        
        self.layer2 = ResidualBlock(128, 256, stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2),
                nn.BatchNorm2d(256)
            ), dropout_rate=0.4)
        
        self.layer3 = ResidualBlock(256, 512, stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2),
                nn.BatchNorm2d(512)
            ), dropout_rate=0.5)
        
        self.layer4 = ResidualBlock(512, 512, stride=1, dropout_rate=0.5)
        self.layer5 = ResidualBlock(512, 512, stride=1, dropout_rate=0.5)
        self.layer6 = ResidualBlock(512, 512, stride=1, dropout_rate=0.6)
        self.layer7 = ResidualBlock(512, 512, stride=1, dropout_rate=0.6)
        self.layer8 = ResidualBlock(512, 512, stride=1, dropout_rate=0.6)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout_fc = nn.Dropout(0.6)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        x = self.fc(x)
        return x
