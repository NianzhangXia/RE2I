import torch
import torch.nn as nn

class EVRNet(nn.Module):
    def __init__(self, num_class:int =40):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            MKRB(64),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size= (1,3), stride=(1,2), padding=(0,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size= (1,3), stride=(1,2), padding=(0,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            MKRB(256),
        )

        self.spatial_temporal = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(512),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(512, num_class),
        )

    def forward(self, eeg):
        batch_size = eeg.size(0)
        eeg = eeg.view(batch_size, 1, eeg.size(1), eeg.size(2))
        eeg = self.temporal(eeg)
        eeg = self.spatial(eeg)
        e = self.spatial_temporal(eeg).view(batch_size, -1)

        pred = self.fc(e)

        return  e, pred


class MKRB(nn.Module):
    def __init__(self,feature_nums=256):
        super(MKRB, self).__init__()
        self.conv = Conv2D(feature_nums=feature_nums, kernel=3)
        self.conv1 = Conv2D(feature_nums=feature_nums, kernel=5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v):
        y0 = self.relu(v+self.conv(v))
        y = self.relu(v + self.conv1(y0))

        return y





