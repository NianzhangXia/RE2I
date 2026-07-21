import torch
from torchvision import models


class EVNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size= (4,1), stride=(2,1), padding=(1,0)),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size= (4,1), stride=(2,1), padding=(1,0)),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(256),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.gen = nn.Sequential(
            nn.Linear(8*4, 64)
        )

        self.fc = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        batch_size = x.size(0)
        #x = x.view(batch_size, 1, 64, 32)

        x = self.cnn(x)
        x = x.view(batch_size, -1)
        x = self.gen(x)
        x1 = self.fc(x).view(batch_size,4)

        return  x, x1

class ResBlock(nn.Module):
    def __init__(self,feature_nums=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_nums, feature_nums, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(feature_nums),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_nums, feature_nums, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_nums),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(x + self.conv(x))

        return x



