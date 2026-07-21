import torch
import torch.nn as nn
import torch.optim

class Residual_Block1(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Residual_Block1, self).__init__()
    self.down = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=3),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    self.dn = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
    self.relu = nn.ReLU(inplace=True)
  def forward(self, x):
    return self.relu(self.down(x) + self.dn(x))

class Residual_Block2(nn.Module):
    def __init__(self,feature_nums=256):
        super(Residual_Block2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_nums, feature_nums, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(feature_nums),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_nums, feature_nums, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_nums)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.conv(x))

class UpSampleDeConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UpSampleDeConv, self).__init__()
    self.up = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
  def forward(self, x):
    return self.up(x)

class UpSampleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UpSampleConv, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        )
  def forward(self, x):
    return self.conv(x)

class EEG_UNet(nn.Module):
    def __init__(self,
                 dim = 64,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 EEG_size = 64
                 ):
        super().__init__()
        self.channels = channels
        dims = [*map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.first_layer = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        for times, (dim_in, dim_out) in enumerate(in_out):
            self.down.append(nn.ModuleList([
                Residual_Block1(dim_in, dim_out),
                Residual_Block2(dim_out)
            ]))
        for times, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.up.append(nn.ModuleList([
                UpSampleDeConv(dim_in, dim_in) if times==0 else UpSampleDeConv(dim_out, dim_in),
                UpSampleConv(dim_out, dim_in)
            ]))
        self.FR_block = UpSampleConv(512, 256)
        self.last_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
            )
        self.L = nn.Sequential(
            nn.Linear(EEG_size, dim*dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, input):
        input = self.L(input.view(input.size(0), -1)).view(input.size(0), 1, 64, 64)
        input = self.first_layer(input)
        copy_data = []
        copy_data.append(input)
        for block1, block2 in self.down: #Cascaded Residual Block
            input = block1(input)
            input = block2(input)
            copy_data.append(input)
        copy_data.pop()
        input = self.FR_block(input)
        for block1, block2 in self.up:
            input = block1(input)
            input = torch.cat((input, copy_data.pop()), dim=1)
            input = block2(input)
        output = self.last_layer(input)
        return output

