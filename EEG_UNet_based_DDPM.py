import torch
import torch.nn as nn
import torch.optim
import math

class Residual_Block1(nn.Module):
  def __init__(self, in_channels, out_channels, ted = None):
    super().__init__()
    self.fc = nn.Sequential(
        nn.GELU(),
        nn.Linear(ted, out_channels)
    )
    self.down = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=3),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
    )
    self.dn = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
  def forward(self, x, t_eeg = None):
      time_eeg = self.fc(t_eeg)
      x = self.down(x) + self.dn(x)
      x = x + time_eeg.view(time_eeg.size(0), time_eeg.size(1), 1, 1)
      return  x

class Residual_Block2(nn.Module):
    def __init__(self,feature_nums=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_nums, feature_nums, kernel_size = 3, stride = 1, padding = 1),
            nn.Conv2d(feature_nums, feature_nums, kernel_size=3, stride=1, padding=1),
        )
    def forward(self, x):
        return x + self.conv(x)

class UpSampleDeConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.up = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
    )
  def forward(self, x):
      return self.up(x)

class UpSampleConv(nn.Module):
  def __init__(self, in_channels, out_channels, ted = None):
    super().__init__()
    self.fc = nn.Sequential(
        nn.GELU(),
        nn.Linear(ted, out_channels)
    )
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
  def forward(self, x, t_eeg = None):
      time_eeg = self.fc(t_eeg)
      x = self.conv(x) + time_eeg.view(time_eeg.size(0), time_eeg.size(1), 1, 1)
      return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        out = (x - mean) /torch.sqrt(var + self.eps) * self.weight + self.bias
        return out

class Residual_SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=dim, out_channels=dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=dim, out_channels=dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.norm = LayerNorm(dim)

    def forward(self, x):
        B, C, W, H = x.size()
        x = self.norm(x)
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W * H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, W * H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x

        return out

class EEG_UNet_based_DDPM(nn.Module):
    def __init__(self,
                 dim = 64,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 EEG_size = 512,
                 ):
        super().__init__()
        self.channels = channels
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=2, stride=2)
        )
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        for times, (dim_in, dim_out) in enumerate(in_out[1:]):
            self.down.append(nn.ModuleList([
                Residual_Block1(dim_in, dim_out, ted = time_dim),
                Residual_Block2(dim_out),
                Residual_SelfAttention(dim_out)
            ]))
        self.FR_block = UpSampleConv(512, 256, ted=time_dim)
        self.FR_block2 = Residual_SelfAttention(256)
        self.FR_block3 = UpSampleConv(256, 256, ted=time_dim)
        for times, (dim_in, dim_out) in enumerate(reversed(in_out)):
            if times != 3:
                self.up.append(nn.ModuleList([
                    UpSampleDeConv(dim_in, dim_in) if times==0 else UpSampleDeConv(dim_out, dim_in),
                    UpSampleConv(dim_out, dim_in, ted = time_dim)
                ]))
            else:
                self.up.append(nn.ModuleList([
                    UpSampleDeConv(dim_out, dim_out),
                    UpSampleConv(dim_out, dim_in, ted = time_dim)
                ]))
        self.eeg_emb = nn.Linear(EEG_size, time_dim)

    def forward(self, input, time, eeg):
        copy_data = []
        t = self.time_mlp(time)
        t_eeg = t + self.eeg_emb(eeg)
        input = self.first_layer(input)
        copy_data.append(input)
        for block1, block2, ra_block in self.down: #Cascaded Residual Block
            input = block1(input, t_eeg)
            copy_data.append(input)
            input = block2(input)
            input = ra_block(input)
        copy_data.pop()
        input = self.FR_block(input, t_eeg)
        input = self.FR_block2(input)
        input = self.FR_block3(input, t_eeg)
        for block1, block2 in self.up:
            input = block1(input)
            input = torch.cat((input, copy_data.pop()), dim=1) if copy_data else input
            input = block2(input, t_eeg)

        return input

