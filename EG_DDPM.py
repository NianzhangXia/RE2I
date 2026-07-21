import math
import torch
from torch import nn

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

class DeConvlayer(nn.Module):
  def __init__(self, dim, din_out, kernel_size, stride, padding):
    super().__init__()
    self.Deconv = nn.Sequential(
        nn.ConvTranspose2d(dim, din_out, kernel_size=kernel_size, stride=stride, padding=padding)
    )
  def forward(self, x):
      return self.Deconv(x)

class Convlayer(nn.Module):
  def __init__(self, dim, din_out, kernel_size, stride, padding):
    super().__init__()
    self.Conv = nn.Sequential(
        nn.Conv2d(dim, din_out, kernel_size=kernel_size, stride=stride, padding=padding)
    )
  def forward(self, x):
      return self.Conv(x)

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

class ConvNext_Block(nn.Module):
    def __init__(self, dim, dim_out, ted = None):
        super().__init__()
        self.fc = nn.Sequential(
            nn.GELU(),
            nn.Linear(ted, dim)
        )
        self.first_conv = nn.Conv2d(dim, dim, kernel_size = 7, stride = 1, padding = 3, groups= dim)
        self.conv1 = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1)
        )
        self.conv2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(dim, dim_out, kernel_size = 3, stride = 1, padding = 1)
        )
        self.conv3 = nn.Conv2d(dim, dim_out, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, input, t_eeg = None):
        x = self.first_conv(input)
        time_eeg = self.fc(t_eeg)
        x = x + time_eeg.view(time_eeg.size(0), time_eeg.size(1), 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        output = x + self.conv3(input)

        return output


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

class EG_DDPM(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        EEG_size = 512
    ):
        super().__init__()
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        for times, (dim_in, dim_out) in enumerate(in_out):
            self.down.append(nn.ModuleList([
                ConvNext_Block(dim_in, dim_out, ted = time_dim),
                ConvNext_Block(dim_out, dim_out, ted = time_dim),
                Residual_SelfAttention(dim_out),
                Convlayer(dim_out, dim_out, 4, 2, 1) if times < (len(in_out) - 1) else nn.Identity()
            ]))
        self.FR_block1 = ConvNext_Block(512, 512, ted = time_dim)
        self.FR_block2 = Residual_SelfAttention(512)
        self.FR_block3 = ConvNext_Block(512, 512, ted = time_dim)
        for times, (dim_in, dim_out) in enumerate(reversed(in_out)):
            if times != 3:
                self.up.append(nn.ModuleList([
                    ConvNext_Block(dim_out*2, dim_in, ted = time_dim),
                    DeConvlayer(dim_in, dim_in, 4, 2, 1)
                ]))
            else:
                self.up.append(nn.ModuleList([
                    ConvNext_Block(dim_out, dim_out, ted=time_dim),
                    Convlayer(dim_out, dim_in, 1, 1, 0)
                ]))
        self.eeg_emb = nn.Linear(EEG_size, time_dim)

    def forward(self, input, time, eeg=None):
        copy_data = []
        t = self.time_mlp(time)
        t_eeg = t + self.eeg_emb(eeg)
        for block1, block2, block3, block4 in self.down:
            input = block1(input, t_eeg)
            input = block2(input, t_eeg)
            copy_data.append(input)
            input = block3(input)
            input = block4(input)
        copy_data.pop(0)
        input = self.FR_block1(input, t_eeg)
        input = self.FR_block2(input)
        input = self.FR_block3(input, t_eeg)
        for block1, block2 in self.up:
            input = torch.cat((input, copy_data.pop()), dim = 1) if copy_data else input
            input = block1(input, t_eeg)
            input = block2(input)

        return input



