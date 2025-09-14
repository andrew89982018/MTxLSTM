import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat,parse_shape
from einops.layers.torch import Rearrange

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class Residual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        #x = self.norm(x)
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class SENet(nn.Module):
    def __init__(self, in_channel, reduction_ratio):
        super(SENet, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=round(in_channel/reduction_ratio))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=round(in_channel/reduction_ratio), out_features=in_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.globalAvgPool(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * x
        return out

def conv1x1(in_channels, out_channels, stride=1):
    ''' 1x1 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1):
    ''' 3x3 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, hid_channels, use_senet=False, ratio=16, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, hid_channels, stride)
        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(hid_channels, hid_channels)
        self.bn2 = nn.BatchNorm2d(hid_channels)
        self.downsample = downsample

        if use_senet:
            self.senet = SENet(hid_channels, ratio)
        else:
            self.senet = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # SENet
        if not self.senet is None:
            out = self.senet(out)

        out += residual
        out = self.relu(out)

        return out

class BottleneckBlock(nn.Module): # bottelneck-block, over the 50 layers.
    expansion = 4
    def __init__(self, in_channels, hid_channels, use_senet=False, ratio=16, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.downsample = downsample
        out_channels = hid_channels * self.expansion
        self.conv1 = conv1x1(in_channels, hid_channels)
        self.bn1 = nn.BatchNorm2d(hid_channels)

        self.conv2 = conv3x3(hid_channels, hid_channels, stride)
        self.bn2 = nn.BatchNorm2d(hid_channels)

        self.conv3 = conv1x1(hid_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if use_senet:
            self.senet = SENet(out_channels, ratio)
        else:
            self.senet = None

    def forward(self, x):
        residual = x # indentity
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.senet is None:
            out = self.senet(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet_Attention(nn.Module):
    '''
    *50-layer
        conv1 (output: 112x112)
            7x7, 64, stride 2
        conv2 (output: 56x56)
            3x3 max pool, stride 2
            [ 1x1, 64  ]
            [ 3x3, 64  ] x 3
            [ 1x1, 256 ]
        cov3 (output: 28x28)
            [ 1x1, 128 ]
            [ 3x3, 128 ] x 4
            [ 1x1, 512 ]
        cov4 (output: 14x14)
            [ 1x1, 256 ]
            [ 3x3, 256 ] x 6
            [ 1x1, 1024]
        cov5 (output: 28x28)
            [ 1x1, 512 ]
            [ 3x3, 512 ] x 3
            [ 1x1, 2048]
        _ (output: 1x1)
            average pool, 100-d fc, softmax
        FLOPs 3.8x10^9
    '''
    '''
    *101-layer
        conv1 (output: 112x112)
            7x7, 64, stride 2
        conv2 (output: 56x56)
            3x3 max pool, stride 2
            [ 1x1, 64  ]
            [ 3x3, 64  ] x 3
            [ 1x1, 256 ]
        cov3 (output: 28x28)
            [ 1x1, 128 ]
            [ 3x3, 128 ] x 4
            [ 1x1, 512 ]
        cov4 (output: 14x14)
            [ 1x1, 256 ]
            [ 3x3, 256 ] x 23
            [ 1x1, 1024]
        cov5 (output: 28x28)
            [ 1x1, 512 ]
            [ 3x3, 512 ] x 3
            [ 1x1, 2048]
        _ (output: 1x1)
            average pool, 100-d fc, softmax
        FLOPs 7.6x10^9
    '''
    def __init__(self, block, layers,patch_size,heads,dim_head,mlp_dim,w,dropout=0., num_classes=1000, use_senet=True, ratio=16, in_ch=3):
        super(ResNet_Attention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.patch_size=patch_size
        self.dim=in_ch * patch_size ** 2
        self.heads=heads
        self.dim_head=dim_head
        self.mlp_dim=mlp_dim
        self.w=w
        self.dropout=dropout

        self.layers = layers
        self.in_channels = 64
        self.use_senet = use_senet
        self.ratio = ratio

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = self.get_layers(block, 64, self.layers[0])
        self.conv3 = self.get_layers(block, 128, self.layers[1], stride=2)
        self.conv4 = self.get_layers(block, 256, self.layers[2], stride=2)
        #self.conv5 = self.get_layers(block, 512, self.layers[3], stride=2)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.state_dict():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self, block, hid_channels, n_layers, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != hid_channels * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.in_channels, hid_channels * block.expansion, stride),
                    nn.BatchNorm2d(hid_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, hid_channels, self.use_senet, self.ratio, stride, downsample))
        self.in_channels = hid_channels * block.expansion

        for _ in range(1, n_layers):
            #layers.append()
            ll=nn.ModuleList([block(self.in_channels, hid_channels, self.use_senet, self.ratio),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = self.patch_size, w2 = self.patch_size),  # block-like attention
                    Residual(self.dim, Attention(dim = self.dim, dim_head = self.dim_head, dropout = self.dropout, window_size = self.w)),
                    Residual(self.dim, FeedForward(dim = self.dim, dropout = self.dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
             ])
            layers.append(nn.Sequential(*ll))
        return nn.Sequential(*layers)
        #return layers

    def forward(self, x):
        '''
            Example tensor shape based on resnet101
        '''

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxPool(x)
        #x = self.conv2(x)
        for stage in self.conv2:
            x = stage(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = self.conv5(x)

        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnetA(**kwargs):
    return ResNet_Attention(BasicBlock, [2, 2, 2],8,16,64,2048,7, **kwargs)