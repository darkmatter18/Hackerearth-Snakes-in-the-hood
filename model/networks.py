import torch
from torch import nn, Tensor
from typing import Callable, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def downsample_(inplanes: int, planes: int, stride: int, norm_layer: Callable[..., nn.Module]):
    return nn.Sequential(
        conv1x1(inplanes, planes * 4, stride),
        norm_layer(planes * 4),
    )


class ResnetBlock(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, groups: int = 1, base_width: int = 64,
                 dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None,
                 use_dropout: bool = False, drop: float = 0.0) -> None:

        super(ResnetBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.use_dropout = use_dropout

        if self.use_dropout:
            self.dropout = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.use_dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.use_dropout:
            out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.use_dropout:
            out = self.dropout(out)

        return out


class CnnEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    """

    def __init__(self, nf=64, res_block3x3: int = 2, norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
                 use_dropout: bool = False):
        """Construct a Resnet-based generator

        :param nf: the number of filters in the first conv layer
        :param norm_layer: normalization layer
        :param use_dropout: if use dropout layers
        """
        super(CnnEncoder, self).__init__()

        use_bias = False
        # First Conv layer
        model = [nn.Conv2d(3, nf, kernel_size=5, padding=2, bias=use_bias),
                 norm_layer(nf),
                 nn.ReLU(inplace=True)]

        if use_dropout:
            model += [nn.Dropout(0.1)]

        model += [nn.MaxPool2d(2)]

        # 2 resnet blocks
        model += [ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer, use_dropout=use_dropout, drop=0.15),
                  ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer, use_dropout=use_dropout, drop=0.15)]

        # 1 downsampling layer
        model += [ResnetBlock(nf, nf * 2, stride=2, norm_layer=norm_layer, use_dropout=use_dropout, drop=0.15,
                              downsample=downsample_(nf, nf * 2, stride=2, norm_layer=norm_layer))]
        nf = nf * 2

        for _ in range(res_block3x3):
            # 3 resnet blocks
            model += [ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer, use_dropout=use_dropout, drop=0.2),
                      ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer, use_dropout=use_dropout, drop=0.2),
                      ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer, use_dropout=use_dropout, drop=0.2)]

            # 1 downsampling layer
            model += [ResnetBlock(nf, nf * 2, stride=2, norm_layer=norm_layer, use_dropout=use_dropout, drop=0.2,
                                  downsample=downsample_(nf, nf * 2, stride=2, norm_layer=norm_layer))]
            nf = nf * 2

        # 2 resnet blocks
        model += [ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer, use_dropout=use_dropout, drop=0.25),
                  ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer, use_dropout=use_dropout, drop=0.25)]

        model += [nn.AdaptiveAvgPool2d((1, 1))]

        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward"""
        return self.model(x)


class BasicCnnEncoder(nn.Module):
    def __init__(self, nf):
        super(BasicCnnEncoder, self).__init__()

        # Input: (x, 3, 128, 128)
        # Output: (x, 64, 64, 64)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2)
        )

        # Input: (x, 64, 64, 64)
        # Output: (x, 128, 32, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2)
        )

        # Input: (x, 128, 32, 32)
        # Output: (x, 256, 16, 16)
        self.block3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2)
        )

        # Input: (x, 256, 16, 16)
        # Output: (x, 512, 8, 8)
        self.block4 = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2)
        )

        self.argPool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.argPool(x)

        return x


class LinearDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LinearDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
