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
        conv1x1(inplanes, planes, stride),
        norm_layer(planes),
    )


class ResnetBlock(nn.Module):

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(ResnetBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CnnEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project
    https://github.com/jcjohnson/fast-neural-style
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
        model = [nn.Conv2d(3, nf, kernel_size=7, padding=3, bias=use_bias),
                 norm_layer(nf),
                 nn.ReLU(inplace=True)]

        if use_dropout:
            model += [nn.Dropout(0.5)]

        model += [nn.MaxPool2d(3)]

        # 2 resnet blocks
        model += [ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer),
                  ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer)]

        # 1 downsampling layer
        model += [ResnetBlock(nf, nf * 2, stride=2, norm_layer=norm_layer,
                              downsample=downsample_(nf, nf * 2, stride=2, norm_layer=norm_layer))]
        nf = nf * 2

        for _ in range(res_block3x3):
            # 3 resnet blocks
            model += [ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer),
                      ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer),
                      ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer), ]

            # 1 downsampling layer
            model += [ResnetBlock(nf, nf * 2, stride=2, norm_layer=norm_layer,
                                  downsample=downsample_(nf, nf * 2, stride=2, norm_layer=norm_layer))]
            nf = nf * 2

        # 2 resnet blocks
        model += [ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer),
                  ResnetBlock(nf, nf, stride=1, norm_layer=norm_layer)]

        model += [nn.AdaptiveAvgPool2d((1, 1))]

        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward"""
        return self.model(x)


class LinearDecoder(nn.Module):
    def __init__(self, o):
        super(LinearDecoder, self).__init__()
