import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

MOMENTUM = 0.9
EPSILON = 1e-5
ACTIVATION = "mish"  # can select ReLU or h-swish, mish
NUM_CLASSES = 1  # classification=4, regression=1

"""references https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py"""


class SE_module(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SE_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Conv2d_fixed_padding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        super(Conv2d_fixed_padding, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size

        if kernel_size > 1:
            self.conv_s1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=[kernel_size // 2, kernel_size // 2],
                stride=stride,
                bias=bias,
            )
        else:
            self.conv_s1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
            )

        self.conv_s2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(self, x):
        if self.stride > 1:
            pad_total = self.kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = F.pad(
                input=x,
                pad=(pad_beg, pad_end, pad_beg, pad_end),
                mode="replicate",
                value=0,
            )
            x = self.conv_s2(x)
        else:
            x = self.conv_s1(x)

        return x


class Norm_activation(nn.Module):
    def __init__(self, num_features, activation="ReLU"):
        super(Norm_activation, self).__init__()

        self.norm = nn.BatchNorm2d(
            num_features=num_features, eps=EPSILON, momentum=MOMENTUM
        )
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "h-swish":
            self.activation = nn.Hardswish()
        elif activation == "mish":
            self.activation = nn.Mish()

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, activation="ReLU"
    ):
        super(ResidualBlock, self).__init__()

        self.stride = stride
        self.avg_pool = nn.MaxPool2d(kernel_size=2, stride=stride)
        #  padding=[kernel_size//2, kernel_size//2])
        self.shortcut_conv = Conv2d_fixed_padding(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            bias=False,
        )
        self.shortcut_norm_act = Norm_activation(
            num_features=out_channels, activation=ACTIVATION
        )

        self.conv1 = Conv2d_fixed_padding(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.conv1_norm_act = Norm_activation(
            num_features=out_channels, activation=ACTIVATION
        )
        self.conv2 = Conv2d_fixed_padding(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=False,
        )
        self.conv2_norm_act = Norm_activation(
            num_features=out_channels, activation=ACTIVATION
        )
        self.se = SE_module(in_channels=out_channels)

        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "h-swish":
            self.activation = nn.Hardswish()
        elif activation == "mish":
            self.activation = nn.Mish()

    def forward(self, x):
        shortcut = x
        if self.stride > 1:
            shortcut = self.avg_pool(shortcut)
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_norm_act(shortcut)

        x = self.conv1(x)
        x = self.conv1_norm_act(x)
        x = self.conv2(x)
        x = self.conv2_norm_act(x)
        x = self.se(x)
        x += shortcut

        # x = self.activation(x)

        return x


class ResNet_18RS(nn.Module):
    def __init__(self, pretrained=False, return_embeddings=False):
        super(ResNet_18RS, self).__init__()
        self.pretrained = pretrained
        self.return_embeddings = return_embeddings

        self.stem_conv0 = Conv2d_fixed_padding(
            in_channels=3, out_channels=32, kernel_size=3, stride=2, bias=False
        )
        self.stem_norm_act0 = Norm_activation(num_features=32, activation=ACTIVATION)
        self.stem_conv1 = Conv2d_fixed_padding(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=False
        )
        self.stem_norm_act1 = Norm_activation(num_features=32, activation=ACTIVATION)
        self.stem_conv2 = Conv2d_fixed_padding(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, bias=False
        )
        self.stem_norm_act2 = Norm_activation(num_features=64, activation=ACTIVATION)

        self.resblock1_1_1 = ResidualBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            activation=ACTIVATION,
        )
        self.resblock1_1_2 = ResidualBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            activation=ACTIVATION,
        )
        # self.resblock1_2_1 = ResidualBlock(
        #     in_channels=64,
        #     out_channels=64,
        #     kernel_size=3,
        #     stride=1,
        #     activation=ACTIVATION,
        # )
        # self.resblock1_2_2 = ResidualBlock(
        #     in_channels=64,
        #     out_channels=64,
        #     kernel_size=3,
        #     stride=1,
        #     activation=ACTIVATION,
        # )

        self.resblock2_1_1 = ResidualBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            activation=ACTIVATION,
        )
        self.resblock2_1_2 = ResidualBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            activation=ACTIVATION,
        )
        # self.resblock2_2_1 = ResidualBlock(
        #     in_channels=128,
        #     out_channels=128,
        #     kernel_size=3,
        #     stride=1,
        #     activation=ACTIVATION,
        # )
        # self.resblock2_2_2 = ResidualBlock(
        #     in_channels=128,
        #     out_channels=128,
        #     kernel_size=3,
        #     stride=1,
        #     activation=ACTIVATION,
        # )

        self.resblock3_1_1 = ResidualBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=2,
            activation=ACTIVATION,
        )
        self.resblock3_1_2 = ResidualBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            activation=ACTIVATION,
        )
        # self.resblock3_2_1 = ResidualBlock(
        #     in_channels=256,
        #     out_channels=256,
        #     kernel_size=3,
        #     stride=1,
        #     activation=ACTIVATION,
        # )
        # self.resblock3_2_2 = ResidualBlock(
        #     in_channels=256,
        #     out_channels=256,
        #     kernel_size=3,
        #     stride=1,
        #     activation=ACTIVATION,
        # )

        self.resblock4_1_1 = ResidualBlock(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=2,
            activation=ACTIVATION,
        )
        self.resblock4_1_2 = ResidualBlock(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            activation=ACTIVATION,
        )
        # self.resblock4_2_1 = ResidualBlock(
        #     in_channels=512,
        #     out_channels=512,
        #     kernel_size=3,
        #     stride=1,
        #     activation=ACTIVATION,
        # )
        # self.resblock4_2_2 = ResidualBlock(
        #     in_channels=512,
        #     out_channels=512,
        #     kernel_size=3,
        #     stride=1,
        #     activation=ACTIVATION,
        # )

        self.pool_head = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(512, 1000, 1, 1, 0, bias=True)
        if ACTIVATION == "ReLU":
            self.activation_head = nn.ReLU()
        elif ACTIVATION == "h-swish":
            self.activation_head = nn.Hardswish()
        elif ACTIVATION == "mish":
            self.activation_head = nn.Mish()
        self.flatten = nn.Flatten(1, -1)
        self.classifier = nn.Linear(1000, NUM_CLASSES)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Stem (ResNet(7*7) -> ResNet_18RS(3*3)*3)
        x = self.stem_conv0(x)
        x = self.stem_norm_act0(x)
        x = self.stem_conv1(x)
        x = self.stem_norm_act1(x)
        x = self.stem_conv2(x)
        x = self.stem_norm_act2(x)

        x = self.resblock1_1_1(x)
        x = self.resblock1_1_2(x)
        # x = self.resblock1_2_1(x)
        # x = self.resblock1_2_2(x)

        x = self.resblock2_1_1(x)
        x = self.resblock2_1_2(x)
        # x = self.resblock2_2_1(x)
        # x = self.resblock2_2_2(x)

        x = self.resblock3_1_1(x)
        x = self.resblock3_1_2(x)
        # x = self.resblock3_2_1(x)
        # x = self.resblock3_2_2(x)

        x = self.resblock4_1_1(x)
        x = self.resblock4_1_2(x)
        # x = self.resblock4_2_1(x)
        # x = self.resblock4_2_2(x)

        x = self.pool_head(x)
        x = self.conv_head(x)
        x = self.activation_head(x)
        embedding = self.flatten(x)
        x = self.classifier(embedding)
        # x = self.softmax(x)
        if self.return_embeddings:
            return x, embedding

        return x


def resnetrs_init_weights(net, zero_init_last_bn=True):
    for n, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    if zero_init_last_bn:
        for m in net.modules():
            if hasattr(m, "zero_init_last_bn"):
                m.zero_init_last_bn()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    resnetrs = ResNet_18RS().to(device)
    # resnetrs.eval()
    print(resnetrs)
    summary(resnetrs, (3, 224, 224))
    resnetrs_init_weights(resnetrs)
    # input = np.ones([1,3,224,224])
    # print(resnetrs(torch.from_numpy(input.astype(np.float32)).to(device)))
