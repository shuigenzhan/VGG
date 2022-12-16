import warnings
import os

import numpy as np
import torch

import torch.nn as nn
import torchvision.transforms


def conv_layer(in_channels, out_channels, kernel_size, padding):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

    return layer


def vgg_conv_block(in_list, out_list, kernel_list, pad_list, pooling_kernel, pooling_stride):
    block = [conv_layer(in_list[i], out_list[i], kernel_list[i], pad_list[i]) for i in range(len(in_list))]
    block += [nn.MaxPool2d(kernel_size=pooling_kernel, stride=pooling_stride)]

    return nn.Sequential(*block)


def vgg_fc_layer(in_feature, out_feature, p=0.5):
    layer = nn.Sequential(
        nn.Linear(in_feature, out_feature),
        nn.BatchNorm1d(out_feature),
        nn.ReLU(),
        nn.Dropout(p=p)
    )

    return layer


class VGG16(nn.Module):
    def __init__(self, num_class=1000, dropout=0.5):
        super().__init__()

        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2,  2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096, dropout)
        self.layer7 = vgg_fc_layer(4096, 4096, dropout)

        self.layer8 = nn.Linear(4096, num_class)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        return x


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    network = VGG16(num_class=1000)
    trans = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224),
                                            torchvision.transforms.ToTensor()])
