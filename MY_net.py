from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class My_Block_1(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out_1 = BasicConv2d(in_channels=3, out_channels=128,kernel_size=2)(x)

        out_2 = BasicConv2d(in_channels=3, out_channels=64, kernel_size=2)(x)
        out_2 = x + (1/2)*out_2

        out_3 = BasicConv2d(in_channels=64, out_channels=64, kernel_size=3)(out_2)
        out_3 = x + (1/2)*out_3

        out_4 = BasicConv2d(in_channels=64, out_channels=64, kernel_size=3)(out_3)
        out_4 = x + out_4

        out_1 = self.relu()(out_1)

        out_2 = BasicConv2d(in_channels=64, out_channels=128, kernel_size=3)(out_2)
        out_2 = self.relu()(out_2)

        out_3 = BasicConv2d(in_channels=64, out_channels=128, kernel_size=3)(out_3)
        out_3 = self.relu()(out_3)

        out_4 = BasicConv2d(in_channels=64, out_channels=128, kernel_size=3)(out_4)
        out_4 = self.relu()(out_4)

        x = x + [out_1, 2*out_2, 2*out_3, out_4]
        x = torch.cat(x, 1)
        return x

class My_Block_2(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out_1 = BasicConv2d(in_channels=128, out_channels=256, kernel_size=3)(x)
        out_1 = self.relu()(out_1)

        out_2 = BasicConv2d(in_channels=256, out_channels=256, kernel_size=3)(x)
        out_2 = x + (1/2)*out_2
        out_2 = BasicConv2d(in_channels=256, out_channels=512, kernel_size=3)(out_2)
        out_2 = self.relu()(out_2)

        out_3 = BasicConv2d(in_channels=512, out_channels=512, kernel_size=3)(out_2)
        out_3 = x + (1/2)*out_3
        out_3 = BasicConv2d(in_channels=512, out_channels=1024, kernel_size=3)(out_3)
        out_3 = self.relu()(out_3)

        out_4 = BasicConv2d(in_channels=1024, out_channels=1024, kernel_size=3)(out_3)
        out_4 = x + out_4
        out_4 = BasicConv2d(in_channels=1024, out_channels=1024, kernel_size=3)(out_4)
        out_4 = self.relu()(out_4)

        x = x + [out_1, 2*out_2, 2*out_3, out_4]
        x = torch.cat(x, 1)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.conv_1 = My_Block_1(num_classes=num_classes)
        self.conv_2 = My_Block_2(num_classes=num_classes)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
