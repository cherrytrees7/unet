import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import resnet50
from nets.vgg import VGG16

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        # 计算卷积核大小
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 计算padding
        padding = kernel_size // 2
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg(x).view([b, 1, c])
        y = self.conv(y)
        y = self.sig(y).view([b, c, 1, 1])
        out = x * y
        return out

class unetUp1(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp1, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

# ECA改进模型
class Unet2(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet2, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]

        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]

        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]


        self.ca4 = ECA(out_filters[3])
        self.ca3 = ECA(out_filters[2])
        self.ca2 = ECA(out_filters[1])
        self.ca1 = ECA(out_filters[0])
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp1(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp1(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp1(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp1(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        feat4 = self.ca4(feat4)
        feat3 = self.ca3(feat3)
        feat2 = self.ca2(feat2)
        feat1 = self.ca1(feat1)


        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
# ==========================Module================================
# 双层卷积
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            # BatchNorm2d：归一化操作
            nn.ReLU(inplace=True),
# CBR操作
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
# 原始unet
class U_Net(nn.Module):
    def __init__(self, img_ch=3, num_classes=2):
        super(U_Net, self).__init__()

# 编码
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

# 解码块
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)

        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)  # mine

        return d1

