import torch
import torch.nn as nn
from nets.vgg import VGG16
import torch.nn.functional as F


# 坐标注意力机制
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
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

class unetUp_RAC(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp_RAC, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

        self.conv_res = nn.Sequential(
            nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        self.DConv2 = nn.Sequential(
            nn.Conv2d(in_size, in_size // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_size // 2, in_size // 2, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(in_size // 2),
            nn.ReLU(),
            nn.Conv2d(in_size // 2, in_size, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.DConv4 = nn.Sequential(
            nn.Conv2d(in_size, in_size // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_size // 2, in_size // 2, kernel_size=3, padding=4, stride=1, dilation=4, bias=True),
            nn.BatchNorm2d(in_size // 2),
            nn.ReLU(),
            nn.Conv2d(in_size // 2, in_size, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.DConv8 = nn.Sequential(
            nn.Conv2d(in_size, in_size // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_size // 2, in_size // 2, kernel_size=3, padding=8, stride=1, dilation=8, bias=True),
            nn.BatchNorm2d(in_size // 2),
            nn.ReLU(),
            nn.Conv2d(in_size // 2, in_size, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.voteConv = nn.Sequential(
            nn.Conv2d(in_size * 3, out_size, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_size),
        )
        self.CA = CoordAtt(out_size, out_size)

    def forward(self, inputs1, inputs2):
        x = torch.cat([inputs1, self.up(inputs2)], 1)
        x_ = self.conv_res(x)

        x1 = self.DConv2(x)
        x1 = x1 + x_

        x2 = self.DConv4(x)
        x2 = x2 + x_

        x3 = self.DConv8(x)
        x3 = x3 + x_

        x_out = torch.cat((x1, x2, x3), dim=1)
        x_out = self.voteConv(x_out)
        x_out = self.CA(x_out)

        return x_out

class unetUp_RAC3(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(unetUp_RAC3, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512 这一层使用的Res_ASPP+CA(坐标注意力)
        self.up_concat4 = unetUp_RAC(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp_RAC(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp_RAC(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

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


    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
