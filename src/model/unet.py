import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

""" Code for the components of the U-Net model is taken from: 
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.triple_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double or tiple conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, double=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = (
            DoubleConv(in_channels, out_channels)
            if double
            else TripleConv(in_channels, out_channels)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, ks=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks)

    def forward(self, x):
        return self.conv(x)


""" The UNet network is based on code from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py,
    but includes changes/adaptions."""


class UNet(nn.Module):
    """
    Unet network to compute keypoint detector values, descriptors, and scores.
    """

    def __init__(self, n_channels, n_classes, layer_size):
        """
        Initialize the network with one encoder and two decoders.

        Args:
            num_channels (int): number of channels in the input image (we use 3 for one RGB image).
            num_classes (int): number of classes (output channels from the decoder), 1 in our case.
            layer_size (int): size of the first layer if the encoder. The size of the following layers are
                              determined from this.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bilinear = True
        m = layer_size

        self.inc = DoubleConv(
            n_channels, m
        )  # 384 x 512 (height/width after layer given 384 x 512 input image)
        self.down1 = Down(m, m * 2)  # 192 x 256
        self.down2 = Down(m * 2, m * 4)  #  96 x 128
        self.down3 = Down(m * 4, m * 8)  #  48 x  64
        self.down4 = Down(m * 8, m * 16)  #  24 x  32

        self.up1_pts = Up(m * 24, m * 8, bilinear)
        self.up2_pts = Up(m * 12, m * 4, bilinear)
        self.up3_pts = Up(m * 6, m * 2, bilinear)
        self.up4_pts = Up(m * 3, m, bilinear)
        self.outc_pts = OutConv(m, n_classes)

        self.up1_score = Up(m * 24, m * 8, bilinear)
        self.up2_score = Up(m * 12, m * 4, bilinear)
        self.up3_score = Up(m * 6, m * 2, bilinear)
        self.up4_score = Up(m * 3, m, bilinear)
        self.outc_score = OutConv(m, n_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of network to get keypoint detector values, descriptors and, scores

        Args:
            x (torch.tensor, Bx3xHxW): RGB images to input to the network.

        Returns:
            logit_pts (torch.tensor, Bx1xHxW): detector values for each pixel, which will be used to compute the
                                               final keypoint coordinates.
            score (torch.tensor, Bx1xHxW): an importance score for each pixel.
            descriptors (torch.tensor, BxCxHxW): descriptors for each pixel, C is length of descriptor.

        """
        batch_size, _, height, width = x.size()

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4_up_pts = self.up1_pts(x5, x4)
        x3_up_pts = self.up2_pts(x4_up_pts, x3)
        x2_up_pts = self.up3_pts(x3_up_pts, x2)
        x1_up_pts = self.up4_pts(x2_up_pts, x1)
        logits_pts = self.outc_pts(x1_up_pts)

        x4_up_score = self.up1_score(x5, x4)
        x3_up_score = self.up2_score(x4_up_score, x3)
        x2_up_score = self.up3_score(x3_up_score, x2)
        x1_up_score = self.up4_score(x2_up_score, x1)
        score = self.outc_score(x1_up_score)
        score = self.sigmoid(score)

        # Resize outputs of each encoder layer to the size of the original image. Features are interpolated using
        # bilinear interpolation to get gradients for back-prop. Concatenate along the feature channel to get
        # pixel-wise descriptors of size BxCxHxW.
        f1 = F.interpolate(x1, size=(height, width), mode="bilinear")
        f2 = F.interpolate(x2, size=(height, width), mode="bilinear")
        f3 = F.interpolate(x3, size=(height, width), mode="bilinear")
        f4 = F.interpolate(x4, size=(height, width), mode="bilinear")
        f5 = F.interpolate(x5, size=(height, width), mode="bilinear")

        feature_list = [f1, f2, f3, f4, f5]
        descriptors = torch.cat(feature_list, dim=1)

        return logits_pts, score, descriptors


class UNetVGG16(nn.Module):
    def __init__(self, n_channels, n_classes, pretrained):
        super(UNetVGG16, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.learn_score = True
        self.learn_detector = True
        bilinear = True
        if pretrained:
            model = torchvision.models.vgg16(weights="DEFAULT")
        else:
            model = torchvision.models.vgg16()

        self.layer1 = nn.Sequential(
            *list(model.features.children())[0:4]
        )  # 64  channels, 512 x 384
        self.layer2 = nn.Sequential(
            *list(model.features.children())[4:9]
        )  # 128 channels, 256 x 192
        self.layer3 = nn.Sequential(
            *list(model.features.children())[9:16]
        )  # 256 channels, 128 x 96
        self.layer4 = nn.Sequential(
            *list(model.features.children())[16:23]
        )  # 512 channels, 64  x 48
        self.layer5 = nn.Sequential(
            *list(model.features.children())[23:30]
        )  # 512 channels, 64  x 48

        if self.learn_detector:
            self.up1_pts = Up(512 + 512, 512, bilinear, double=False)
            self.up2_pts = Up(512 + 256, 256, bilinear, double=False)
            self.up3_pts = Up(256 + 128, 128, bilinear, double=True)
            self.up4_pts = Up(128 + 64, 64, bilinear, double=True)
            self.outc_pts = OutConv(64, n_classes)

        if self.learn_score:
            self.up1_score = Up(512 + 512, 512, bilinear, double=False)
            self.up2_score = Up(512 + 256, 256, bilinear, double=False)
            self.up3_score = Up(256 + 128, 128, bilinear, double=True)
            self.up4_score = Up(128 + 64, 64, bilinear, double=True)
            self.outc_score = OutConv(64, n_classes)
            self.sigmoid = nn.Sigmoid()

        print(f"==> using UNETVGG16, pretrain:{pretrained}")

    def forward(self, x, learn_feature=True):
        batch_size, _, height, width = x.size()

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        if self.learn_detector:
            x4_up_pts = self.up1_pts(x5, x4)
            x3_up_pts = self.up2_pts(x4_up_pts, x3)
            x2_up_pts = self.up3_pts(x3_up_pts, x2)
            x1_up_pts = self.up4_pts(x2_up_pts, x1)
            logits_pts = self.outc_pts(x1_up_pts)
        else:
            logits_pts = torch.ones(batch_size, 1, height, width).cuda()

        if self.learn_score:
            x4_up_score = self.up1_score(x5, x4)
            x3_up_score = self.up2_score(x4_up_score, x3)
            x2_up_score = self.up3_score(x3_up_score, x2)
            x1_up_score = self.up4_score(x2_up_score, x1)
            score = self.outc_score(x1_up_score)
            score = self.sigmoid(score)
        else:
            score = torch.ones(batch_size, 1, height, width).cuda()

        # Resize outputs of downsampling layers to the size of the original
        # image. Features are interpolated using bilinear interpolation to
        # get gradients for back-prop. Concatenate along the feature channel
        # to get pixel-wise descriptors of size Bx248xHxW
        if learn_feature:
            f1 = F.interpolate(x1, size=(height, width), mode="bilinear")
            f2 = F.interpolate(x2, size=(height, width), mode="bilinear")
            f3 = F.interpolate(x3, size=(height, width), mode="bilinear")
            f4 = F.interpolate(x4, size=(height, width), mode="bilinear")

            feature_list = [f1, f2, f3, f4]
            features = torch.cat(feature_list, dim=1)
        else:
            features = torch.ones(batch_size, 960, height, width).cuda()

        return logits_pts, score, features
