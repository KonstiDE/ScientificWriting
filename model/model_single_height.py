import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

from model.layers_single_shape import (
    DoubleConv, UpConv, FinalConv
)


class UNET_HEIGHT(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=None):
        super(UNET_HEIGHT, self).__init__()

        if features is None:
            features = [4, 64, 128, 256, 512]

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for i in range(len(features) - 2):
            self.down_convs.append(DoubleConv(features[i], features[i + 1]))

        for i in range(1, len(features) - 1):
            self.skip_convs.append(nn.Conv2d(features[i], features[i], kernel_size=(3, 3), padding=1))

        self.bottleneck = DoubleConv(features[-2], features[-1])

        features = features[::-1]

        for i in range(len(features) - 2):
            self.up_convs.append(DoubleConv(features[i], features[i + 1]))
            self.up_trans.append(UpConv(features[i], features[i + 1]))

        self.final = FinalConv(features[-2])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, predicted_mask):
        masks = [
            predicted_mask,
            tf.resize(predicted_mask, [256, 256], tf.InterpolationMode.BILINEAR),
            tf.resize(predicted_mask, [128, 128], tf.InterpolationMode.BILINEAR)
        ]

        skip_connections = []

        for down_conv in self.down_convs:
            x = down_conv(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for t in range(len(skip_connections)):
            skip_connections[t] = torch.multiply(skip_connections[t], masks[t])
            skip_connections[t] = self.skip_convs[t](skip_connections[t])

        skip_connections = skip_connections[::-1]

        for i in range(len(self.up_convs)):
            x = self.up_trans[i](x)

            x = torch.cat((x, skip_connections[i]), dim=1)
            x = self.up_convs[i](x)

        return self.final(x)


def test():
    unet = UNET_HEIGHT(in_channels=4, out_channels=1)

    x = torch.randn(1, 4, 512, 512)

    out = unet(x, torch.randn([1, 1, 512, 512]))

    print(out.shape)


if __name__ == "__main__":
    test()
