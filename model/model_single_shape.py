import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

from model.layers_single_shape import (
    DoubleConv, UpConv, FinalConv
)


class UNET_SHAPE(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=None):
        super(UNET_SHAPE, self).__init__()

        if features is None:
            features = [4, 64, 128, 256, 512]

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for i in range(len(features) - 2):
            self.down_convs.append(DoubleConv(features[i], features[i + 1]))

        self.bottleneck = DoubleConv(features[-2], features[-1])

        features = features[::-1]

        for i in range(len(features) - 2):
            self.up_convs.append(DoubleConv(features[i], features[i + 1]))
            self.up_trans.append(UpConv(features[i], features[i + 1]))

        self.final = FinalConv(features[-2])

    def forward(self, x):
        skip_connections = []

        for down_conv in self.down_convs:
            x = down_conv(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for i in range(len(self.up_convs)):
            x = self.up_trans[i](x)

            x = torch.cat((x, skip_connections[i]), dim=1)
            x = self.up_convs[i](x)

        return self.final(x)


def test():
    unet = UNET(in_channels=4, out_channels=1).cuda()

    x = torch.randn(1, 4, 512, 512).cuda()

    out = unet(x)

    print(out.shape)


if __name__ == "__main__":
    test()
