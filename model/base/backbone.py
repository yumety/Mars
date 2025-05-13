import torch.nn as nn
from model.base.components import Conv, C2f, SPPF

class Backbone(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.imageChannel = 3
        self.kernelSize = 3
        self.stride = 2

        # Stem
        self.stem = Conv(3, int(64 * w), k=3, s=2)

        # Stage 1
        self.conv1 = Conv(int(64 * w), int(128 * w), k=3, s=2)
        self.c2f1 = C2f(int(128 * w), int(128 * w), n=n, shortcut=True)

        # Stage 2
        self.conv2 = Conv(int(128 * w), int(256 * w), k=3, s=2)
        self.c2f2 = C2f(int(256 * w), int(256 * w), n=2*n, shortcut=True)

        # Stage 3
        self.conv3 = Conv(int(256 * w), int(512 * w), k=3, s=2)
        self.c2f3 = C2f(int(512 * w), int(512 * w), n=2*n, shortcut=True)

        # Stage 4
        self.conv4 = Conv(int(512 * w), int(512 * w * r), k=3, s=2)
        self.c2f4 = C2f(int(512 * w * r), int(512 * w * r), n=n, shortcut=True)

        # SPPF
        self.sppf = SPPF(int(512 * w * r), int(512 * w * r), k=5)

        # raise NotImplementedError("Backbone::__init__")

    def forward(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape:
            feat0: (B, 128 * w, 160, 160)
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        """
        x = self.stem(x)           # (B, 64*w, 320, 320)
        x = self.conv1(x)
        f0 = self.c2f1(x)          # (B, 128*w, 160, 160)

        x = self.conv2(f0)
        f1 = self.c2f2(x)          # (B, 256*w, 80, 80)

        x = self.conv3(f1)
        f2 = self.c2f3(x)          # (B, 512*w, 40, 40)

        x = self.conv4(f2)
        x = self.c2f4(x)
        f3 = self.sppf(x)          # (B, 512*w*r, 20, 20)

        return f0, f1, f2, f3

        # raise NotImplementedError("Backbone::forward")
