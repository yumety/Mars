import torch
import torch.nn as nn
from model.base.components import Conv, C2f

class Neck(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2

        c2 = int(256 * w)
        c3 = int(512 * w)
        c4 = int(512 * w * r)
        c_concat1 = int(512 * w * (1 + r))  # for feat2 + upsample(feat3)
        c_concat2 = int(256 * w + 512 * w)  # for feat1 + upsample(output1)
        c_concat3 = int(256 * w + 512 * w)  # for downsample(output2) + output1
        c_concat4 = int(512 * w + 512 * w * r)  # for downsample(output3) + feat3

        # Top-down
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f1 = C2f(c_concat1, c3, n=n, shortcut=False)  # TopDown Layer 1

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f2 = C2f(c_concat2, c2, n=n, shortcut=False)  # TopDown Layer 2

        # Bottom-up
        self.down1 = Conv(c2, c2, k=3, s=2)  # stride 2 = downsample
        self.c2f3 = C2f(c_concat3, c3, n=n, shortcut=False)  # BottomUp Layer 1

        self.down2 = Conv(c3, c3, k=3, s=2)
        self.c2f4 = C2f(c_concat4, c4, n=n, shortcut=False)  # BottomUp Layer 2

        # raise NotImplementedError("Neck::__init__")

    def forward(self, feat1, feat2, feat3):
        """
        Input shape:
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        Output shape:
            C: (B, 512 * w, 40, 40)
            X: (B, 256 * w, 80, 80)
            Y: (B, 512 * w, 40, 40)
            Z: (B, 512 * w * r, 20, 20)
        """
        
        x = self.up1(feat3)
        x = torch.cat([x, feat2], dim=1)
        x = self.c2f1(x)
        C = x

        y = self.up2(x)
        y = torch.cat([y, feat1], dim=1)
        X = self.c2f2(y)  # 80x80 output

        y = self.down1(X)
        y = torch.cat([y, x], dim=1)
        Y = self.c2f3(y)  # 40x40 output

        z = self.down2(Y)
        z = torch.cat([z, feat3], dim=1)
        Z = self.c2f4(z)  # 20x20 output

        return C, X, Y, Z
        
        # raise NotImplementedError("Neck::forward")
