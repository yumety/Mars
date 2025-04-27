import torch.nn as nn


class Backbone(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.imageChannel = 3
        self.kernelSize = 3
        self.stride = 2

        raise NotImplementedError("Backbone::__init__")

    def forward(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape:
            feat0: (B, 128 * w, 160, 160)
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        """

        raise NotImplementedError("Backbone::forward")
