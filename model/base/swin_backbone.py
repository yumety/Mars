import torch
import torch.nn as nn
import timm
from model.base.components import Conv, SPPF

class SwinBackbone(nn.Module):
    def __init__(self, w, r, n):
        super().__init__()
        # 在 create_model 里指定 img_size
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            features_only=True,
            img_size=(640, 640)
        )

        # Swin 输出的四个 stage 通道
        swin_channels = self.swin.feature_info.channels()
        # print(">>> Swin 输出的特征通道：", swin_channels)

        # 1x1 映射 to YOLOv8 期望通道
        self.map_f0 = Conv(swin_channels[0], int(128 * w), k=1, s=1, p=0)
        self.map_f1 = Conv(swin_channels[1], int(256 * w), k=1, s=1, p=0)
        self.map_f2 = Conv(swin_channels[2], int(512 * w), k=1, s=1, p=0)
        self.map_f3 = Conv(swin_channels[3], int(512 * w * r), k=1, s=1, p=0)

        # SPPF 保持原逻辑
        self.sppf = SPPF(int(512 * w * r), int(512 * w * r), k=5)

        # Backbone 最终输出通道列表
        self.out_channels = [
            int(128 * w),
            int(256 * w),
            int(512 * w),
            int(512 * w * r)
        ]

    def forward(self, x):
        # Swin 提取多尺度特征，假设这里 feats[i] 是 NHWC（B, H, W, C）
        feats = self.swin(x) 

        # 把每个特征从 NHWC -> NCHW，再送给 1×1 Conv
        f0 = self.map_f0(feats[0].permute(0, 3, 1, 2))  # (B, 128*w, 160, 160)
        f1 = self.map_f1(feats[1].permute(0, 3, 1, 2))  # (B, 256*w,  80,  80)
        f2 = self.map_f2(feats[2].permute(0, 3, 1, 2))  # (B, 512*w,  40,  40)

        x3 = feats[3].permute(0, 3, 1, 2)               # (B, 768,  20, 20)  或 (B, out_channels_before映射, 20, 20)
        x3 = self.map_f3(x3)                            # (B, 512*w*r, 20, 20)
        f3 = self.sppf(x3)                              # (B, 512*w*r, 20, 20)

        return f0, f1, f2, f3