import torch.nn as nn
import torch.nn.functional as F

# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.AvgPool2d(2)
    )


class ConvNet(nn.Module):

    def __init__(self, x_dim=3, num_class=10):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, 16),
            conv_block(16, 64),
            conv_block(64, 64),
            # conv_block(256, 512),
        )
        self.fc = nn.Linear(64, num_class)
        self.feature_dim = 64  # 设置特征维度
    def forward(self, x):
        x = self.encoder(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_features(self, x):
        x = self.encoder(x)
        #x = F.adaptive_avg_pool2d(x, 1)  # 使用自适应平均池化
        #x = x.view(x.size(0), -1)  # 展平特征
        return x