import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Single convolutional block with optional batch normalization, dropout, skip connection, 
    flexible activation, and order of BN/activation.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int, optional): Size of convolution kernel. Default is 3.
        stride (int, optional): Stride of convolution. Default is 1.
        padding (int, optional): Padding size. Default is 1.
        use_bn (bool, optional): Whether to use BatchNorm2d. Default is True.
        dropout_p (float, optional): Dropout probability. Default is 0.0 (no dropout).
        skip (bool, optional): Enable skip (residual) connection if possible. Default is False.
        activation (str, optional): Activation function name (e.g. 'ReLU', 'LeakyReLU'). Default is 'relu'.
        is_bn_pre_act (bool, optional): If True, BN before activation; otherwise, after activation. Default is True.
    """
    def __init__(
        self,
        in_ch, out_ch,
        kernel_size=3, stride=1, padding=1,
        use_bn=True,
        dropout_p=0.0,
        skip=False,
        activation='relu',
        is_bn_pre_act=True
    ):
        super().__init__()
        self.is_bn_pre_act = is_bn_pre_act
        self.skip = skip and (in_ch == out_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.act  = getattr(nn, activation)() if activation else nn.Identity()
        self.bn   = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.act(self.bn(out)) if self.is_bn_pre_act else self.bn(self.act(out))
        out = self.drop(out)
        if self.skip:
            out = out + x
        return out

class FlexibleCNN(nn.Module):
    """
    Convolutional Neural Network with flexible architecture.

    The model is constructed from a sequence of ConvBlock layers, as specified in `blocks_cfg`, 
    followed by pooling and a fully connected head.

    Example usage:
        blocks_cfg = [
            dict(out_ch=32, use_bn=True, dropout_p=0.1, skip=False, activation='ReLU', is_bn_pre_act=True),
            dict(out_ch=64, use_bn=True, dropout_p=0.1, skip=True, activation='ReLU', is_bn_pre_act=True),
        ]

        model = FlexibleCNN(num_classes=2, input_height=80, input_time=501, blocks_cfg=blocks_cfg)

    Args:
        num_classes (int): Number of output classes.
        input_height (int): Height of input spectrogram/image.
        input_time (int): Width/time dimension of input spectrogram.
        blocks_cfg (list): List of dicts configuring each ConvBlock.

    Attributes:
        encoder (nn.Sequential): Feature extractor (stack of ConvBlocks and MaxPool2d).
        head (nn.Sequential): Classification head (fully connected layers).
    """
    def __init__(self, num_classes, input_height, input_time, blocks_cfg):
        super().__init__()
        layers = []
        in_ch = 1
        for cfg in blocks_cfg:
            layers.append(ConvBlock(in_ch=in_ch, **cfg))
            layers.append(nn.MaxPool2d(2))      # stały pooling → /2
            in_ch = cfg['out_ch']

        self.encoder = nn.Sequential(*layers)

        # wyliczamy wymiar po przejściu przez bloki
        h = input_height // (2 ** len(blocks_cfg))
        t = input_time   // (2 ** len(blocks_cfg))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * h * t, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)
