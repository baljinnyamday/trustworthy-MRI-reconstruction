"""U-Net architecture for MRI reconstruction."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm, ReLU, and optional Dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv(x))


class UNet(nn.Module):
    """Standard U-Net encoder-decoder with skip connections.

    Architecture (default features=[32, 64, 128, 256]):
        Encoder: 1→32→64→128→256 (4 levels, MaxPool2d)
        Bottleneck: 256→512→256
        Decoder: 256→128→64→32 (4 levels, ConvTranspose2d)
        Final: 32→1 (1x1 conv)

    Input/Output: (B, 1, 320, 320)
    Spatial: 320→160→80→40→20 (bottleneck) →40→80→160→320
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for feat in features:
            self.encoders.append(ConvBlock(prev_ch, feat, dropout_rate))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_ch = feat

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, dropout_rate)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feat in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(feat * 2, feat, dropout_rate))

        # Final 1x1 conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        skip_connections: list[torch.Tensor] = []
        for encoder, pool in zip(self.encoders, self.pools, strict=True):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for upconv, decoder, skip in zip(
            self.upconvs, self.decoders, reversed(skip_connections), strict=True
        ):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.final_conv(x)

    def enable_mc_dropout(self) -> None:
        """Enable dropout at inference time for MC Dropout uncertainty."""
        for module in self.modules():
            if isinstance(module, nn.Dropout2d):
                module.train()
