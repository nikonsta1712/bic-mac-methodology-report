import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """
    Standard 3D convolutional block with a residual (shortcut) connection 
    to improve gradient flow in deep networks.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch)

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += res  # Residual addition
        return self.relu(out)


class AttentionGate(nn.Module):
    """
    Attention Gate to help the decoder focus on relevant features from
    the encoder skip connections.
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal from the lower level (upsampled)
        # x: spatial feature map from the encoder skip connection
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, metadata_dim=5):
        super().__init__()

        # Encoder
        self.enc1 = ResidualConvBlock(in_channels, 16)
        self.enc2 = ResidualConvBlock(16, 32)
        self.enc3 = ResidualConvBlock(32, 64)
        self.enc4 = ResidualConvBlock(64, 128)

        self.pool = nn.MaxPool3d(2)

        # Bottleneck + Metadata context
        self.bottleneck = ResidualConvBlock(128, 256)
        self.meta_fc = nn.Sequential(
            nn.Linear(metadata_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout3d(0.25)

        # Attention Gates
        self.att4 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att3 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.att2 = AttentionGate(F_g=32, F_l=32, F_int=16)
        self.att1 = AttentionGate(F_g=16, F_l=16, F_int=8)

        # Decoder
        self.up4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec4 = ResidualConvBlock(256, 128)

        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ResidualConvBlock(128, 64)

        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ResidualConvBlock(64, 32)

        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ResidualConvBlock(32, 16)

        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x, metadata=None):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bottleneck
        b = self.bottleneck(self.pool(s4))

        # Inject Metadata
        if metadata is not None:
            # metadata shape: [Batch, metadata_dim]
            meta_weights = self.meta_fc(metadata).view(-1, 256, 1, 1, 1)
            b = b * meta_weights  # Feature modulation based on subject metadata

        b = self.dropout(b)

        # Decoder with Attention Gates
        d4 = self.up4(b)
        s4 = self.att4(g=d4, x=s4)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        d3 = self.up3(d4)
        s3 = self.att3(g=d3, x=s3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self.up2(d3)
        s2 = self.att2(g=d2, x=s2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up1(d2)
        s1 = self.att1(g=d1, x=s1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.out_conv(d1)


def build_model():
    """
    Builds the model specifically for the BIC-MAC challenge.
    Input: 4 channels (NAC-PET, MRI-In, MRI-Out, Topogram).
    Metadata: 5 dimensions (Age, Sex, Height, Weight, BMI).
    """
    return UNet3D(in_channels=4, out_channels=1, metadata_dim=5)


if __name__ == "__main__":
    model = build_model()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")