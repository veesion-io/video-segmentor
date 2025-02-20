# ------------------------------
# Temporal UNet Transformer Model
# ------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TemporalUNetTransformer(nn.Module):
    def __init__(
        self,
        out_channels=1,
        num_frames=50,
        image_size=1,
    ):
        super().__init__()

        # Load Swin3D
        self.swin3d = torchvision.models.video.swin3d_t(weights="DEFAULT")

        self.num_frames = num_frames
        self.image_size = image_size

        # Dictionary to store outputs from hooks
        self.feature_maps = {}

        # Register hooks for feature extraction
        self._register_hooks()

        # Decoder with Skip Connections
        decoder_channels = [512, 256, 128, 64]
        encoder_channels = [96, 192, 384, 768]

        self.up1 = nn.ConvTranspose3d(
            encoder_channels[-1],
            decoder_channels[0],
            kernel_size=(3, 5, 5),
            stride=(2, 4, 4),
            padding=(1, 2, 2),
            output_padding=(1, 3, 3),
        )
        self.up2 = nn.ConvTranspose3d(
            decoder_channels[0] + encoder_channels[2],
            decoder_channels[1],
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
            output_padding=(0, 1, 1),
        )
        self.up3 = nn.ConvTranspose3d(
            decoder_channels[1] + encoder_channels[1],
            decoder_channels[2],
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
            output_padding=(0, 1, 1),
        )
        self.up4 = nn.ConvTranspose3d(
            decoder_channels[2] + encoder_channels[0],
            decoder_channels[3],
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
            output_padding=(0, 1, 1),
        )

        self.final_conv = nn.Conv3d(decoder_channels[3], out_channels, kernel_size=1)
        # Initialize weights for decoder layers
        self._init_weights()

    def _init_weights(self):
        """
        Initialize only decoder layers using Kaiming He initialization.
        """
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose3d, nn.Conv3d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _register_hooks(self):
        """
        Register forward hooks to capture feature maps.
        """

        def hook_fn(module, input, output, name):
            # Permute to (B, C, T, H, W) before storing
            self.feature_maps[name] = output.permute(0, 4, 1, 2, 3)

        # Attach hooks to extract feature maps
        self.swin3d.features[0].register_forward_hook(
            lambda mod, inp, out: hook_fn(mod, inp, out, "stage1")
        )
        self.swin3d.features[2].register_forward_hook(
            lambda mod, inp, out: hook_fn(mod, inp, out, "stage2")
        )
        self.swin3d.features[4].register_forward_hook(
            lambda mod, inp, out: hook_fn(mod, inp, out, "stage3")
        )
        self.swin3d.features[6].register_forward_hook(
            lambda mod, inp, out: hook_fn(mod, inp, out, "stage4")
        )

    def forward(self, x):
        self.feature_maps = {}  # Reset stored feature maps
        _ = self.swin3d(x)  # Forward pass through Swin3D (hooks will capture features)

        # Extract permuted feature maps
        x1 = self.feature_maps["stage1"]  # (B, C, T, H, W)
        x2 = self.feature_maps["stage2"]  # (B, C, T, H, W)
        x3 = self.feature_maps["stage3"]  # (B, C, T, H, W)
        x4 = self.feature_maps["stage4"]  # (B, C, T, H, W)

        # Decoder with Skip Connections
        x = self.up1(x4)
        x = x[:, :, :-1]  # go from T = 26 to T = 25
        x = F.leaky_relu(x, 0.2)
        x3 = F.interpolate(x3, size=x.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, x3], dim=1)  # Skip connection

        x = self.up2(x)
        x = F.leaky_relu(x, 0.2)
        x2 = F.interpolate(x2, size=x.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)  # Skip connection

        x = self.up3(x)
        x = F.leaky_relu(x, 0.2)
        x1 = F.interpolate(x1, size=x.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)  # Skip connection

        x = self.up4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.final_conv(x)  # Final output

        return x
