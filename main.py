import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pickle
import os
import numpy as np


import cv2


import cv2


import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import pickle
import os
import torchvision.transforms as transforms

NUM_CLASSES = 13
TARGET_FPS = 5
VIDEO_DURATION = 5
IMAGE_SIZE = 224
NUM_FRAMES = int(VIDEO_DURATION * TARGET_FPS)
BATCH_SIZE = 2


class VideoMaskDataset(Dataset):
    def __init__(
        self,
        data_dir,
        target_fps=TARGET_FPS,
        duration=VIDEO_DURATION,
        num_classes=NUM_CLASSES,
        transform=None,
    ):
        self.data_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pkl")
        ]
        self.video_path = "/home/veesion/Bag-detector/videos/"
        self.target_fps = target_fps
        self.num_frames = target_fps * duration
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        pkl_file = self.data_files[idx]
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        frames = []
        masks = np.zeros(
            (self.num_classes, self.num_frames, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8
        )  # (C, T, H, W)

        video_name = os.path.basename(pkl_file).replace(".pkl", ".mp4")
        cap = cv2.VideoCapture(os.path.join(self.video_path, video_name))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        if not video_fps or video_fps <= 0:
            cap.release()
            raise ValueError(f"Invalid FPS detected in video: {video_name}")

        # Compute the exact frame IDs to fetch
        frame_interval = max(1, int(round(video_fps / self.target_fps)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ids = [
            i * frame_interval
            for i in range(self.num_frames)
            if i * frame_interval < total_frames
        ]

        for t, frame_id in enumerate(frame_ids):  # t is the index in our time dimension
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frame_data = transforms.ToTensor()(
                    cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                )
            else:
                frame_data = torch.zeros(
                    (3, IMAGE_SIZE, IMAGE_SIZE)
                )  # Placeholder if frame missing

            frames.append(frame_data)

            # Process masks for each track_id
            for track_id in data["tracks"]:
                if frame_id in data["tracks"][track_id]:
                    contours = data["tracks"][track_id][frame_id][1]  # Extract contours
                    class_id = data["classes"][track_id]  # Get class for this track_id
                    if (
                        0 <= class_id < self.num_classes
                    ):  # Ensure class_id is within range
                        cv2.drawContours(
                            masks[class_id, t], contours, -1, 255, thickness=cv2.FILLED
                        )

        cap.release()

        frames = torch.stack(frames)  # (T, C, H, W)
        if self.transform:
            frames = self.transform(frames)
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        masks = (
            torch.tensor(masks, dtype=torch.float32) / 255.0
        )  # (num_classes, T, H, W)

        return frames, masks


import torchvision


# ------------------------------
# Temporal UNet Transformer Model
# ------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class TemporalUNetTransformer(nn.Module):
    def __init__(
        self,
        out_channels=1,
        num_frames=50,
        image_size=IMAGE_SIZE,
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


# ------------------------------
# Training Loop
# ------------------------------


def train_model(data_dir, epochs=20, batch_size=BATCH_SIZE, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoMaskDataset(
        data_dir,
        transform=transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),  # Rescale to [0,1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    model = TemporalUNetTransformer(NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    # model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor(
        [100.0], device=device
    )  # Increase weight for positive pixels
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_pixels = 0
        total_pixels = 0
        baseline_correct_pixels = 0  # When prediction is always zero

        for frames, masks in dataloader:
            with torch.amp.autocast("cuda", dtype=torch.float32):
                frames, masks = frames.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(frames)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            print(outputs[0, :, 13, 112, 112], masks[0, :, 13, 112, 112])
            # Compute pixel-wise accuracy
            predicted = (outputs > 0).float()  # Convert logits to binary predictions
            correct_pixels += (predicted == masks).sum().item()
            total_pixels += masks.numel()

            # Baseline accuracy (assume all predictions are 0)
            baseline_correct_pixels += (masks == 0).sum().item()

        pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
        baseline_accuracy = (
            baseline_correct_pixels / total_pixels if total_pixels > 0 else 0
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}, "
            f"Pixel Accuracy: {pixel_accuracy:.4f}, Baseline Accuracy: {baseline_accuracy:.4f}"
        )

    torch.save(model.state_dict(), "temporal_unet_transformer.pth")
    print("Model training complete!")


# ------------------------------
# Run Training
# ------------------------------
if __name__ == "__main__":
    train_model(
        "/home/veesion/Bag-detector/valid_masks_tracks/",
    )
