import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pickle
import os
import numpy as np
import resnext


import cv2


import cv2


class VideoMaskDataset(Dataset):
    def __init__(self, data_dir, target_fps=10, duration=5, transform=None):
        self.data_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pkl")
        ]
        self.video_path = "/home/veesion/Bag-detector/videos/"
        self.target_fps = target_fps
        self.num_frames = target_fps * duration
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        pkl_file = self.data_files[idx]
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        frames = []
        masks = []

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

        for frame_id in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frame_data = transforms.ToTensor()(frame)
            else:
                frame_data = torch.zeros((3, 256, 256))  # Placeholder if frame missing

            mask_data = [
                cv2.drawContours(
                    np.zeros((256, 256), dtype=np.uint8),
                    data["tracks"][track_id][frame_id][1],
                    -1,
                    255,
                    thickness=cv2.FILLED,
                )
                for track_id in data["tracks"]
                if frame_id in data["tracks"][track_id]
            ]
            if mask_data:
                print(np.stack(mask_data).shape)
            else:
                print((1, 256, 256))

            frames.append(frame_data)
            masks.append(np.stack(mask_data) if mask_data else np.zeros((1, 256, 256)))

        cap.release()

        frames = torch.stack(frames)  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        masks = torch.tensor(np.array(masks), dtype=torch.float32).unsqueeze(
            0
        )  # (1, T, H, W)

        if self.transform:
            frames = self.transform(frames)

        return frames, masks


# ------------------------------
# Temporal UNet Transformer Model
# ------------------------------
class TemporalUNetTransformer(nn.Module):
    def __init__(
        self,
        out_channels=1,
        num_frames=16,
        embed_dim=128,
        num_heads=4,
        depth=2,
    ):
        super().__init__()

        # UNet Encoder
        self.encoder = resnext.resnet50()

        # Transformer for Temporal Modeling
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, batch_first=True
            ),
            num_layers=depth,
        )

        # UNet Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)  # (B, C, T, H, W)

        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(
            B, T, C * H * W
        )  # Flatten for transformer (B, T, D)
        x = self.temporal_transformer(x)
        x = x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # Restore shape

        x = self.decoder(x)  # (B, 1, T, H, W)
        return x


# ------------------------------
# Training Loop
# ------------------------------


def train_model(data_dir, epochs=10, batch_size=4, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoMaskDataset(data_dir, transform=transforms.Normalize(0.5, 0.5))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TemporalUNetTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for frames, masks in dataloader:
            frames, masks = frames.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "temporal_unet_transformer.pth")
    print("Model training complete!")


# ------------------------------
# Run Training
# ------------------------------
if __name__ == "__main__":
    train_model(
        "/home/veesion/Bag-detector/valid_masks_tracks/",
        epochs=20,
        batch_size=4,
        lr=1e-4,
    )
