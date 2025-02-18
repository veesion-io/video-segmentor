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


# ------------------------------
# Dataset
# ------------------------------
class VideoMaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pkl")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        with open(self.data_files[idx], "rb") as f:
            data = pickle.load(f)

        frames = []
        masks = []

        for frame_id in sorted(data["tracks"].values())[0]:  # Iterate over frames
            frame_data = [
                data["tracks"][track_id][frame_id][0]
                for track_id in data["tracks"]
                if frame_id in data["tracks"][track_id]
            ]
            mask_data = [
                data["tracks"][track_id][frame_id][1]
                for track_id in data["tracks"]
                if frame_id in data["tracks"][track_id]
            ]

            frames.append(
                np.stack(frame_data) if frame_data else np.zeros((1, 3, 256, 256))
            )
            masks.append(np.stack(mask_data) if mask_data else np.zeros((1, 256, 256)))

        frames = torch.tensor(np.array(frames), dtype=torch.float32).permute(
            1, 0, 2, 3
        )  # (C, T, H, W)
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
