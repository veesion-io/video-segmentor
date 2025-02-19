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


import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import pickle
import os
import torchvision.transforms as transforms

NUM_CLASSES = 13


class VideoMaskDataset(Dataset):
    def __init__(
        self,
        data_dir,
        target_fps=10,
        duration=5,
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
            (self.num_classes, self.num_frames, 256, 256), dtype=np.uint8
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
                frame_data = transforms.ToTensor()(cv2.resize(frame, (256, 256)))
            else:
                frame_data = torch.zeros((3, 256, 256))  # Placeholder if frame missing

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
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        masks = torch.tensor(masks, dtype=torch.float32)  # (num_classes, T, H, W)

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
        num_heads=4,
        depth=2,
    ):
        super().__init__()

        # UNet Encoder
        self.encoder = resnext.resnet50()
        encoder_output_dim = 2048

        # UNet Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                encoder_output_dim,
                encoder_output_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                encoder_output_dim // 2,
                encoder_output_dim // 4,
                kernel_size=(3, 5, 5),
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                encoder_output_dim // 4,
                encoder_output_dim // 8,
                kernel_size=(3, 5, 5),
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                encoder_output_dim // 8,
                encoder_output_dim // 16,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 0, 0),
            ),
            nn.ReLU(),
            nn.ConvTranspose3d(
                encoder_output_dim // 16,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=(0, 2, 2),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)  # (B, C, T, H, W)

        x = self.decoder(x)  # (B, 1, T, H, W)
        return x


# ------------------------------
# Training Loop
# ------------------------------


def train_model(data_dir, epochs=10, batch_size=4, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoMaskDataset(data_dir, transform=transforms.Normalize(0.5, 0.5))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TemporalUNetTransformer(NUM_CLASSES).to(device)
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
