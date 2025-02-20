import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pickle
import os
import numpy as np
import cv2
from model import TemporalUNetTransformer  # Ensure your model class is in model.py


NUM_CLASSES = 13
TARGET_FPS = 3
VIDEO_DURATION = 8.333334
IMAGE_SIZE = 224
NUM_FRAMES = int(VIDEO_DURATION * TARGET_FPS)
BATCH_SIZE = 4


class VideoMaskDataset(Dataset):
    def __init__(
        self,
        data_dir,
        target_fps=TARGET_FPS,
        duration=VIDEO_DURATION,
        num_classes=NUM_CLASSES,
        transform=None,
        mode="train",
    ):
        self.mode = mode
        self.data_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".pkl")
            ]
        )
        np.random.seed(42)
        np.random.shuffle(self.data_files)
        if mode == "train":
            self.data_files = self.data_files[: int(0.9 * len(self.data_files))]
        else:
            self.data_files = self.data_files[int(0.9 * len(self.data_files)) :]
        self.video_path = "/home/veesion/Bag-detector/videos/"
        self.target_fps = target_fps
        self.duration = duration
        self.num_frames = int(target_fps * duration)
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

        # Get original frame dimensions
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Compute scaling factors
        scale_x = IMAGE_SIZE / original_width
        scale_y = IMAGE_SIZE / original_height

        # Compute the exact frame IDs to fetch
        frame_interval = video_fps / self.target_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > self.duration * video_fps:
            start_frame_id = np.random.choice(
                int(total_frames - self.duration * video_fps)
            )
        else:
            start_frame_id = 0
        frame_ids = [
            start_frame_id + int(round(i * frame_interval))
            for i in range(self.num_frames)
            if start_frame_id + i * frame_interval < total_frames
        ]

        for t, frame_id in enumerate(frame_ids):  # t is the index in our time dimension
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                frame_data = transforms.ToTensor()(frame_resized)
            else:
                frame_data = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))  # Placeholder

            frames.append(frame_data)

            # Process masks for each track_id
            for track_id in data["tracks"]:
                if frame_id in data["tracks"][track_id]:
                    _, contours, hierarchy = data["tracks"][track_id][frame_id]
                    class_id = data["classes"][track_id]

                    # Rescale contours to IMAGE_SIZE
                    rescaled_contours = [
                        (contour * np.array([scale_x, scale_y])).astype(np.int32)
                        for contour in contours
                    ]

                    # Draw resized contours
                    cv2.drawContours(
                        masks[class_id, t],
                        rescaled_contours,
                        -1,
                        255,
                        thickness=cv2.FILLED,
                        hierarchy=hierarchy,
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


# ------------------------------
# Training Loop
# ------------------------------

from torch.utils.tensorboard import SummaryWriter
import time


def train_model(data_dir, epochs=20, batch_size=BATCH_SIZE, lr=1e-4):
    surname = f"distilled_bag_semgentor_{int(time.time())}"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"checkpoints/{surname}", exist_ok=True)
    writer = SummaryWriter(f"tensorboard_logs/{surname}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = VideoMaskDataset(
        data_dir,
        mode="train",
        transform=transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),  # Rescale to [0,1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
        ),
    )
    val_dataset = VideoMaskDataset(
        data_dir,
        mode="val",
        transform=transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),  # Rescale to [0,1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
        ),
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    model = TemporalUNetTransformer(
        NUM_CLASSES, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE
    ).to(device)
    model = torch.compile(model)
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

        for frames, masks in train_dataloader:
            with torch.amp.autocast("cuda", dtype=torch.float32):
                frames, masks = frames.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(frames)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
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
        avg_train_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Accuracy/Train", pixel_accuracy, epoch + 1)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Train loss: {avg_train_loss:.4f}, "
            f"Pixel Accuracy: {pixel_accuracy:.4f}, Baseline Accuracy: {baseline_accuracy:.4f}"
        )
        model.eval()
        epoch_loss = 0
        correct_pixels = 0
        total_pixels = 0
        baseline_correct_pixels = 0  # When prediction is always zero

        for frames, masks in val_dataloader:
            with torch.amp.autocast("cuda", dtype=torch.float32):
                frames, masks = frames.to(device), masks.to(device)
                with torch.no_grad():
                    outputs = model(frames)
                    loss = criterion(outputs, masks)

            epoch_loss += loss.item()
            # print(outputs[0, :, 13, 112, 112])
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
        avg_val_loss = epoch_loss / len(val_dataloader)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch + 1)
        writer.add_scalar("Accuracy/Val", pixel_accuracy, epoch + 1)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Val loss: {avg_val_loss:.4f}, "
            f"Pixel Accuracy: {pixel_accuracy:.4f}, Baseline Accuracy: {baseline_accuracy:.4f}"
        )
        torch.save(
            model.state_dict(), f"checkpoints/{surname}/{surname}_epoch_{epoch}.pth"
        )

    print("Model training complete!")


# ------------------------------
# Run Training
# ------------------------------
if __name__ == "__main__":
    train_model(
        "/home/veesion/Bag-detector/valid_masks_tracks/",
    )
