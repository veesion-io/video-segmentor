import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from model import TemporalUNetTransformer  # Ensure your model class is in model.py

NUM_CLASSES = 13
TARGET_FPS = 5
VIDEO_DURATION = 5
IMAGE_SIZE = 224
NUM_FRAMES = int(VIDEO_DURATION * TARGET_FPS)
BATCH_SIZE = 4

# Color map for 13 classes
COLORS = [
    (255, 0, 0),  # Class 0 - Blue
    (0, 255, 0),  # Class 1 - Green
    (0, 0, 255),  # Class 2 - Red
    (255, 255, 0),  # Class 3 - Cyan
    (255, 0, 255),  # Class 4 - Magenta
    (0, 255, 255),  # Class 5 - Yellow
    (128, 0, 0),  # Class 6 - Dark Blue
    (0, 128, 0),  # Class 7 - Dark Green
    (0, 0, 128),  # Class 8 - Dark Red
    (128, 128, 0),  # Class 9 - Olive
    (128, 0, 128),  # Class 10 - Purple
    (0, 128, 128),  # Class 11 - Teal
    (128, 128, 128),  # Class 12 - Gray
]


def load_model(checkpoint_path):
    model = TemporalUNetTransformer(
        NUM_CLASSES, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE
    ).to("cuda")
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()
    }  # Remove _orig_mod. prefix
    model.load_state_dict(new_state_dict, strict=False)
    return model.eval()


def process_video(video_path, model):
    """
    Process the input video and predict segmentation masks using a fixed window duration and target FPS.
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not video_fps or video_fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS detected in video: {video_path}")

    # Compute frame interval to match TARGET_FPS
    frame_interval = video_fps / TARGET_FPS
    start_frame_id = 0  # Start from the beginning
    frame_ids = [
        start_frame_id + int(round(i * frame_interval))
        for i in range(NUM_FRAMES)
        if start_frame_id + i * frame_interval < total_frames
    ]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    frames = []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            frames.append(
                torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
            )  # Placeholder if frame missing
            continue

        frame_tensor = transform(frame)
        frames.append(frame_tensor)

    cap.release()

    # Ensure we have exactly NUM_FRAMES (pad with zeros if needed)
    while len(frames) < NUM_FRAMES:
        frames.append(torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)))

    frames = (
        torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to("cuda")
    )  # (1, C, T, H, W)

    with torch.no_grad():
        output_masks = torch.sigmoid(model(frames))  # Get probability maps
        binary_masks = (output_masks > 0.5).float()  # Convert to binary masks

    return binary_masks.cpu().numpy()


def save_masks_as_video(masks, output_path, fps=5):
    """
    Save the predicted masks as an MP4 video.
    """
    height, width = IMAGE_SIZE, IMAGE_SIZE
    num_frames = masks.shape[1]  # T dimension

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for t in range(num_frames):
        mask_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id in range(NUM_CLASSES):
            mask = masks[0, class_id, t]  # (H, W)
            color = np.array(COLORS[class_id], dtype=np.uint8)

            mask_frame[mask > 0] = color  # Apply class color
            if np.any(mask > 0):
                print("found amsks")
        out.write(mask_frame)

    out.release()
    print(f"Saved segmentation video to {output_path}")


def main(video_path, checkpoint_path, output_path):
    model = load_model(checkpoint_path)
    masks = process_video(video_path, model)
    save_masks_as_video(masks, output_path)


import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--video_dir",
        default="/home/veesion/Bag-detector/videos",
        type=str,
        help="Directory containing input videos",
    )
    parser.add_argument(
        "--output_dir",
        default="output_masks",
        type=str,
        help="Directory to save output masks",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_paths = glob.glob(os.path.join(args.video_dir, "*.mp4"))

    for video_path in video_paths:
        output_path = os.path.join(args.output_dir, os.path.basename(video_path))
        main(video_path, args.checkpoint_path, output_path)
