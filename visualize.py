import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import glob
import argparse
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
    """
    Load the trained segmentation model.
    """
    model = TemporalUNetTransformer(
        NUM_CLASSES, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE
    ).to("cuda")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    new_state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()
    }  # Remove _orig_mod. prefix
    model.load_state_dict(new_state_dict, strict=False)
    return model.eval()


def process_video(video_path, model):
    """
    Process the input video and predict segmentation masks using a fixed window duration and target FPS.
    Returns:
        - binary_masks: numpy array of shape (1, NUM_CLASSES, T, H, W)
        - frame_ids: list of selected frame indices
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not video_fps or video_fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS detected in video: {video_path}")

    # Compute exact frame selection to maintain alignment
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
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),  # Keep original frames untouched
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    frames = []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            frames.append(torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)))
            continue

        frame_tensor = transform(frame)
        frames.append(frame_tensor)

    cap.release()

    # Ensure we have exactly NUM_FRAMES (pad with black frames if needed)
    while len(frames) < NUM_FRAMES:
        frames.append(torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)))

    frames = (
        torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to("cuda")
    )  # (1, C, T, H, W)

    with torch.no_grad():
        output_masks = torch.sigmoid(model(frames))
        binary_masks = (output_masks > 0.5).float()

    return binary_masks.cpu().numpy(), frame_ids


def save_masks_as_video(
    video_path, masks, frame_ids, output_path, fps=TARGET_FPS, alpha=0.5
):
    """
    Save the original video with segmentation masks overlaid correctly.
    Reads frames explicitly using frame_ids to ensure proper alignment.
    """
    cap = cv2.VideoCapture(video_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

    for t, frame_id in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        mask_frame = np.zeros_like(frame, dtype=np.uint8)  # Blank mask
        for class_id in range(NUM_CLASSES):
            mask = masks[0, class_id, t]  # (H, W)
            color = np.array(COLORS[class_id], dtype=np.uint8)

            # Resize mask to match the original frame size
            resized_mask = cv2.resize(
                mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST
            )

            # Apply mask only on detected areas
            mask_indices = resized_mask > 0
            mask_frame[mask_indices] = color  # Colorize detected areas

        # Blend only the mask regions onto the original frame
        blended_frame = frame.copy()
        mask_indices = mask_frame.sum(axis=2) > 0  # Detect where masks exist
        blended_frame[mask_indices] = (
            (1 - alpha) * frame[mask_indices] + alpha * mask_frame[mask_indices]
        ).astype(np.uint8)

        out.write(blended_frame)

    cap.release()
    out.release()
    print(f"Saved segmented video to {output_path}")


def main(video_path, checkpoint_path, output_path):
    model = load_model(checkpoint_path)
    masks, frame_ids = process_video(
        video_path, model
    )  # Get masks & exact frame selection
    save_masks_as_video(video_path, masks, frame_ids, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
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
        help="Directory to save output videos",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_paths = glob.glob(os.path.join(args.video_dir, "*.mp4"))
    data_dir = "/home/veesion/Bag-detector/valid_masks_tracks/"
    data_files = sorted(
        [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".pkl")]
    )
    np.random.seed(42)
    np.random.shuffle(data_files)
    data_files = data_files[: int(0.9 * len(data_files))]

    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        if os.path.splitext(video_name)[0] not in data_files:
            continue
        output_path = os.path.join(args.output_dir, video_name)
        main(video_path, args.checkpoint_path, output_path)
