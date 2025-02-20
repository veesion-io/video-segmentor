import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import glob
import argparse
import pickle
from model import TemporalUNetTransformer  # Ensure your model class is in model.py

NUM_CLASSES = 13
TARGET_FPS = 5
VIDEO_DURATION = 5
IMAGE_SIZE = 224
NUM_FRAMES = int(VIDEO_DURATION * TARGET_FPS)

# Color map for 13 classes
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
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


def extract_ground_truth_masks(pkl_path):
    """
    Extract ground truth masks from the .pkl file.
    Returns: (T, H, W, 3) numpy array with GT masks overlaid.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    masks = np.zeros(
        (NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
    )  # (T, H, W, 3)

    for track_id in data["tracks"]:
        for frame_id, (_, contours) in data["tracks"][track_id].items():
            class_id = data["classes"][track_id]
            if 0 <= class_id < NUM_CLASSES:
                cv2.drawContours(
                    masks[frame_id],
                    contours,
                    -1,
                    COLORS[class_id],
                    thickness=cv2.FILLED,
                )

    return masks


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

    frame_interval = video_fps / TARGET_FPS
    start_frame_id = 0
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
            frames.append(torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)))
            continue

        frame_tensor = transform(frame)
        frames.append(frame_tensor)

    cap.release()

    while len(frames) < NUM_FRAMES:
        frames.append(torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)))

    frames = (
        torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to("cuda")
    )  # (1, C, T, H, W)

    with torch.no_grad():
        output_masks = torch.sigmoid(model(frames))  # Get probability maps
        binary_masks = (output_masks > 0.5).float()  # Convert to binary masks

    return binary_masks.cpu().numpy(), frame_ids


def save_comparison_video(
    video_path, masks, frame_ids, gt_masks, output_path, fps=TARGET_FPS, alpha=0.5
):
    """
    Save a side-by-side comparison video with:
    - Left side: Original frames with **Predicted Segmentation**
    - Right side: Original frames with **Ground Truth Segmentation**
    """
    cap = cv2.VideoCapture(video_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    combined_width = original_width * 2  # Double the width to fit side-by-side

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, original_height))

    for t, frame_id in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        mask_frame_pred = np.zeros_like(frame, dtype=np.uint8)  # Blank mask
        for class_id in range(NUM_CLASSES):
            mask = masks[0, class_id, t]  # (H, W)
            color = np.array(COLORS[class_id], dtype=np.uint8)

            resized_mask = cv2.resize(
                mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST
            )
            mask_indices = resized_mask > 0
            mask_frame_pred[mask_indices] = color  # Apply color mask

        blended_pred = frame.copy()
        mask_indices = mask_frame_pred.sum(axis=2) > 0
        blended_pred[mask_indices] = (
            (1 - alpha) * frame[mask_indices] + alpha * mask_frame_pred[mask_indices]
        ).astype(np.uint8)

        # Ground truth overlay
        gt_mask_resized = cv2.resize(
            gt_masks[t],
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
        blended_gt = frame.copy()
        mask_indices_gt = gt_mask_resized.sum(axis=2) > 0
        blended_gt[mask_indices_gt] = (
            (1 - alpha) * frame[mask_indices_gt]
            + alpha * gt_mask_resized[mask_indices_gt]
        ).astype(np.uint8)

        # Concatenate left (predicted) and right (ground truth)
        combined_frame = np.concatenate((blended_pred, blended_gt), axis=1)

        out.write(combined_frame)

    cap.release()
    out.release()
    print(f"Saved comparison video to {output_path}")


def main(video_path, pkl_path, checkpoint_path, output_path):
    model = load_model(checkpoint_path)
    masks, frame_ids = process_video(video_path, model)
    gt_masks = extract_ground_truth_masks(pkl_path)
    save_comparison_video(video_path, masks, frame_ids, gt_masks, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--video_dir",
        default="/home/veesion/Bag-detector/videos",
        type=str,
        required=False,
        help="Directory containing input videos",
    )
    parser.add_argument(
        "--output_dir",
        default="comparison_videos",
        type=str,
        help="Directory to save output",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pkl_dir = "/home/veesion/Bag-detector/valid_masks_tracks/"

    video_paths = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    pkl_paths = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))

    video_paths = glob.glob(os.path.join(args.video_dir, "*.mp4"))
    data_files = sorted(
        [os.path.splitext(f)[0] for f in os.listdir(pkl_dir) if f.endswith(".pkl")]
    )
    np.random.seed(42)
    np.random.shuffle(data_files)
    data_files = data_files[: int(0.9 * len(data_files))]

    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        if os.path.splitext(video_name)[0] not in data_files:
            continue
        output_path = os.path.join(args.output_dir, video_name)
        pkl_path = os.path.join(pkl_dir, video_name + ".pkl")
        main(video_path, pkl_path, args.checkpoint_path, output_path)
