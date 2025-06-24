import os
import pickle
import numpy as np
import torch
import cv2
from pathlib import Path

TASK_NAME = ""
DATA_PATH = f"/path_to_data/{TASK_NAME}.pkl"
OUTPUT_VIDEO_DIR = Path(f"/path_to_data/bcrl_pipeline/{TASK_NAME}/XMem/videos")

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

print("Creating video files from trajectories...")
for idx, trajectory in enumerate(data["observations"]):
    frames = trajectory["pixels52"]

    original_height, original_width, _ = frames[0].shape

    video_path = OUTPUT_VIDEO_DIR / f"{idx}.mp4"

    video_writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (original_width, original_height),
    )

    for frame_idx, frame_np in enumerate(frames):
        video_writer.write(frame_np)

    video_writer.release()
    torch.cuda.empty_cache()
    print(f"Saved video for trajectory {idx} at {video_path}")

print("Extracting frames from video files...")

for i, trajectory in enumerate(data["observations"]):
    video_file = OUTPUT_VIDEO_DIR / f"{i}.mp4"
    if not video_file.exists():
        print(f"Video file {video_file} does not exist. Skipping...")
        continue

    output_frames_dir = OUTPUT_VIDEO_DIR / f"{i}"
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_file))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"{frame_idx:04d}.jpg"
        frame_path = output_frames_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1

    cap.release()
    print(
        f"Processed {video_file}: extracted {frame_idx} frames to {output_frames_dir}"
    )
