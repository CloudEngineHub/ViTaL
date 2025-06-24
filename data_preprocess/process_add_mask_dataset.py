import os
import pickle
import numpy as np
import cv2

TASK_NAME = ""
DATA_PATH = f"/path_to_data/{TASK_NAME}.pkl"
MASKS_DIR = f"/path_to_data/bcrl_pipeline/{TASK_NAME}/XMem/results"
OUTPUT_PATH = f"/path_to_data/{TASK_NAME}_add_mask.pkl"

TARGET_MASK_PATH = (
    f"/path_to_data/bcrl_pipeline/{TASK_NAME}/base/original_mask_target.png"
)
ORIGINAL_MASK_OBJECT_PATH = (
    f"/path_to_data/bcrl_pipeline/{TASK_NAME}/base/original_mask_object.png"
)
GRIPPER_MASK_PATH = f"/path_to_data/bcrl_pipeline/{TASK_NAME}/base/gripper_mask.png"

ORIGINAL_MASK_PLUG_PATH = (
    "/path_to_data/bcrl_pipeline/plug_insertion_base/original_mask_plug.png"
)

video_output_dir = f"/path_to_data/bcrl_pipeline/{TASK_NAME}/XMem/masked_videos"
os.makedirs(video_output_dir, exist_ok=True)

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

for idx, trajectory in enumerate(data["observations"]):
    frames = trajectory["pixels52"]
    binary_masks_list = []
    masked_frames_for_video = []

    masks_path = os.path.join(MASKS_DIR, str(idx), "masks")

    for frame_idx, frame in enumerate(frames):
        mask_file = os.path.join(masks_path, f"{frame_idx:04d}.png")
        if os.path.exists(mask_file):
            socket_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            plug_mask = cv2.imread(ORIGINAL_MASK_OBJECT_PATH, cv2.IMREAD_GRAYSCALE)
            plug_mask = cv2.resize(
                plug_mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            gripper_mask = cv2.imread(GRIPPER_MASK_PATH, cv2.IMREAD_GRAYSCALE)
            gripper_mask = cv2.resize(
                gripper_mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            combined_mask = np.maximum(np.maximum(socket_mask, gripper_mask), plug_mask)
            mask = cv2.resize(
                combined_mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            binary_mask = (mask > 0).astype(np.uint8)[..., None]
        else:
            print(
                f"Mask file {mask_file} not found, skipping frame {frame_idx} of trajectory {idx}"
            )
            binary_mask = np.zeros(frame.shape[:2], dtype=np.uint8)[..., None]

        binary_masks_list.append(binary_mask)

        masked_frame = frame * binary_mask

        masked_frames_for_video.append(masked_frame)

    trajectory["bitwise_masks"] = np.array(binary_masks_list)

    if len(masked_frames_for_video) > 0:
        h, w, _ = masked_frames_for_video[0].shape
        video_output_path = os.path.join(video_output_dir, f"masked_video_{idx}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_output_path, fourcc, 20, (w, h))
        for frame in masked_frames_for_video:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        video_writer.release()
        print(f"Saved masked video for trajectory {idx} at {video_output_path}")

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(data, f)

print(f"Dataset saved with bitwise masks added to each trajectory in {OUTPUT_PATH}")
print(f"Note that the original frame is not modified")
