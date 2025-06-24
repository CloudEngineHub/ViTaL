import os
import cv2
import torch
import numpy as np
from pathlib import Path
from inference.inference_core import InferenceCore
from model.network import XMem
from tqdm import tqdm

TASK_NAME = ""

SAM_MASK_DIR = Path(f"/path_to_data/bcrl_pipeline/{TASK_NAME}/sam2/masked_image")
VIDEOS_DIR = Path(
    f"/path_to_data/bcrl_pipeline/{TASK_NAME}/XMem/videos"
)  # Assuming raw videos are here
RESULTS_DIR = Path(f"/path_to_data/bcrl_pipeline/{TASK_NAME}/XMem/results")

MODEL_PATH = "./saves/XMem.pth"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

num_videos = 32

# XMem configuration
config = {
    "model": MODEL_PATH,
    "dataset": "G",
    "split": "val",
    "save_all": True,
    "benchmark": False,
    "disable_long_term": False,
    "max_mid_term_frames": 10,
    "min_mid_term_frames": 5,
    "max_long_term_elements": 10000,
    "num_prototypes": 128,
    "top_k": 30,
    "mem_every": 5,
    "deep_update_every": -1,
    "save_scores": False,
    "flip": False,
    "enable_long_term": True,
    "key_dim": 64,
    "value_dim": 512,
    "hidden_dim": 64,
    "enable_long_term_count_usage": False,
}


def process_video_with_xmem(video_path, initial_mask_path, output_dir, network, config):
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_mask = cv2.imread(str(initial_mask_path), cv2.IMREAD_GRAYSCALE)
    if initial_mask is None:
        print(f"Error reading mask from {initial_mask_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    processor = InferenceCore(network, config)
    processor.set_all_labels([1])

    frame_idx = 0

    print(f"Processing {total_frames} frames...")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to 768x768 (same as in server)
            frame_resized = cv2.resize(
                frame, (768, 768), interpolation=cv2.INTER_LINEAR
            )

            frame_tensor = (
                torch.from_numpy(frame_resized).permute(2, 0, 1).float().cuda() / 255.0
            )

            if frame_idx == 0:
                mask_resized = cv2.resize(
                    initial_mask, (768, 768), interpolation=cv2.INTER_NEAREST
                )
                mask_tensor = torch.from_numpy(mask_resized.copy()).unsqueeze(0).cuda()
                mask_tensor = (mask_tensor > 128).long()  # Binarize mask

                prob = processor.step(frame_tensor, mask_tensor, [1])
                del mask_tensor
            else:
                prob = processor.step(frame_tensor)

            mask_pred = torch.argmax(prob, dim=0).cpu().numpy().astype(np.uint8)

            original_height, original_width = frame.shape[:2]
            if (original_height, original_width) != (768, 768):
                mask_pred = cv2.resize(
                    mask_pred,
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            mask_output = (mask_pred * 255).astype(np.uint8)
            mask_filename = output_dir / f"{frame_idx:05d}.png"
            cv2.imwrite(str(mask_filename), mask_output)
            del frame_tensor, prob

            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames")

    cap.release()
    torch.cuda.empty_cache()

    print(f"Completed processing {frame_idx} frames")
    return True


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("Loading XMem model...")
    network = XMem({}, MODEL_PATH).cuda().eval()
    network.load_weights(torch.load(MODEL_PATH), init_as_zero_if_needed=True)
    for i in range(num_videos):
        print(f"\n----- Processing video {i}/{num_videos-1} -----")

        mask_file = SAM_MASK_DIR / f"mask_{i}.png"
        video_file = VIDEOS_DIR / f"{i}.mp4"  # Adjust extension if needed
        output_dir = RESULTS_DIR / f"{i}"

        if not mask_file.exists():
            print(f"Warning: {mask_file} does not exist. Skipping...")
            continue

        if not video_file.exists():
            for ext in [".avi", ".mov", ".MOV", ".MP4"]:
                alt_video = VIDEOS_DIR / f"{i}{ext}"
                if alt_video.exists():
                    video_file = alt_video
                    break
            else:
                print(f"Warning: No video file found for index {i}. Skipping...")
                continue

        success = process_video_with_xmem(
            video_path=video_file,
            initial_mask_path=mask_file,
            output_dir=output_dir,
            network=network,
            config=config,
        )

        if success:
            print(f"Successfully processed video {i}")
        else:
            print(f"Failed to process video {i}")

        torch.cuda.empty_cache()

    print("\nAll videos processed!")


if __name__ == "__main__":
    main()
