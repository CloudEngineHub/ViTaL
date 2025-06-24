import os
import pickle
import numpy as np
import cv2


def crop_demos(input_file, output_file, videos_dir, fps=30):
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    observations = data.get("observations", [])
    processed_observations = []
    os.makedirs(videos_dir, exist_ok=True)

    for demo_idx, rollout in enumerate(observations):
        processed_rollout = rollout.copy()
        if "pixels52" in rollout:
            pixels52 = rollout["pixels52"]
            T, H, W, C = pixels52.shape

            crop_height = int(0.8 * H)
            crop_width = int(0.5 * W)

            start_row = H - crop_height
            start_col = (W - crop_width) // 2 + 10

            masked_frames = []
            print(
                f"Demo {demo_idx}: Cropping {T} frames to region ({crop_height}, {crop_width})"
            )

            for t in range(T):
                frame = pixels52[t]
                masked_frame = np.zeros_like(frame)

                # Extract and normalize region
                region = frame[
                    start_row : start_row + crop_height,
                    start_col : start_col + crop_width,
                ]
                if region.dtype != np.uint8:
                    region = (
                        255
                        * (region - region.min())
                        / (region.max() - region.min() + 1e-6)
                    ).astype(np.uint8)

                masked_frame[
                    start_row : start_row + crop_height,
                    start_col : start_col + crop_width,
                ] = region
                masked_frames.append(masked_frame)
            cropped_frames = np.array(masked_frames)
            processed_rollout["pixels52"] = cropped_frames

            video_path = os.path.join(videos_dir, f"demo_{demo_idx}_cropped.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

            for frame in cropped_frames:
                out.write(frame)
            out.release()

            print(f"Demo {demo_idx}: Saved cropped video to {video_path}")
        else:
            print(f"Demo {demo_idx}: No 'pixels52' key present, skipping")

        processed_observations.append(processed_rollout)

    new_data = data.copy()
    new_data["observations"] = processed_observations

    with open(output_file, "wb") as f:
        pickle.dump(new_data, f)

    print(f"\nProcessing complete!")
    print(f"Saved cropped demonstrations to {output_file}")
    print(f"Saved {len(processed_observations)} cropped videos to {videos_dir}")


def main():
    TASK_NAME = ""
    input_file = f"/path_to_data/{TASK_NAME}_add_mask.pkl"
    output_file = f"/path_to_data/{TASK_NAME}_add_mask_cropped.pkl"
    videos_dir = f"/path_to_data/{TASK_NAME}/cropped_videos"
    crop_demos(input_file, output_file, videos_dir, fps=30)


if __name__ == "__main__":
    main()
