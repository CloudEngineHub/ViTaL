from enum import IntEnum
import re
import time
import csv
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


class ActionType(IntEnum):
    CONTINUOUS = 0
    DISCRETE = 1


class ActorCriticType(IntEnum):
    MLP = 0
    GPT = 1


class ObsType(IntEnum):
    PIXELS = 0
    FEATURES = 1


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()
        # Keep track of evaluation time so that total time only includes train time
        self._eval_start_time = 0
        self._eval_time = 0
        self._eval_flag = False

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time - self._eval_time
        return elapsed_time, total_time

    def eval(self):
        if not self._eval_flag:
            self._eval_flag = True
            self._eval_start_time = time.time()
        else:
            self._eval_time += time.time() - self._eval_start_time
            self._eval_flag = False
            self._eval_start_time = 0

    def total_time(self):
        return time.time() - self._start_time - self._eval_time


class TruncatedNormal(tfp.distributions.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        self.low = low
        self.high = high
        self.eps = eps
        super().__init__(loc, scale)

    def _clamp(self, x):
        return jnp.clip(x, self.low + self.eps, self.high - self.eps)

    # TODO: Add a clip argument
    def sample(self, key, sample_shape=()):
        return self._clamp(super().sample(key, sample_shape))


def get_stddev_schedule(stddev_schedule: str):
    try:
        const_schedule = float(stddev_schedule)
        return lambda step: const_schedule
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", stddev_schedule)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            init = jnp.array(init, dtype=jnp.float32)
            final = jnp.array(final, dtype=jnp.float32)
            duration = jnp.array(duration, dtype=jnp.float32)

            def linear_schedule(step):
                mix = jnp.clip(step / duration, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final

            return linear_schedule
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", stddev_schedule)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]

            def step_linear_schedule(step):
                if step <= duration1:
                    mix = np.clip(step / duration1, 0.0, 1.0)
                    return (1.0 - mix) * init + mix * final1
                else:
                    mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                    return (1.0 - mix) * final1 + mix * final2

            return step_linear_schedule
    raise NotImplementedError(stddev_schedule)


def plot_reward_vs_timestep(
    key, timesteps, key_rewards_norm, episode, work_dir, normalize
):
    plt.figure(figsize=(10, 6))
    plt.plot(
        timesteps, key_rewards_norm, marker="o", linestyle="-", label=f"{key} Reward"
    )
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(f"Reward vs. Timestep for '{key}' (Episode {episode})")
    plt.legend()
    plt.grid(True)

    if key == "combined":
        rewards_dir = work_dir / "rewards_combined"
        rewards_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = rewards_dir / f"reward_{episode}_{key}.png"

    else:
        if normalize:
            rewards_dir = work_dir / "rewards_normalized"
            rewards_dir.mkdir(parents=True, exist_ok=True)
            plot_filename = rewards_dir / f"reward_{episode}_{key}_normalized.png"
        else:
            rewards_dir = work_dir / "rewards"
            rewards_dir.mkdir(parents=True, exist_ok=True)
            plot_filename = rewards_dir / f"reward_{episode}_{key}.png"

    plt.savefig(plot_filename)
    if key == "combined":
        plt.savefig(work_dir / "rewards_combined")
    plt.close()


def record_and_plot_key_rewards(
    csv_path, irl_rewards_history, episode_outcomes, work_dir
):
    """
    Records the per-key IRL reward history into a CSV file and generates side-by-side plots for each key.

    For each key, a single figure is created with three subplots (side-by-side):
      1. Combined: All episodes (green if outcome True, red if False).
      2. Success Only: Only episodes with outcome True (green lines).
      3. Failure Only: Only episodes with outcome False (red lines).

    The reward values for each key are normalized to [0,1] based on the overall min and max
    across all valid episodes for that key. Additionally, a combined reward (sum over all keys)
    is computed and similarly plotted.

    Parameters:
        csv_path (Path or str): Path to the CSV file to write the reward history.
        irl_rewards_history (dict): Mapping each key to a list of reward sequences (one per episode).
        episode_outcomes (list): List of booleans for each episode (True = success, False = failure).
        work_dir (Path or str): Working directory where the plots will be saved (in "rewards_history" subfolder).
    """
    eps = 1e-8
    rewards_dir = work_dir / "rewards_history"
    rewards_dir.mkdir(parents=True, exist_ok=True)

    num_episodes = min(len(episodes) for episodes in irl_rewards_history.values())

    with csv_path.open(mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Key", "RewardSequence", "Outcome"])
        for ep_idx in range(num_episodes):
            valid = True
            for key, episodes in irl_rewards_history.items():
                reward_seq = episodes[ep_idx]
                if len(reward_seq) == 0 or reward_seq[0] > -0.01 or ep_idx < 5:
                    valid = False
                    break
            if not valid:
                continue
            for key, episodes in irl_rewards_history.items():
                reward_seq = episodes[ep_idx]
                reward_str = ",".join(map(str, reward_seq))
                outcome = (
                    episode_outcomes[ep_idx] if ep_idx < len(episode_outcomes) else ""
                )
                writer.writerow([ep_idx + 1, key, reward_str, outcome])

    valid_data = {}
    for key, episodes in irl_rewards_history.items():
        valid_episodes = []
        valid_outcomes = []
        for ep_idx in range(num_episodes):
            reward_seq = episodes[ep_idx]
            if len(reward_seq) == 0 or reward_seq[0] > -0.01 or ep_idx < 5:
                continue
            valid_episodes.append(np.array(reward_seq))
            valid_outcomes.append(
                episode_outcomes[ep_idx] if ep_idx < len(episode_outcomes) else False
            )
        valid_data[key] = (valid_episodes, valid_outcomes)

    def plot_side_by_side(key_label, episodes, outcomes, suffix):
        """
        Creates a side-by-side figure with 3 subplots:
          - Combined: All episodes (green if success, red if failure).
          - Success Only: Only episodes with outcome True (green lines).
          - Failure Only: Only episodes with outcome False (red lines).
        The y-axis is normalized to [0, 1] based on the overall data.
        """
        if not episodes:
            return
        # Compute overall min and max across all episodes.
        all_rewards = np.concatenate(episodes)
        overall_min = np.min(all_rewards)
        overall_max = np.max(all_rewards)
        normalize = lambda x: (x - overall_min) / (overall_max - overall_min + eps)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        axs[0].set_title(f"{key_label} Reward History (Combined)")
        axs[1].set_title(f"{key_label} Reward History (Success Only)")
        axs[2].set_title(f"{key_label} Reward History (Failure Only)")
        for ax in axs:
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Normalized Reward")
            ax.grid(True)
            ax.set_ylim(0, 1)

        for seq, outcome in zip(episodes, outcomes):
            norm_seq = normalize(seq)
            x = np.arange(len(norm_seq))
            color = "green" if outcome else "red"
            axs[0].plot(x, norm_seq, color=color, alpha=0.7)
            if outcome:
                axs[1].plot(x, norm_seq, color="green", alpha=0.7)
            else:
                axs[2].plot(x, norm_seq, color="red", alpha=0.7)

        fig.tight_layout()
        plot_path = rewards_dir / f"{key_label}_{suffix.replace(' ', '_').lower()}.png"
        fig.savefig(plot_path)
        plt.close(fig)

    # Create side-by-side plots for each key.
    for key, (episodes, outcomes) in valid_data.items():
        plot_side_by_side(key, episodes, outcomes, "side_by_side")

    # Compute combined reward across all keys for each episode (only if every key is valid for that episode).
    combined_rewards = []
    combined_outcomes = []
    for ep_idx in range(num_episodes):
        valid = True
        episode_sum = None
        for key, episodes in irl_rewards_history.items():
            reward_seq = np.array(episodes[ep_idx])
            if len(reward_seq) == 0 or reward_seq[0] > -0.01 or ep_idx < 5:
                valid = False
                break
            if episode_sum is None:
                episode_sum = reward_seq
            else:
                if len(reward_seq) < len(episode_sum):
                    reward_seq = np.concatenate(
                        [
                            reward_seq,
                            np.full(len(episode_sum) - len(reward_seq), np.nan),
                        ]
                    )
                elif len(reward_seq) > len(episode_sum):
                    episode_sum = np.concatenate(
                        [
                            episode_sum,
                            np.full(len(reward_seq) - len(episode_sum), np.nan),
                        ]
                    )
                episode_sum = np.nansum(np.vstack([episode_sum, reward_seq]), axis=0)
        if valid and episode_sum is not None:
            combined_rewards.append(episode_sum)
            combined_outcomes.append(
                episode_outcomes[ep_idx] if ep_idx < len(episode_outcomes) else False
            )

    if combined_rewards:
        plot_side_by_side(
            "combined", combined_rewards, combined_outcomes, "side_by_side"
        )

    min_rewards = []
    min_outcomes = []
    for ep_idx in range(num_episodes):
        valid = True
        key_seqs = []
        for key, episodes in irl_rewards_history.items():
            reward_seq = np.array(episodes[ep_idx])
            if len(reward_seq) == 0 or reward_seq[0] > -0.01 or ep_idx < 5:
                valid = False
                break
            key_seqs.append(reward_seq)
        if not valid or not key_seqs:
            continue
        # Pad sequences to the maximum length.
        max_len = max(len(seq) for seq in key_seqs)
        padded_seqs = []
        for seq in key_seqs:
            pad_width = max_len - len(seq)
            if pad_width > 0:
                padded = np.concatenate([seq, np.full(pad_width, np.nan)])
            else:
                padded = seq
            padded_seqs.append(padded)
        # Compute elementwise minimum ignoring NaNs.
        episode_min = np.nanmin(np.stack(padded_seqs, axis=0), axis=0)
        min_rewards.append(episode_min)
        min_outcomes.append(
            episode_outcomes[ep_idx] if ep_idx < len(episode_outcomes) else False
        )

    if min_rewards:
        plot_side_by_side("min_combined", min_rewards, min_outcomes, "side_by_side")


def apply_crop_view(image):
    """
    Crop the view of the input image by keeping a sub-region (with dimensions 0.8x0.5 of 128)
    and zeroing out the rest. Assumes the input image is in [H, W, C] format.
    For an image of size 128x128, the crop is defined as:
      crop_height = int(0.8 * 128)
      crop_width  = int(0.5 * 128)
      start_row   = 128 - crop_height
      start_col   = (128 - crop_width) // 2 + 10
    """
    crop_height = int(0.8 * 128)
    crop_width = int(0.5 * 128)
    start_row = 128 - crop_height
    start_col = (128 - crop_width) // 2 + 10
    # Create a blank (black) image of the same size.
    cropped_image = np.zeros_like(image)
    # Copy the cropped region from the input image.
    cropped_image[
        start_row : start_row + crop_height, start_col : start_col + crop_width, :
    ] = image[
        start_row : start_row + crop_height, start_col : start_col + crop_width, :
    ]
    return cropped_image


# def std_scheduler(global_step, initial_std=0.5, final_std=0.1, decay_steps=100000):
#     return final_std + (initial_std - final_std) * np.exp(-global_step / decay_steps)


def plot_norms(norms, work_dir):
    plt.figure()
    # n_norm = (np.array(norms[key]) - np.array(norms[key]).min()) / (np.array(norms[key]).max() - np.array(norms[key]).min())
    # n_max = (np.array(maxs[key]) - np.array(maxs[key]).min()) / (np.array(maxs[key]).max() - np.array(maxs[key]).min())

    plt.plot(norms, label=f"Norm")
    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.title(f"Norm")
    plt.legend()
    plt.grid()

    plot_path = work_dir / f"norm_max_plot.png"
    plt.savefig(plot_path)
    plt.close()


def mark_point_on_image(
    src_path: Path,
    dst_path: Path,
    x: int,
    y: int,
    radius: int = 2,
    outline: tuple = (255, 0, 0),
    width: int = 3,
):
    img = Image.open(src_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        outline=outline,
        width=width,
    )
    font = ImageFont.load_default()
    draw.text((x + radius + 2, y - radius), f"({x},{y})", fill=outline, font=font)
    img.save(dst_path)
    print(f"Saved marked image to {dst_path}")


def pixel_to_camera_frame(point2d, depth, K):
    """
    Un-project a 2D pixel (u,v) with depth Z into 3D camera frame.
    Returns (X,Y,Z) in camera coords.
    """
    u, v = point2d
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])


def to_pixel(value: float, dim: int) -> int:
    """
    Convert a normalized or percentage value to pixel coordinate.
    """
    if 0.0 <= value <= 1.0:
        return int(value * dim)
    elif 1.0 < value <= 100.0:
        return int((value / 100.0) * dim)
    else:
        return int(value)
