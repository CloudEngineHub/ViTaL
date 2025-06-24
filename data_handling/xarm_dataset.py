from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pickle as pkl
from jax import numpy as jnp

import jax
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import cv2
from data_handling.dataset import Dataset

from img_utils import jax_color_jitter, jax_random_crop


def get_relative_action(actions):
    """
    Convert absolute axis angle actions to relative axis angle actions
    Action has both position and orientation. Convert to transformation matrix, get
    relative transformation matrix, convert back to axis angle
    """

    relative_actions = []
    for i in range(len(actions)):
        # Get relative transformation matrix
        # previous pose
        pos_prev = actions[i, :3]
        ori_prev = actions[i, 3:6]
        r_prev = R.from_rotvec(ori_prev).as_matrix()
        matrix_prev = np.eye(4)
        matrix_prev[:3, :3] = r_prev
        matrix_prev[:3, 3] = pos_prev
        # current pose
        next_idx = min(i + 1, len(actions) - 1)
        pos = actions[next_idx, :3]
        ori = actions[next_idx, 3:6]
        gripper = actions[next_idx, 6:]
        r = R.from_rotvec(ori).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = r
        matrix[:3, 3] = pos
        # relative transformation
        matrix_rel = np.linalg.inv(matrix_prev) @ matrix
        # relative pose
        # pos_rel = matrix_rel[:3, 3]
        pos_rel = pos - pos_prev
        r_rel = R.from_matrix(matrix_rel[:3, :3]).as_rotvec()
        relative_actions.append(np.concatenate([pos_rel, r_rel, gripper]))
    # last action
    last_action = np.zeros_like(actions[-1])
    last_action[-1] = actions[-1][-1]
    while len(relative_actions) < len(actions):
        relative_actions.append(last_action)
    return np.array(relative_actions, dtype=np.float32)


def get_quaternion_orientation(cartesian):
    """
    Get quaternion orientation from axis angle representation
    """
    new_cartesian = []
    for i in range(len(cartesian)):
        pos = cartesian[i, :3]
        ori = cartesian[i, 3:]
        quat = R.from_rotvec(ori).as_quat()
        new_cartesian.append(np.concatenate([pos, quat], axis=-1))
    return np.array(new_cartesian, dtype=np.float32)


class XarmDataset(Dataset):
    def __init__(
        self,
        path: str,
        temporal_agg: bool,
        bg_augs: bool,
        bg_aug_threshold: float,
        crop_view: bool,
        num_queries: int,
        img_size: int,
        pixel_keys: List[str],
        aux_keys: List[str],
        relative_actions: bool,
        subsample: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self.path = Path(path)
        self.temporal_agg = temporal_agg
        self.num_queries = num_queries
        self.img_size = img_size
        self.pixel_keys = pixel_keys
        self.aux_keys = aux_keys
        self.relative_actions = relative_actions
        self.subsample = subsample

        self._num_sensors = 2

        self._max_episode_len = 0
        self._num_samples = 0
        self.stats = {
            "max": defaultdict(lambda: -np.inf),
            "min": defaultdict(lambda: np.inf),
        }

        self.dataset_dict = {}
        self.bg_augs = bg_augs
        self.bg_aug_threshold = bg_aug_threshold
        self.crop_view = crop_view
        print(f"Loading {str(self.path)}")
        self._load_dataset()
        super().__init__(
            self.dataset_dict,
            image_augs=self.aug,
            bg_augs=bg_augs,
            bg_aug_threshold=bg_aug_threshold,
            crop_view=self.crop_view,
        )

    def _load_dataset(self):
        data = pkl.load(open(self.path, "rb"))
        observations = data["observations"]
        episodes = []
        for i, obs in enumerate(observations):
            actions = np.concatenate(
                [
                    obs["cartesian_states"],
                    obs["gripper_states"][:, None],
                ],
                axis=1,
            )
            if len(actions) == 0:
                continue

            if self.subsample is not None:
                for key in obs.keys():
                    obs[key] = obs[key][:: self.subsample]
                actions = actions[:: self.subsample]

            if self.relative_actions:
                actions = get_relative_action(actions)

            obs["cartesian_states"] = get_quaternion_orientation(
                obs["cartesian_states"]
            )
            if any("sensor" in k for k in self.aux_keys):
                try:
                    sensor_baseline = np.median(
                        obs["sensor_states"][:5], axis=0, keepdims=True
                    )
                    obs["sensor_states"] = obs["sensor_states"] - sensor_baseline
                    self.stats["max"]["sensor"] = np.maximum(
                        self.stats["max"]["sensor"],
                        np.max(obs["sensor_states"], axis=0),
                    )
                    self.stats["min"]["sensor"] = np.zeros_like(
                        self.stats["max"]["sensor"]
                    )
                    for sensor_idx in range(self._num_sensors):
                        obs[f"sensor{sensor_idx}"] = obs["sensor_states"][
                            ..., sensor_idx * 15 : (sensor_idx + 1) * 15
                        ]
                except KeyError:
                    raise ValueError(
                        "Sensor states in aux_keys but not in observations"
                    )

            episode = dict(
                observation=obs,
                action=actions,
            )

            episodes.append(episode)
            self._max_episode_len = max(self._max_episode_len, len(actions))
            self._num_samples += len(actions)

            self.stats["min"]["action"] = np.minimum(
                self.stats["min"]["action"], np.min(actions, axis=0)
            )
            self.stats["max"]["action"] = np.maximum(
                self.stats["max"]["action"], np.max(actions, axis=0)
            )

        self.stats["max"]["proprioceptive"] = np.concatenate(
            [data["max_cartesian"][:3], [1] * 4, [data["max_gripper"]]]
        )
        self.stats["min"]["proprioceptive"] = np.concatenate(
            [data["min_cartesian"][:3], [-1] * 4, [data["min_gripper"]]]
        )

        self.stats["min"]["action"][3:] = 0
        self.stats["max"]["action"][3:] = 1

        if any("sensor" in k for k in self.aux_keys):
            for _sidx in range(self._num_sensors):
                # Compute stats for individual sensors
                sensor_mask = np.zeros_like(self.stats["min"]["sensor"], dtype=bool)
                sensor_mask[_sidx * 15 : (_sidx + 1) * 15] = True
                self.stats["min"][f"sensor{_sidx}"] = self.stats["min"]["sensor"][
                    sensor_mask
                ]
                self.stats["max"][f"sensor{_sidx}"] = self.stats["max"]["sensor"][
                    sensor_mask
                ]

                sensor_states = np.concatenate(
                    [obs[f"sensor{_sidx}"] for obs in observations], axis=0
                )
                sensor_std = np.std(sensor_states, axis=0).reshape((5, 3)).max(axis=0)
                sensor_std[:2] = sensor_std[:2].max()
                sensor_std = np.clip(sensor_std * 3, a_min=100, a_max=None)
                self.stats["max"][f"sensor{_sidx}"] = np.tile(
                    sensor_std, int(self.stats["max"][f"sensor{_sidx}"].shape[0] / 3)
                )

        self.stats["min"] = dict(self.stats["min"])
        self.stats["max"] = dict(self.stats["max"])

        def augment_image(image: jax.Array, rng: jax.Array) -> jax.Array:
            # image = image / 255.0
            image, rng = jax.jit(
                jax_random_crop,
                static_argnames=("crop_height", "crop_width", "padding"),
            )(image, rng, self.img_size, self.img_size, padding=4)
            image, rng = jax.jit(
                jax_color_jitter,
                static_argnames=("brightness", "contrast", "saturation"),
            )(image, rng, 0.3, 0.3, 0.2)
            return image

        self.aug = augment_image
        self._aggregate_dataset_dict(episodes)

    def _aggregate_dataset_dict(self, episode_list):
        dataset_dict = defaultdict(list)
        for i, episode in enumerate(episode_list):
            if self.bg_augs:
                dataset_dict["bitwise_masks"].append(
                    jnp.array(episode["observation"]["bitwise_masks"])
                )
            for key in self.pixel_keys:
                dataset_dict[key].append(
                    jnp.array(self.preprocess(episode["observation"][key], key))
                )
            for key in self.aux_keys:
                dataset_dict[key].append(
                    jnp.array(self.preprocess(episode["observation"][key], key))
                )
            dataset_dict["proprioceptive"].append(
                jnp.concatenate(
                    [
                        jnp.array(episode["observation"]["cartesian_states"]),
                        jnp.array(episode["observation"]["gripper_states"][:, None]),
                    ],
                    axis=1,
                )
            )
            if self.temporal_agg:
                # num_actions = self.num_queries
                act = np.zeros(
                    (
                        len(episode["action"]),
                        self.num_queries,
                        episode["action"].shape[-1],
                    )
                )
                act[: len(episode["action"]) - self.num_queries + 1] = (
                    np.lib.stride_tricks.sliding_window_view(
                        episode["action"],
                        (self.num_queries, episode["action"].shape[-1]),
                    ).squeeze()
                )
                act[len(episode["action"]) - self.num_queries + 1 :] = act[-1]
                act = self.preprocess(act, "action")
                dataset_dict["action"].append(jnp.array(act))
            else:
                dataset_dict["action"].append(
                    jnp.array(self.preprocess(episode["action"][:, None, :], "action"))
                )

        for key in dataset_dict.keys():
            dataset_dict[key] = jnp.concatenate(dataset_dict[key], axis=0)
        self.dataset_dict = dict(dataset_dict)

    def preprocess(self, x, key):
        if key.startswith("pixels"):
            return x / 255.0
        return (x - self.stats["min"][key]) / (
            self.stats["max"][key] - self.stats["min"][key]
        )

    def postprocess(self, x, key):
        if key.startswith("pixels"):
            return x * 255.0
        return (
            x * (self.stats["max"][key] - self.stats["min"][key])
            + self.stats["min"][key]
        )
