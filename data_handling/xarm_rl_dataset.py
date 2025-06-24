from typing import Dict, List
import dm_env
import jax
from jax import numpy as jnp
import numpy as np
from pathlib import Path
from data_handling.dataset import Dataset, DatasetDict
from img_utils import jax_color_jitter, jax_random_crop
import ipdb
import zmq
import torch
import cv2
from PIL import Image


@jax.jit
def composite_fn(img, mask, rng, bg_images, bg_aug_threshold):
    mask = jnp.reshape(mask, (mask.shape[0], mask.shape[1], 1))

    p = jax.random.uniform(rng)

    def do_composite(_):
        bg_idx = jax.random.randint(rng, (), 0, bg_images.shape[0])
        bg = bg_images[bg_idx]
        return img * mask + bg * (1.0 - mask)

    return jax.lax.cond(p < bg_aug_threshold, do_composite, lambda _: img, operand=None)


@jax.jit
def crop_fn(img):
    crop_height = int(0.8 * 128)
    crop_width = int(0.5 * 128)
    start_row = 128 - crop_height
    start_col = (128 - crop_width) // 2 + 10

    black = jnp.zeros_like(img)
    black = jax.lax.dynamic_update_slice(
        black,
        img[start_row : start_row + crop_height, start_col : start_col + crop_width, :],
        (start_row, start_col, 0),
    )
    return black


def compute_quadrant_centers(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    mx, my = x0 + w // 2, y0 + h // 2

    quads = [
        (x0, y0, mx - 1, my - 1),  # TL
        (mx, y0, x1, my - 1),  # TR
        (x0, my, mx - 1, y1),  # BL
        (mx, my, x1, y1),  # BR
    ]
    centers = []
    for xa, ya, xb, yb in quads:
        sub = mask[ya : yb + 1, xa : xb + 1]
        ys_q, xs_q = np.where(sub > 0)
        if len(xs_q):
            cx = int(xs_q.mean()) + xa
            cy = int(ys_q.mean()) + ya
            centers.append((cx, cy))
    return centers


def compute_mask_center(mask):
    ys, xs = np.where(mask > 0)
    return int(xs.mean()), int(ys.mean())


class XarmRLDataset(Dataset):
    def __init__(
        self,
        img_size: int,
        capacity: int,
        bg_augs: bool,
        pixel_keys: List[str],
        aux_keys: List[str],
        obs_spec: Dict[str, dm_env.specs.Array],
        action_spec: dm_env.specs.Array,
        offset_action_scale: float = 5.0,
        task_name: str = "key_insertion",
        *args,
        **kwargs,
    ):
        if bg_augs:
            self.bg_aug_threshold = 1.0
            bg_folder = Path("/path_to_data/augmented_backgrounds")
            self.bg_files = sorted(bg_folder.glob("augmented_background_*.png"))
            bg_list = []
            for f in self.bg_files:
                img = np.array(Image.open(f)).astype(np.float32) / 255.0
                bg_list.append(img)

            self.bg_images = jnp.array(np.stack(bg_list, axis=0))
            context = zmq.Context()
            self.dift_socket = context.socket(zmq.REQ)
            self.dift_socket.connect("tcp://localhost:5556")
            self.sam2_socket = context.socket(zmq.REQ)
            self.sam2_socket.connect("tcp://localhost:5557")
            self.xmem_socket = context.socket(zmq.REQ)
            self.xmem_socket.connect("tcp://localhost:5558")

            base_dir = Path(
                f"/path_to_data/anchor_data/{task_name}/base/"
            )

            self.base_dir = str(base_dir)
            self.original_feature = torch.load(
                base_dir / "dift_feature_map.pt"
            ).squeeze(0)
            self.target_mask = cv2.imread(str(base_dir / "original_mask_target.png"), 0)
            self.target_mask = cv2.resize(
                self.target_mask, (768, 768), interpolation=cv2.INTER_LINEAR
            )
            self.object_mask = cv2.imread(str(base_dir / "original_mask_object.png"), 0)
            self.object_mask = cv2.resize(
                self.object_mask, (768, 768), interpolation=cv2.INTER_LINEAR
            )
            self.gripper_mask = cv2.imread(str(base_dir / "gripper_mask.png"), 0)

            self.gripper_mask = cv2.resize(
                self.gripper_mask, (768, 768), interpolation=cv2.INTER_LINEAR
            )
            self.prev_mask = None
            self.bg_augs = bg_augs
            self.mask_count = 0

        dataset_dict = {
            "obs": {
                **{
                    key: np.empty((capacity, *obs_spec[key].shape), dtype=np.float32)
                    for key in pixel_keys + aux_keys
                },
                "bitwise_mask": np.empty((capacity, 128, 128), dtype=np.uint8),
            },
            "obs_next": {
                **{
                    key: np.empty((capacity, *obs_spec[key].shape), dtype=np.float32)
                    for key in pixel_keys + aux_keys
                },
                "bitwise_mask": np.empty((capacity, 128, 128), dtype=np.uint8),
            },
            "reward": np.empty((capacity,), dtype=np.float32),
            "action": np.empty(
                (capacity, action_spec["action"].shape[0]), dtype=np.float32
            ),
            "action_base": np.empty(
                (capacity, action_spec["action_base"].shape[0]), dtype=np.float32
            ),
            "action_base_next": np.empty(
                (capacity, action_spec["action_base"].shape[0]), dtype=np.float32
            ),
            "discount": np.empty((capacity,), dtype=np.float32),
        }

        self.stats = {
            "min": {},
            "max": {},
        }
        self.stats["min"]["action"] = 0.0
        self.stats["max"]["action"] = offset_action_scale

        def augment_image(image: jax.Array, rng: jax.Array) -> jax.Array:
            # image = image / 255.0
            # image, rng = jax.jit(
            #     jax_random_crop,
            #     static_argnames=("crop_height", "crop_width", "padding"),
            # )(image, rng, img_size, img_size, padding=4)
            image, rng = jax.jit(
                jax_color_jitter,
                static_argnames=("brightness", "contrast", "saturation"),
            )(image, rng, 0.3, 0.3, 0.2)
            return image

        super().__init__(dataset_dict, augment_image)

    def update_stats(self, bc_stats_dict):
        min_stats = {k: v for k, v in bc_stats_dict["min"].items() if not k == "action"}
        max_stats = {k: v for k, v in bc_stats_dict["max"].items() if not k == "action"}
        min_stats["action_base"] = bc_stats_dict["min"]["action"]
        max_stats["action_base"] = bc_stats_dict["max"]["action"]
        self.stats["min"].update(min_stats)
        self.stats["max"].update(max_stats)

    def preprocess_dict(self, data_dict: DatasetDict):
        processed_dict = {}
        for k, v in data_dict.items():
            if isinstance(v, dict):
                processed_dict[k] = self.preprocess_dict(v)
            else:
                processed_dict[k] = self.preprocess(v, k)
        return processed_dict

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

    def track_mask(self, frame, prev_mask, first_frame=False):
        _, frame_encoded = cv2.imencode(".jpg", frame)
        self.xmem_socket.send_multipart(
            [
                frame_encoded.tobytes(),
                prev_mask.tobytes(),
                str(int(first_frame)).encode(),
            ]
        )
        mask = self.xmem_socket.recv()
        return np.frombuffer(mask, dtype=np.uint8).reshape(frame.shape[:2])

    def get_features(self, frame):
        _, frame_encoded = cv2.imencode(".jpg", frame)
        self.dift_socket.send(frame_encoded.tobytes())
        features = self.dift_socket.recv()
        return np.frombuffer(features, dtype=np.float32).reshape((1280, 48, 48))

    def get_mask(self, frame, points, labels):
        _, frame_encoded = cv2.imencode(".jpg", frame)
        coords_arr = np.array(points, dtype=np.int32)  # shape: (N, 2)
        labels_arr = np.array(labels, dtype=np.int32)  # shape: (N,)

        # send three frames: image, coords, labels
        self.sam2_socket.send_multipart(
            [frame_encoded.tobytes(), coords_arr.tobytes(), labels_arr.tobytes()]
        )

        mask_bytes = self.sam2_socket.recv()
        return np.frombuffer(mask_bytes, dtype=np.uint8).reshape((768, 768))

    def insert_from_timestep(
        self, time_step, bg_augs, preprocess: bool = False, action_base_t1=None
    ):
        if bg_augs:
            frame = cv2.resize(
                time_step.observation["pixels52"],
                (768, 768),
                interpolation=cv2.INTER_LINEAR,
            )
            if self.prev_mask is None:
                print("Getting Feature for the first mask")
                features = self.get_features(frame)
                self.mask_count = 1

                global_pt = compute_mask_center(self.target_mask)
                quad_pts = compute_quadrant_centers(self.target_mask)
                original_pts = [global_pt] + quad_pts

                C, H, W = features.shape
                flat_curr = features.reshape(C, -1).copy()
                flat_curr /= np.linalg.norm(flat_curr, axis=0, keepdims=True) + 1e-8

                scale = 768 // 48
                mapped = []
                for mx, my in original_pts:
                    fy, fx = my // scale, mx // scale
                    orig_vec = self.original_feature[:, fy, fx]
                    orig_vec = orig_vec / (np.linalg.norm(orig_vec) + 1e-8)
                    cos_sim = orig_vec[:, None] * flat_curr
                    cos_sim = cos_sim.sum(axis=0)
                    idx = int(cos_sim.argmax())
                    yy, xx = divmod(idx, W)  # W==48
                    mapped.append([xx * scale, yy * scale])

                pts = np.array(mapped)
                lbls = np.ones(len(mapped), dtype=int)
                current_mask = self.get_mask(frame, pts, lbls)
                self.prev_mask = current_mask
                self.track_mask(frame, self.prev_mask, first_frame=True)

            self.mask_count += 1
            current_mask = self.track_mask(frame, self.prev_mask, first_frame=False)
            self.prev_mask = current_mask
            combined_mask = np.maximum(
                np.maximum(current_mask, self.gripper_mask), self.object_mask
            )
            binary_mask = np.where(combined_mask > 0, 1, 0).astype(np.uint8)
            resized_mask = cv2.resize(
                binary_mask, (128, 128), interpolation=cv2.INTER_NEAREST
            )
            time_step.observation["bitwise_mask"] = resized_mask
            from pathlib import Path

            sanity_dir = Path("/mnt/robotlab/zifan/visk_rl_jax/sanity_check_rl")
            sanity_dir.mkdir(exist_ok=True, parents=True)
            mask_file = sanity_dir / f"generated_mask_{self.mask_count}.png"
            cv2.imwrite(str(mask_file), resized_mask * 255)

        if preprocess:
            for k in self.dataset_dict["obs"].keys():
                if not k.startswith("bit"):
                    time_step.observation[k] = self.preprocess(
                        time_step.observation[k], k
                    )
            for k in ["action", "action_base"]:
                time_step = time_step._replace(**{k: self.preprocess(time_step[k], k)})

        for obs_key in self.dataset_dict["obs"].keys():
            if not time_step.last():
                if not obs_key.startswith("bit"):
                    self.dataset_dict["obs"][obs_key][self._insert_index] = (
                        time_step.observation[obs_key]
                    )
                if bg_augs:
                    self.dataset_dict["obs"]["bitwise_mask"][self._insert_index] = (
                        resized_mask
                    )

            if not time_step.first():
                if not obs_key.startswith("bit"):
                    self.dataset_dict["obs_next"][obs_key][self._insert_index - 1] = (
                        time_step.observation[obs_key]
                    )
                if bg_augs:
                    self.dataset_dict["obs_next"]["bitwise_mask"][
                        self._insert_index - 1
                    ] = resized_mask

        if time_step.last():
            self.prev_mask = None
        if not time_step.last():
            self._size = min(self._size + 1, self._capacity)
            self._insert_index = (self._insert_index + 1) % self._capacity
        # Note that this might be a bit broken in a sparse reward setting because we don't quite store the last observation.
        if not time_step.first():
            for k in ["reward", "action", "action_base", "discount"]:
                self.dataset_dict[k][self._insert_index - 1] = time_step[k]
            self.dataset_dict["action_base_next"][self._insert_index - 1] = time_step[
                "action_base"
            ]

    def save_data(self, dir):
        save_dir = Path(dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        n = self._size

        save_dict = {}
        for key, val in self.dataset_dict.items():
            if isinstance(val, dict):
                for subkey, arr in val.items():
                    save_dict[f"{key}.{subkey}"] = arr[:n]
            else:
                save_dict[key] = val[:n]

        save_dict["_size"] = np.array(self._size, dtype=np.int64)
        save_dict["_insert_index"] = np.array(self._insert_index, dtype=np.int64)

        np.savez_compressed(save_dir / "dataset.npz", **save_dict)

    def load_data(self, data_dir):
        load_path = Path(data_dir) / "dataset.npz"
        data = np.load(load_path)

        loaded_size = int(data["_size"])
        for key, val in self.dataset_dict.items():
            if isinstance(val, dict):
                for subkey in val:
                    arr = data[f"{key}.{subkey}"]
                    self.dataset_dict[key][subkey][: arr.shape[0]] = arr
            else:
                arr = data[key]
                self.dataset_dict[key][: arr.shape[0]] = arr

        self._size = loaded_size
        self._insert_index = int(data["_insert_index"])

    def __len__(self) -> int:
        return self._size

    def sample(
        self,
        rng: jax.Array,
        batch_size: int,
        bg_augs: bool,
    ) -> Dict:
        rng, batch_rng = jax.random.split(rng)
        idx = jax.random.randint(batch_rng, (batch_size,), 0, self._size)
        keys = self.dataset_dict.keys()
        batch = {"obs": {}, "obs_next": {}}
        for k in keys:
            if k.startswith("obs"):
                for obs_k in self.dataset_dict[k]:
                    if obs_k == "bitwise_mask":
                        continue
                    if obs_k.startswith("pixels"):
                        masks = self.dataset_dict[k]["bitwise_mask"][idx]
                        rng, key_rng = jax.random.split(rng)
                        key_rngs = jax.random.split(key_rng, batch_size)
                        aug_images = jax.vmap(self.image_augs)(
                            self.dataset_dict[k][obs_k][idx], key_rngs
                        )

                        if bg_augs:
                            num_bg_files = self.bg_images.shape[0]

                            selected_indices = np.random.choice(
                                num_bg_files, size=88, replace=False
                            )

                            bg_images = self.bg_images[selected_indices]

                            def aug_then_composite(img, m, r):
                                comp_img = composite_fn(
                                    img,
                                    m,
                                    r,
                                    bg_images,
                                    bg_aug_threshold=1.0,
                                )
                                return crop_fn(
                                    comp_img
                                )  # background augmentation first and then cropping

                            batch[k][obs_k] = jax.vmap(aug_then_composite)(
                                aug_images, masks, key_rngs
                            )

                        else:
                            batch[k][obs_k] = aug_images

                    else:
                        batch[k][obs_k] = self.dataset_dict[k][obs_k][idx]
            else:
                batch[k] = self.dataset_dict[k][idx]
        return batch
