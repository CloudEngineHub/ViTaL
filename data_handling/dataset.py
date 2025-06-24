from typing import Callable, Dict, Iterable, Optional, Union
from jax import numpy as jnp
import jax
import numpy as np
from pathlib import Path
from PIL import Image
from type_utils import DataType

DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, jnp.ndarray) or isinstance(v, np.ndarray):
            print(k, v.shape)
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError("Unsupported type in Dataset _check_lengths.")
    return dataset_len


def _sample(
    dataset_dict: Union[DatasetDict, jnp.ndarray], idx: np.ndarray
) -> DatasetDict:
    if isinstance(dataset_dict, jnp.ndarray):
        return dataset_dict[idx]
    elif isinstance(dataset_dict, dict):
        return {k: _sample(v, idx) for k, v in dataset_dict.items()}
    else:
        raise ValueError(f"Invalid type for dataset_dict: {type(dataset_dict)}")


def composite_fn(img, mask, rng, bg_images, bg_aug_threshold):
    p = jax.random.uniform(rng)

    def do_composite(_):
        bg_idx = jax.random.randint(rng, (), 0, bg_images.shape[0])
        bg = bg_images[bg_idx]
        return img * mask + bg * (1.0 - mask)

    return jax.lax.cond(p < bg_aug_threshold, do_composite, lambda _: img, operand=None)


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


class Dataset:
    def __init__(
        self,
        dataset_dict: DatasetDict,
        image_augs=None,
        bg_augs=None,
        bg_aug_threshold=1.0,
        crop_view=None,
    ):
        self.dataset_dict = dataset_dict
        self.image_augs = image_augs
        self.dataset_len = _check_lengths(dataset_dict)
        self.bg_augs = bg_augs
        self.crop_view = crop_view

        if bg_augs:
            self.bg_aug_threshold = 1.0  # Always composite when bg_augs is enabled.
            bg_folder = Path("/mnt/robotlab/zifan/visk_rl_jax/augmented_backgrounds")
            # Store file paths instead of loading images right away.
            self.bg_files = sorted(bg_folder.glob("augmented_background_*.png"))

        if crop_view:
            H = W = 128
            crop_h = int(0.8 * H)
            crop_w = int(0.5 * W)
            start_r = H - crop_h
            start_c = (W - crop_w) // 2 + 10
            self._crop_params = (start_r, start_c, crop_h, crop_w)

    def __len__(self) -> int:
        return self.dataset_len

    def sample(
        self,
        rng: jax.Array,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        idx: Optional[np.ndarray] = None,
    ) -> Dict:
        if idx is None:
            rng, batch_rng = jax.random.split(rng)
            idx = jax.random.randint(batch_rng, (batch_size,), 0, self.dataset_len)
        if keys is None:
            keys = self.dataset_dict.keys()

        batch = {}

        for k in keys:
            if k.startswith("pixels"):
                raw_images = self.dataset_dict[k][idx]
                if k == "pixels52" and self.bg_augs:
                    masks = self.dataset_dict["bitwise_masks"][idx]
                    rng, comp_rng = jax.random.split(rng)
                    rngs = jax.random.split(comp_rng, batch_size)
                    files = (
                        np.random.choice(self.bg_files, 36, replace=False)
                        if len(self.bg_files) > 36
                        else self.bg_files
                    )
                    if self.crop_view:
                        sr, sc, ch, cw = self._crop_params
                        bg_list = []
                        for f in files:
                            full = np.array(Image.open(f), dtype=np.float32) / 255.0
                            patch = full[sr : sr + ch, sc : sc + cw, :]
                            bg_list.append(patch)
                    else:
                        bg_list = [
                            np.array(Image.open(f), dtype=np.float32) / 255.0
                            for f in files
                        ]
                    bg_images = jnp.array(np.stack(bg_list, axis=0))
                    if self.crop_view:

                        def patch_pipeline(img, mask, key):
                            key, subkey = jax.random.split(key)
                            p = jax.random.uniform(subkey)

                            def do_patch(k2):
                                idx2 = jax.random.randint(k2, (), 0, bg_images.shape[0])
                                bgp = bg_images[idx2]
                                imgp = img[sr : sr + ch, sc : sc + cw, :]
                                mskp = mask[sr : sr + ch, sc : sc + cw, :]
                                comp = imgp * mskp + bgp * (1.0 - mskp)
                                out = jnp.zeros_like(img)
                                return out.at[sr : sr + ch, sc : sc + cw, :].set(comp)

                            comp_img = jax.lax.cond(
                                p < self.bg_aug_threshold,
                                do_patch,
                                lambda _: img,
                                operand=key,
                            )

                            key, subkey = jax.random.split(key)
                            return crop_fn(self.image_augs(comp_img, subkey))

                        batch[k] = jax.vmap(patch_pipeline)(raw_images, masks, rngs)
                    else:

                        def full_pipeline(img, mask, key):
                            comp = composite_fn(
                                img, mask, key, bg_images, self.bg_aug_threshold
                            )
                            key, subkey = jax.random.split(key)
                            return self.image_augs(comp, subkey)

                        batch[k] = jax.vmap(full_pipeline)(raw_images, masks, rngs)
                else:
                    rng, key_rng = jax.random.split(rng)
                    key_rngs = jax.random.split(key_rng, batch_size)
                    batch[k] = jax.vmap(self.image_augs)(raw_images, key_rngs)
            else:
                batch[k] = self.dataset_dict[k][idx]
        return batch
