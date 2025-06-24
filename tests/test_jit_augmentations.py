import sys

sys.path.append("./")

import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from img_utils import jax_color_jitter, jax_random_crop


def augment_image_jitted(image: jax.Array, rng: jax.Array) -> jax.Array:
    image, rng = jax.jit(
        jax_random_crop, static_argnames=("crop_height", "crop_width", "padding")
    )(image, rng, 128, 128, padding=4)
    image, rng = jax.jit(
        jax_color_jitter, static_argnames=("brightness", "contrast", "saturation")
    )(image, rng, 0.3, 0.3, 0.2)
    return image


def augment_image_non_jitted(image: jax.Array, rng: jax.Array) -> jax.Array:
    image, rng = jax_random_crop(image, rng, 128, 128, padding=4)
    image, rng = jax_color_jitter(image, rng, 0.3, 0.3, 0.2)
    return image


def test_augmentations():
    """Test image augmentations with and without JIT, and compare results."""

    # ✅ Generate synthetic test images (Batch of 4)
    rng = jax.random.key(42)
    batch_size = 4
    image_size = (128, 128, 3)

    for i in range(20):
        batch_rng, rng = jax.random.split(rng)
        batch_images = jax.random.uniform(
            batch_rng, shape=(batch_size, *image_size), minval=0, maxval=1
        )

        rng, *vmap_rngs = jax.random.split(rng, batch_size + 1)

        # ✅ Test JIT Augmentation
        start_time = time.time()
        aug_images_jit = jax.vmap(augment_image_jitted)(
            batch_images, jnp.array(vmap_rngs)
        )
        jit_time = time.time() - start_time
        print(f"⚡ JIT Execution Time: {jit_time:.6f} seconds")

        # ✅ Test Non-JIT Augmentation
        start_time = time.time()
        aug_images_non_jit = jax.vmap(augment_image_non_jitted)(
            batch_images, jnp.array(vmap_rngs)
        )
        non_jit_time = time.time() - start_time
        print(f"🚀 Non-JIT Execution Time: {non_jit_time:.6f} seconds")

        # ✅ Check if both methods produce similar results
        assert jnp.allclose(
            aug_images_non_jit, aug_images_jit, atol=1e-5
        ), "❌ JIT and Non-JIT outputs do not match!"
        print("✅ JIT and Non-JIT outputs match!")

    # ✅ Visualize Original & Augmented Images
    fig, axs = plt.subplots(3, batch_size, figsize=(batch_size * 3, 9))

    for i in range(batch_size):
        axs[0, i].imshow(np.array(batch_images[i]))  # Original
        axs[0, i].set_title("Original")
        axs[0, i].axis("off")

        axs[1, i].imshow(np.array(aug_images_non_jit[i]))  # Non-JIT Augmented
        axs[1, i].set_title("Non-JIT Augmented")
        axs[1, i].axis("off")

        axs[2, i].imshow(np.array(aug_images_jit[i]))  # JIT Augmented
        axs[2, i].set_title("JIT Augmented")
        axs[2, i].axis("off")

    plt.tight_layout()
    plt.savefig("augmentation_comparison.png")


# ✅ Run the test function
test_augmentations()
