from functools import partial
import jax
import jax.numpy as jnp


# NOTE: FUNCTIONS ASSUME DATA IS IN [0, 1] RANGE
# @partial(jax.jit, static_argnames=("brightness", "contrast", "saturation", "hue"))
def jax_color_jitter(
    image: jax.Array,
    rng: jax.random.key,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float = 0.0,
) -> jax.Array:
    rng_b, rng_c, rng_s, rng = jax.random.split(rng, 4)

    if brightness > 0.0:
        brightness_factor = jax.random.uniform(
            rng_b, (), minval=1.0 - brightness, maxval=1.0 + brightness
        )
        image = image * brightness_factor

    if contrast > 0.0:
        contrast_factor = jax.random.uniform(
            rng_c, (), minval=1.0 - contrast, maxval=1.0 + contrast
        )
        mean = jnp.mean(image, axis=(0, 1), keepdims=True)
        image = (image - mean) * contrast_factor + mean

    if saturation > 0.0:
        grayscale = jnp.mean(image, axis=-1, keepdims=True)
        saturation_factor = jax.random.uniform(
            rng_s, (), minval=1.0 - saturation, maxval=1.0 + saturation
        )
        image = (image - grayscale) * saturation_factor + grayscale

    if hue > 0:
        raise NotImplementedError("Hue not implemented yet")

    return jnp.clip(image, 0.0, 1.0), rng


# @partial(jax.jit, static_argnames=("crop_height", "crop_width", "padding"))
def jax_random_crop(
    image: jax.Array,
    rng: jax.random.key,
    crop_height: int,
    crop_width: int,
    padding: int = 0,
) -> jax.Array:
    if padding > 0:
        image = jnp.pad(
            image,
            ((padding, padding), (padding, padding), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    img_h, img_w = image.shape[:2]

    rng_y, rng_x, rng = jax.random.split(rng, 3)

    max_y = img_h - crop_height
    max_x = img_w - crop_width

    start_x = jax.random.randint(rng_x, (), 0, max_x)
    start_y = jax.random.randint(rng_y, (), 0, max_y)

    # return image[start_y:start_y+crop_height, start_x:start_x+crop_width]
    return jax.lax.dynamic_slice(
        image, (start_y, start_x, 0), (crop_height, crop_width, 3)
    ), rng
