"""JIT-compatible image augmentations for JAX (NHWC format).

All functions operate on batched images with shape (B, H, W, C).
"""

import jax
import jax.numpy as jnp

# ImageNet statistics
IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406])
IMAGENET_STD = jnp.array([0.229, 0.224, 0.225])

# CIFAR-10 statistics
CIFAR10_MEAN = jnp.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = jnp.array([0.2023, 0.1994, 0.2010])


def normalize(images, mean, std):
    """Channel-wise normalize: (images - mean) / std."""
    return (images - mean) / std


def normalize_imagenet(images):
    return normalize(images, IMAGENET_MEAN, IMAGENET_STD)


def normalize_cifar10(images):
    return normalize(images, CIFAR10_MEAN, CIFAR10_STD)


def center_crop(images, crop_h, crop_w):
    """Center-crop a batch of images.

    Args:
        images: (B, H, W, C) array.
        crop_h, crop_w: target spatial dimensions.
    """
    h, w = images.shape[1], images.shape[2]
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return jax.lax.dynamic_slice_in_dim(
        jax.lax.dynamic_slice_in_dim(images, top, crop_h, axis=1),
        left, crop_w, axis=2,
    )


def _random_crop_single(rng, image, crop_h, crop_w):
    """Random crop a single image (H, W, C)."""
    h, w = image.shape[0], image.shape[1]
    rng_h, rng_w = jax.random.split(rng)
    top = jax.random.randint(rng_h, (), 0, h - crop_h + 1)
    left = jax.random.randint(rng_w, (), 0, w - crop_w + 1)
    return jax.lax.dynamic_slice(image, (top, left, 0), (crop_h, crop_w, image.shape[2]))


def random_crop(rng, images, crop_h, crop_w):
    """Random crop each image in a batch independently.

    Args:
        rng: PRNG key.
        images: (B, H, W, C) array where H >= crop_h and W >= crop_w.
        crop_h, crop_w: target spatial dimensions.
    """
    rngs = jax.random.split(rng, images.shape[0])
    return jax.vmap(lambda r, x: _random_crop_single(r, x, crop_h, crop_w))(rngs, images)


def random_hflip(rng, images):
    """Randomly flip each image horizontally with probability 0.5.

    Args:
        rng: PRNG key.
        images: (B, H, W, C) array.
    """
    flip = jax.random.bernoulli(rng, 0.5, (images.shape[0],))
    return jnp.where(flip[:, None, None, None], images[:, :, ::-1, :], images)


def resize(images, size):
    """Resize a batch to (size, size).

    Args:
        images: (B, H, W, C) array.
        size: target spatial dimension (square).
    """
    b, _, _, c = images.shape
    return jax.image.resize(images, (b, size, size, c), method='bilinear')


# Dataset-specific composed pipelines

def waterbirds_train_augment(rng, images):
    """Train augmentation for Waterbirds: random crop 224, hflip, imagenet norm.

    Expects images stored at 256x256.
    """
    rng1, rng2 = jax.random.split(rng)
    images = random_crop(rng1, images, 224, 224)
    images = random_hflip(rng2, images)
    return normalize_imagenet(images)


def waterbirds_eval_transform(images):
    """Eval transform for Waterbirds: center crop 224, imagenet norm.

    Expects images stored at 256x256.
    """
    images = center_crop(images, 224, 224)
    return normalize_imagenet(images)


def celeba_train_augment(rng, images):
    """Train augmentation for CelebA: random crop 224, hflip, imagenet norm.

    Expects images stored at 256x256.
    """
    rng1, rng2 = jax.random.split(rng)
    images = random_crop(rng1, images, 224, 224)
    images = random_hflip(rng2, images)
    return normalize_imagenet(images)


def celeba_eval_transform(images):
    """Eval transform for CelebA: center crop 178, resize 224, imagenet norm.

    Expects images stored at 256x256 (aligned+cropped CelebA faces).
    """
    images = center_crop(images, 178, 178)
    images = resize(images, 224)
    return normalize_imagenet(images)


def ccifar10_train_augment(rng, images):
    """Train augmentation for cCIFAR-10: pad 4 + random crop 32x32 + hflip + normalize.

    Standard CIFAR-10 augmentation pipeline (He et al. 2016).
    Expects images in NHWC float32 [0, 1].
    """
    rng1, rng2 = jax.random.split(rng)
    images = jnp.pad(images, ((0, 0), (4, 4), (4, 4), (0, 0)))  # 32 -> 40
    images = random_crop(rng1, images, 32, 32)
    images = random_hflip(rng2, images)
    return normalize_cifar10(images)


def ccifar10_eval_transform(images):
    """Eval transform for cCIFAR-10: CIFAR-10 normalize only."""
    return normalize_cifar10(images)
