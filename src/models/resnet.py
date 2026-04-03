"""Flax ResNet-18 and ResNet-50 for image classification.

Follows the project convention:
    __call__(self, x, train=True) -> (logits, {"z": bottleneck})
where x is NHWC and z is the global-average-pooled representation.

BatchNorm layers require mutable='batch_stats' during training and stored
batch_stats during eval. See BatchNormTrainState in train_state.py.
"""

import jax.numpy as jnp
from flax import linen as nn


class BasicBlock(nn.Module):
    """ResNet basic block (two 3x3 convolutions)."""
    channels: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        y = nn.Conv(self.channels, (3, 3), strides=(self.stride, self.stride),
                     padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)

        if residual.shape[-1] != self.channels or self.stride > 1:
            residual = nn.Conv(self.channels, (1, 1),
                               strides=(self.stride, self.stride),
                               use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        return nn.relu(y + residual)


class BottleneckBlock(nn.Module):
    """ResNet bottleneck block (1x1 -> 3x3 -> 1x1, expansion=4)."""
    channels: int   # narrow (mid) channel count; output = channels * 4
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        out_channels = self.channels * 4
        residual = x

        y = nn.Conv(self.channels, (1, 1), use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)

        y = nn.Conv(self.channels, (3, 3), strides=(self.stride, self.stride),
                     padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)

        y = nn.Conv(out_channels, (1, 1), use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)

        if residual.shape[-1] != out_channels or self.stride > 1:
            residual = nn.Conv(out_channels, (1, 1),
                               strides=(self.stride, self.stride),
                               use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        return nn.relu(y + residual)


class ResNet(nn.Module):
    """Generic ResNet with configurable depth and stem.

    Args:
        num_classes: number of output logits.
        stage_sizes: tuple of block counts per stage, e.g. (2,2,2,2).
        stage_channels: tuple of channel widths per stage, e.g. (64,128,256,512).
        bottleneck: if True use BottleneckBlock (ResNet-50+), else BasicBlock.
        cifar_mode: if True use 3x3 stem without maxpool (for 32x32 inputs).
    """
    num_classes: int
    stage_sizes: tuple = (2, 2, 2, 2)
    stage_channels: tuple = (64, 128, 256, 512)
    bottleneck: bool = False
    cifar_mode: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Stem
        if self.cifar_mode:
            x = nn.Conv(64, (3, 3), padding='SAME', use_bias=False)(x)
        else:
            x = nn.Conv(64, (7, 7), strides=(2, 2),
                         padding=((3, 3), (3, 3)), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        if not self.cifar_mode:
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        # Residual stages
        BlockCls = BottleneckBlock if self.bottleneck else BasicBlock
        for i, (n_blocks, ch) in enumerate(
            zip(self.stage_sizes, self.stage_channels)
        ):
            for j in range(n_blocks):
                stride = 2 if i > 0 and j == 0 else 1
                x = BlockCls(channels=ch, stride=stride,
                             name=f"block_{i}_{j}")(x, train=train)

        # Global average pool -> bottleneck -> logits
        z = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(self.num_classes)(z)
        return logits, {"z": z}


def ResNet18(num_classes: int, cifar_mode: bool = False) -> ResNet:
    """ResNet-18 (BasicBlock, [2,2,2,2])."""
    return ResNet(
        num_classes=num_classes,
        stage_sizes=(2, 2, 2, 2),
        stage_channels=(64, 128, 256, 512),
        bottleneck=False,
        cifar_mode=cifar_mode,
    )


def ResNet50(num_classes: int) -> ResNet:
    """ResNet-50 (BottleneckBlock, [3,4,6,3]) for 224x224 inputs."""
    return ResNet(
        num_classes=num_classes,
        stage_sizes=(3, 4, 6, 3),
        stage_channels=(64, 128, 256, 512),
        bottleneck=True,
    )
