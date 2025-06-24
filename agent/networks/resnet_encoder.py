from typing import Callable, List, Optional
from flax import linen as nn

import jax
from jax import numpy as jnp


class AdaptiveAvgPool2d(nn.Module):
    output_size: tuple  # (H_out, W_out)

    def __call__(self, x):
        """Mimics torch.nn.AdaptiveAvgPool2d in Flax (JAX)."""
        B, H, W, C = x.shape  # Assume (Batch, Height, Width, Channels)
        H_out, W_out = self.output_size

        # ✅ Compute kernel size & stride dynamically
        kernel_size = (H // H_out, W // W_out)
        stride = (H // H_out, W // W_out)

        # ✅ Perform average pooling
        return nn.avg_pool(x, window_shape=kernel_size, strides=stride, padding="VALID")


def conv3x3(out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv:
    """3x3 convolution with padding"""
    return nn.Conv(
        features=out_planes,
        kernel_size=(3, 3),
        strides=stride,
        padding=dilation,
        use_bias=False,
        kernel_dilation=dilation,
        kernel_init=nn.initializers.he_normal(),
    )


def conv1x1(out_planes: int, stride: int = 1) -> nn.Conv:
    """1x1 convolution"""
    return nn.Conv(
        out_planes,
        kernel_size=(1, 1),
        strides=stride,
        use_bias=False,
        kernel_init=nn.initializers.he_normal(),
    )


class BasicBlock(nn.Module):
    planes: int
    expansion: int = 1
    stride: int = 1
    downsample: Optional[nn.Module] = None
    dilation: int = 1

    def setup(self):
        norm_layer = nn.BatchNorm
        if self.dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(self.planes, self.stride)
        self.bn1 = norm_layer()
        self.relu = nn.relu
        self.conv2 = conv3x3(self.planes)
        self.bn2 = norm_layer()
        if self.downsample is not None:
            self.bn3 = norm_layer()

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not training)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn3(identity, use_running_average=not training)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    output_dim: int = 1000
    # Reimplemented from torch. Does not take gropu argument and does not support
    # pretraininged weights.

    def setup(self):
        self._norm_layer = nn.BatchNorm
        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv(
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=3,
            use_bias=False,
            kernel_init=nn.initializers.he_normal(),
        )
        self.bn1 = self._norm_layer()

        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # TODO: Add this avgpool in forward pass
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Dense(self.output_dim)

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes:
            downsample = conv1x1(planes, stride)

        layers = []
        layers.append(
            block(
                planes=planes,
                stride=stride,
                downsample=downsample,
                dilation=previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    planes=planes,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(layers)

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not training)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), stride=(2, 2), padding=(1, 1))
        # self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


GROUP_NORM_LOOKUP = {
    16: 2,  # -> channels per group: 8
    32: 4,  # -> channels per group: 8
    64: 8,  # -> channels per group: 8
    128: 8,  # -> channels per group: 16
    256: 16,  # -> channels per group: 16
    512: 32,  # -> channels per group: 16
    1024: 32,  # -> channels per group: 32
    2048: 32,  # -> channels per group: 64
}


class SpatialSoftmax(nn.Module):
    num_kp: int
    # Tested that this matches torch spatial softmax for H=W

    @nn.compact
    def __call__(self, x: jax.Array):
        B, H, W, C = x.shape
        if self.num_kp != C:
            x = nn.Conv(
                self.num_kp, kernel_size=(1, 1), kernel_init=nn.initializers.he_normal()
            )(x)
        x = nn.softmax(x, axis=(1, 2))
        pos_x, pos_y = jnp.meshgrid(
            jnp.linspace(-1, 1, W), jnp.linspace(-1, 1, H), indexing="xy"
        )

        keypoint_x = jnp.sum(x * pos_x[..., None], axis=(1, 2))
        keypoint_y = jnp.sum(x * pos_y[..., None], axis=(1, 2))
        keypoints = jnp.concatenate([keypoint_x, keypoint_y], axis=-1)
        return keypoints


class SpatialProjection(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x: jax.Array):
        x = SpatialSoftmax(self.output_dim)(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class ResNet18Encoder(ResNet18):
    def setup(self):
        self._norm_layer = nn.GroupNorm
        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv(
            64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=3,
            use_bias=False,
            kernel_init=nn.initializers.he_normal(),
        )
        self.bn1 = nn.GroupNorm(num_groups=GROUP_NORM_LOOKUP[64])

        self.layer1 = self._make_layer(BasicBlockGroupNorm, 64, 2)
        self.layer2 = self._make_layer(BasicBlockGroupNorm, 128, 2, stride=2)

        self.projection_layer = SpatialProjection(self.output_dim)

    def __call__(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        # self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.projection_layer(x)
        return x


class BasicBlockGroupNorm(nn.Module):
    planes: int
    expansion: int = 1
    stride: int = 1
    downsample: Optional[nn.Module] = None
    dilation: int = 1

    def setup(self):
        norm_layer = nn.GroupNorm
        if self.dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(self.planes, self.stride)
        self.bn1 = norm_layer(GROUP_NORM_LOOKUP[self.planes])
        self.relu = nn.relu
        self.conv2 = conv3x3(self.planes)
        self.bn2 = norm_layer(GROUP_NORM_LOOKUP[self.planes])
        if self.downsample is not None:
            self.bn3 = norm_layer(GROUP_NORM_LOOKUP[self.downsample.features])

    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn3(identity)

        out += identity
        out = self.relu(out)

        return out
