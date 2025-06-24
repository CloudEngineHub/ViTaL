import sys

sys.path.append("./")
import numpy as np
import torch

import jax
import jax.numpy as jnp
from agent.networks.resnet_encoder import SpatialSoftmax


class TorchSpatialSoftmax(torch.nn.Module):
    """
    The spatial softmax layer (https://rll.berkeley.edu/dsae/dsae.pdf)
    """

    def __init__(self, in_c, in_h, in_w, num_kp=None):
        super().__init__()
        self._spatial_conv = torch.nn.Conv2d(in_c, num_kp, kernel_size=1)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, in_w).float(),
            torch.linspace(-1, 1, in_h).float(),
        )

        pos_x = pos_x.reshape(1, in_w * in_h)
        pos_y = pos_y.reshape(1, in_w * in_h)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

        if num_kp is None:
            self._num_kp = in_c
        else:
            self._num_kp = num_kp

        self._in_c = in_c
        self._in_w = in_w
        self._in_h = in_h

    def forward(self, x):
        assert x.shape[1] == self._in_c
        assert x.shape[2] == self._in_h
        assert x.shape[3] == self._in_w

        h = x
        if self._num_kp != self._in_c:
            h = self._spatial_conv(h)
        h = h.contiguous().view(-1, self._in_h * self._in_w)

        attention = torch.nn.functional.softmax(h, dim=-1)
        keypoint_x = (
            (self.pos_x * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        keypoint_y = (
            (self.pos_y * attention).sum(1, keepdims=True).view(-1, self._num_kp)
        )
        print(self.pos_x[0])
        # print(keypoint_x[0], keypoint_y[0])
        keypoints = torch.cat([keypoint_y, keypoint_x], dim=1)
        return keypoints


def torch_to_jax(torch_tensor):
    return jnp.array(torch_tensor.detach().cpu().numpy())


# ✅ Convert JAX Array to PyTorch Tensor
def jax_to_torch(jax_array):
    return torch.tensor(np.array(jax_array), dtype=torch.float32)


B, C, H, W = 2, 3, 8, 8
torch_input = torch.randn(B, C, H, W)
jax_input = torch_to_jax(torch_input).transpose(0, 2, 3, 1)  # Convert to (B, H, W, C)

torch_model = TorchSpatialSoftmax(C, H, W, C)
flax_model = SpatialSoftmax(num_kp=C)


# ✅ Compute Outputs
torch_output = torch_model(torch_input)
flax_params = flax_model.init(jax.random.PRNGKey(0), jax_input)
flax_output = flax_model.apply(flax_params, jax_input)

# ✅ Convert Flax Output to PyTorch for Comparison
flax_output_torch = jax_to_torch(flax_output)

# ✅ Compare Outputs
max_diff = torch.max(torch.abs(torch_output - flax_output_torch)).item()
print(f"🔥 Max Absolute Difference: {max_diff:.6e}")

assert torch.allclose(
    torch_output, flax_output_torch, atol=1e-5
), "❌ Outputs do not match!"
print("✅ PyTorch and Flax outputs match!")
