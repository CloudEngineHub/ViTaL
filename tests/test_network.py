import sys

sys.path.append("./")

import jax
import jax.numpy as jnp

from agent.networks.mlp import MLP


def test_mlp():
    """Tests the MLP model's forward pass and output shape."""

    # Define dummy input
    key = jax.random.PRNGKey(0)  # Random key for initialization
    batch_size = 4
    repr_dim = 16
    action_dim = 6
    hidden_dim = 32

    # Create an instance of the model
    model = MLP(repr_dim=repr_dim, action_dim=action_dim, hidden_dim=hidden_dim)

    # Initialize parameters
    rng, init_key = jax.random.split(key)
    dummy_input = jnp.ones((batch_size, repr_dim))
    params = model.init(init_key, dummy_input)

    # Perform forward pass
    output = model.apply(params, dummy_input)

    # Assertions
    assert output.shape == (
        batch_size,
        action_dim,
    ), f"Unexpected output shape: {output.shape}"

    print("MLP Test Passed! ✅")


# Run the test
if __name__ == "__main__":
    test_mlp()
