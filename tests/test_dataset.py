import sys

sys.path.append("./")

import numpy as np
import pickle as pkl
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List
from utils import ObsType  # Ensure ObsType is available
from data_handling.xarm_dataset import XarmDataset  # Import your class


def create_dummy_data(num_episodes=5, num_samples_per_episode=100):
    """Creates a small, multi-episode dataset for testing."""
    dummy_data = {
        "observations": [],
        "max_cartesian": np.array([1.0] * 7),
        "min_cartesian": np.array([-1.0] * 7),
        "max_gripper": 1.0,
        "min_gripper": 0.0,
    }

    for _ in range(num_episodes):
        episode = {
            "cartesian_states": np.random.rand(num_samples_per_episode, 6),
            "gripper_states": np.random.rand(num_samples_per_episode),
            "sensor_states": np.random.rand(num_samples_per_episode, 30),
            "image": np.random.rand(num_samples_per_episode, 64, 64, 3),
        }
        dummy_data["observations"].append(episode)

    return dummy_data


def test_xarm_dataset():
    """Tests the XarmDataset class functionality."""

    # ✅ Step 1: Create Dummy Dataset and Write to Temporary File
    dummy_data = create_dummy_data(num_episodes=5, num_samples_per_episode=100)

    with tempfile.TemporaryDirectory() as tmpdirname:
        dataset_path = Path(tmpdirname) / "test_dataset.pkl"
        with open(dataset_path, "wb") as f:
            pkl.dump(dummy_data, f)

        # ✅ Step 2: Initialize Dataset
        dataset = XarmDataset(
            path=str(dataset_path),
            obs_type=ObsType.PIXELS,
            temporal_agg=False,
            num_queries=5,
            img_size=64,
            pixel_keys=["image"],
            aux_keys=["sensor0", "sensor1"],
            relative_actions=True,
        )

        # ✅ Step 3: Verify Dataset Properties
        assert dataset._num_samples > 0, "Dataset should contain samples"
        assert "action" in dataset.dataset_dict, "Dataset should contain actions"
        assert (
            "image" in dataset.dataset_dict
        ), "Dataset should contain image observations"
        assert dataset.dataset_dict["image"].shape[-3:] == (
            64,
            64,
            3,
        ), "Image shape should be (64, 64, 3)"

        # ✅ Step 4: Sample a Batch and Check Output
        sample_batch = dataset.sample(batch_size=5)
        assert "action" in sample_batch, "Sampled batch should contain 'action'"
        assert sample_batch["action"].shape[0] == 5, "Batch size should be 5"

        # ✅ Step 5: Test Preprocessing Function
        test_action = sample_batch["action"][0]
        preprocessed_action = dataset.preprocess("action", test_action)
        assert np.all(preprocessed_action >= -1) and np.all(
            np.abs(preprocessed_action) <= 1
        ), "Preprocessed action should be normalized between -1 and 1"

        print("✅ XarmDataset Test Passed!")


# Run the test
if __name__ == "__main__":
    test_xarm_dataset()
