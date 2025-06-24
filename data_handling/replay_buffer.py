import numpy as np
from data_handling.dataset import Dataset, DatasetDict


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError(f"Invalid type for dataset_dict: {type(dataset_dict)}")


def ReplayBufferFactory(BaseDataset: Dataset, capacity: int, *args, **kwargs):
    class ReplayBuffer(BaseDataset):
        def __init__(self, init_capacity: int, *init_args, **init_kwargs):
            super().__init__(*init_args, **init_kwargs, capacity=init_capacity)

            self._size = 0
            self._capacity = init_capacity
            self._insert_index = 0

        def insert(self, data_dict: DatasetDict, preprocess: bool = False):
            if preprocess:
                data_dict = self.preprocess_dict(data_dict)
            _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

            self._size = min(self._size + 1, self._capacity)
            self._insert_index = (self._insert_index + 1) % self._capacity

        def preprocess_dict(self, data_dict: DatasetDict):
            raise NotImplementedError

    return ReplayBuffer(capacity, *args, **kwargs)
