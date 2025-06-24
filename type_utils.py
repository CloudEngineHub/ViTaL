from typing import Dict, Union
from jax import numpy as jnp
import numpy as np


DataType = Union[jnp.ndarray, Dict[str, "DataType"]]
