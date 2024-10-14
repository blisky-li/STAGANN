import os
from easytorch.utils.registry import scan_modules

from .registry import SCALER_REGISTRY
from .datasets import STKrigingDataset,node_index_get, read_location

__all__ = ["SCALER_REGISTRY", "STKrigingDataset", "node_index_get", "read_location"]
# fix bugs on Windows systems and on jupyter
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
scan_modules(project_dir, __file__, ["__init__.py", "registry.py"])
