import os
from functools import lru_cache


@lru_cache()
def default_cache_dir() -> str:
    return os.path.join(os.path.abspath(os.getcwd()), "point_e_model_cache")
