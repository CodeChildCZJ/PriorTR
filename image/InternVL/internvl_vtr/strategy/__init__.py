from .registry import get_strategy, register_strategy
from .base import PruningStrategy

# Import strategy modules to trigger @register_strategy decorators
from . import fastv  # noqa: F401
from . import priortr  # noqa: F401
