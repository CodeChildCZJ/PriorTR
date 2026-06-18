# Strategy Registry
from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import PruningStrategy

# Global strategy registry
VTR_REGISTRY: Dict[str, Type["PruningStrategy"]] = {}


def register_strategy(name: str):
    """
    Strategy registration decorator.

    Usage:
        @register_strategy("fastv")
        class FastVStrategy(PruningStrategy):
            ...
    """
    def decorator(cls: Type["PruningStrategy"]):
        VTR_REGISTRY[name] = cls
        return cls
    return decorator


def get_strategy(name: str) -> "PruningStrategy":
    """
    Get a strategy instance by name.

    Args:
        name: strategy name (must be registered)

    Returns:
        Strategy instance

    Raises:
        ValueError: if strategy is not registered
    """
    if name not in VTR_REGISTRY:
        available = list(VTR_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    return VTR_REGISTRY[name]()
