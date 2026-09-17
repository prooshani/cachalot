"""Cachalot: SSD-backed DeepSeek V4.1 Flash runtime for Apple Silicon."""

from cachalot.config import DEFAULT_CONFIG, RuntimeConfig
from cachalot.model.api import ChatResponse, V41Model

__version__ = "0.4.0"

__all__ = ["ChatResponse", "DEFAULT_CONFIG", "RuntimeConfig", "V41Model", "__version__"]
