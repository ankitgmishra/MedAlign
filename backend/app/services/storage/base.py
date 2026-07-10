"""Abstract base for storage services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseStorageService(ABC):
    """Interface for reading / writing evaluation artefacts."""

    @abstractmethod
    def save(self, data: Any, path: str | Path) -> Path:
        """Persist *data* to *path* and return the resolved path."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> Any:
        """Load and return data from *path*."""
        ...

    @abstractmethod
    def exists(self, path: str | Path) -> bool:
        """Return ``True`` if *path* exists in the storage backend."""
        ...
