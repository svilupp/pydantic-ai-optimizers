"""Generic dataset interfaces for prompt optimization."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ReportCase:
    """A single evaluation case with input and expected output."""

    input: Any
    expected: Any
    metadata: dict[str, Any] | None = None


class Dataset(Protocol):
    """Protocol for evaluation datasets."""

    def __iter__(self) -> Iterator[ReportCase]:
        """Iterate over evaluation cases."""
        ...

    def __len__(self) -> int:
        """Return the number of cases in the dataset."""
        ...
