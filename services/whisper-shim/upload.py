"""Upload ceiling, concurrency slots, and the multipart form rebuilt for the backend."""
from __future__ import annotations

from typing import Any


class TooLarge(Exception):
    """Upload past the size ceiling."""


class Slots:
    """Bounded count of transcriptions in progress; a full set refuses rather than queues."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True

    def give(self) -> None:
        self.used = max(0, self.used - 1)


async def rebuild_form(form: Any, model_name: str, max_bytes: int) -> tuple[dict, dict]:
    """(files, fields) for httpx, with `model` replaced by `model_name`.

    Raises TooLarge once the files together pass `max_bytes`."""
    files: dict[str, tuple] = {}
    fields: dict[str, str] = {}
    total = 0
    for key, value in form.multi_items():
        if hasattr(value, "filename"):
            size = getattr(value, "size", None)
            if size is not None and total + size > max_bytes:
                raise TooLarge(f"upload exceeds {max_bytes} bytes")
            # Read one byte past the remaining budget to detect an overrun.
            data = await value.read(max_bytes - total + 1)
            total += len(data)
            if total > max_bytes:
                raise TooLarge(f"upload exceeds {max_bytes} bytes")
            files[key] = (value.filename, data,
                          value.content_type or "application/octet-stream")
        else:
            fields[key] = str(value)
    fields["model"] = model_name
    return files, fields
