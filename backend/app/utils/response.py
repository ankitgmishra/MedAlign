"""Standardised API response helpers.

Every endpoint returns the same envelope:

.. code-block:: json

    {
        "success": true,
        "message": "...",
        "data": {...},
        "errors": null
    }
"""

from __future__ import annotations

from typing import Any, Optional


def api_response(
    *,
    success: bool = True,
    message: str = "OK",
    data: Optional[Any] = None,
    errors: Optional[Any] = None,
) -> dict[str, Any]:
    """Build a consistent API response envelope."""
    return {
        "success": success,
        "message": message,
        "data": data,
        "errors": errors,
    }
