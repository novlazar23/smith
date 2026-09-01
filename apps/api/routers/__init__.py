"""API Router-Exports.

Bündelt alle Router-Module für die Zentralisierung in main.py.
"""

from __future__ import annotations

from .dashboard import router as dashboard_router  # noqa: F401
from .live_health import router as live_health_router  # noqa: F401
from .live_orders import router as live_orders_router  # noqa: F401
from .live_signal import router as live_signal_router  # noqa: F401
