"""Virtual Media Lab — Native Low-Overhead Desktop GUI Package.

Eliminates browser DOM/Canvas overhead, HTTP loopback polling, and per-frame
PNG/JPEG encoding by driving `atsim` directly in-process with bounded caches.
"""

from .controller import DesktopBackendController, SELECTORS, PRESET_CATEGORIES

__all__ = ["DesktopBackendController", "SELECTORS", "PRESET_CATEGORIES"]
