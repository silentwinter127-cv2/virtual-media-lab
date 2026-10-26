"""Virtual Media Lab (atsim) root package."""
import importlib.util
import os
import sys
from pathlib import Path

# Ensure submodules created on FUSE/overlay mounts with stale readdir caches remain importable
_pkg_dir = Path(__file__).resolve().parent
_desktop_init = _pkg_dir / "desktop" / "__init__.py"
if _desktop_init.exists() and "atsim.desktop" not in sys.modules:
    for _mod_name, _mod_file in [
        ("atsim.desktop.controller", _pkg_dir / "desktop" / "controller.py"),
        ("atsim.desktop.app", _pkg_dir / "desktop" / "app.py"),
        ("atsim.desktop", _desktop_init),
    ]:
        if _mod_file.exists() and _mod_name not in sys.modules:
            _spec = importlib.util.spec_from_file_location(
                _mod_name,
                str(_mod_file),
                submodule_search_locations=[str(_mod_file.parent)] if _mod_file.name == "__init__.py" else None,
            )
            if _spec and _spec.loader:
                _mod = importlib.util.module_from_spec(_spec)
                sys.modules[_mod_name] = _mod
                _spec.loader.exec_module(_mod)
