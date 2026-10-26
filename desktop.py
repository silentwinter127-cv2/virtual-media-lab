#!/usr/bin/env python3
"""Virtual Media Lab — Native Low-Overhead Desktop GUI Launcher.

Supports both package layout (`atsim/desktop/app.py`) and flat root layout
(`desktop_app.py` + `controller.py` placed directly in the project root).

Usage:
  python desktop.py                  # Launch Native Low-Memory Desktop GUI
  python desktop.py --clip demo.mp4  # Launch and load a specific video/image clip
  python desktop.py --self-test      # Run headless engine & memory release self-test
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from desktop_app import main
except ImportError:
    from atsim.desktop.app import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
