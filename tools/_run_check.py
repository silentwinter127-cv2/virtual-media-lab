"""Run named checks out of tools/verify.py, so one failing check can be iterated on in seconds.

    python tools/_run_check.py check_fisheye_and_beam check_workspace_layout

`verify.py` has no switch for a single check (only `--quick`, `--json`, `--no-sheet`), and the whole suite
takes minutes because most of its cost is rendering.  This imports the module and calls the named
functions, so the result is the same `record()` path and the same PASS/FAIL formatting, on the same code.

Each check prints its own rows as it records them; this only adds the summary line and the list of
failures, which is what a single-check run needs to be readable.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import verify


def main(argv):
    names = argv or ["check_drag_tracking"]
    for n in names:
        fn = getattr(verify, n, None)
        if fn is None:
            print("  no such check: %s" % n)
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- a check that raises is a result, not a crash to hide
            print("  %s RAISED %s: %s" % (n, type(e).__name__, e))

    bad = [r for r in verify.RESULTS if not r.ok]
    for r in bad:
        print("  FAILED: %s -- %s" % (r.name, r.measured or ""))
    print()
    print("%d passed, %d failed" % (len(verify.RESULTS) - len(bad), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
