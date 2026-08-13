# Fugal — Apache-2.0. See NOTICE.
"""Make stdout/stderr survive the non-ASCII characters this project's output uses.

Every user-facing surface here — the ranked table, help text, error messages, the price
drift report — uses em-dashes, arrows and >= signs. A Windows console defaults to cp1252
and raises UnicodeEncodeError on them, which turns a working tool into a crash with a
traceback that has nothing to do with what the user asked for.

stderr matters as much as stdout: argparse and sys.exit() write there, so an un-wrapped
stderr means the *error message explaining the problem* is itself the thing that crashes.
"""
from __future__ import annotations
import io
import sys


def use_utf8():
    """Rebind sys.stdout/sys.stderr to UTF-8 with replacement.

    Call this from a program's entry point ONLY. Importing a library must never reach in
    and rebind the streams of whatever application imported it — that is why this is a
    function you call rather than something that happens on import.
    """
    for name in ("stdout", "stderr"):
        s = getattr(sys, name)
        if hasattr(s, "buffer"):
            setattr(sys, name, io.TextIOWrapper(s.buffer, encoding="utf-8",
                                                errors="replace"))
