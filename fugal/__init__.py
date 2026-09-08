"""Fugal — one forward pass picks the model, then calls it once."""
import io
import sys

from .router import Fugal, FugalRouter, or_call, or_request

__version__ = "1.0.0"
__all__ = ["Fugal", "FugalRouter", "or_call", "or_request"]


def use_utf8():
    """Rebind sys.stdout/sys.stderr to UTF-8 with replacement, line-buffered.

    Line-buffered because the rebound stream is usually a pipe or a journal, not a
    terminal, and the server's banner and per-request log lines must reach it as they
    happen — not when a buffer fills, and not never, if the process is killed first.

    Call this from a program's entry point ONLY. Importing a library must never reach in
    and rebind the streams of whatever application imported it."""
    for name in ("stdout", "stderr"):
        s = getattr(sys, name)
        if hasattr(s, "buffer"):
            setattr(sys, name, io.TextIOWrapper(s.buffer, encoding="utf-8",
                                                errors="replace", line_buffering=True))
