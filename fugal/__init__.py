"""Fugal — one forward pass picks the model, then calls it once."""
from .router import Fugal, FugalRouter, or_call, or_request

__version__ = "1.0.0"
__all__ = ["Fugal", "FugalRouter", "or_call", "or_request"]
