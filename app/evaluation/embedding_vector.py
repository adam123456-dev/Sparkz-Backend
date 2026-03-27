"""Normalize embedding values from Supabase/PostgREST into a float32 vector."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np


def embedding_to_float_vector(raw: Any) -> np.ndarray:
    """
    PostgREST often returns ``vector`` as a JSON array *string* (not a Python list).
    ``np.asarray(string, dtype=float32)`` fails because it iterates characters.
    """
    if raw is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(raw, np.ndarray):
        return raw.astype(np.float32, copy=False)
    if isinstance(raw, (list, tuple)):
        return np.asarray(raw, dtype=np.float32)
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            return np.asarray(json.loads(s), dtype=np.float32)
        if s.startswith("{") and s.endswith("}"):
            inner = s[1:-1].strip()
            if inner:
                return np.asarray([float(x) for x in inner.split(",")], dtype=np.float32)
            return np.zeros(0, dtype=np.float32)
        m = re.search(r"\[[\s\-0-9eE+.,]+\]", s)
        if m:
            return np.asarray(json.loads(m.group(0)), dtype=np.float32)
        return np.asarray(json.loads(s), dtype=np.float32)
    raise TypeError(f"Unsupported embedding type: {type(raw)!r}")
