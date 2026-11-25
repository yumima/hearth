"""Asking a line of the person at the terminal."""

from __future__ import annotations

import re
import sys

_ESC = re.compile(r"\033\[[0-9;]*m")


def ask(prompt: str) -> str:
    """input(), with a prompt that may have colours: readline counts what it thinks
    is shown, and puts the cursor wrong unless it is told what is not."""
    lead = prompt[:len(prompt) - len(prompt.lstrip("\n"))]
    sys.stdout.write(lead)
    sys.stdout.flush()
    rest = prompt[len(lead):]
    return input(_ESC.sub(lambda m: f"\001{m.group(0)}\002", rest) if "readline" in sys.modules and sys.stdin.isatty() else rest)
