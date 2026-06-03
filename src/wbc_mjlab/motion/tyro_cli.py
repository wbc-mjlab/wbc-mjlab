"""Helpers for mjlab tyro CLIs (``FlagConversionOff`` requires ``True``/``False`` values)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

import tyro


def normalize_bool_flag_argv(
    argv: Sequence[str],
    *field_names: str,
) -> list[str]:
    """Turn bare ``--flag`` into ``--flag True`` for ``FlagConversionOff`` CLIs."""
    out = list(argv)
    for name in field_names:
        flag = f"--{name.replace('_', '-')}"
        i = 0
        while i < len(out):
            if out[i] == flag:
                has_value = i + 1 < len(out) and not out[i + 1].startswith("-")
                if not has_value:
                    out.insert(i + 1, "True")
                    i += 2
                    continue
            i += 1
    return out


def cli(
    fn: Callable[..., None],
    /,
    *,
    bool_shorthand: tuple[str, ...] = (),
) -> None:
    import mjlab

    argv = normalize_bool_flag_argv(sys.argv[1:], *bool_shorthand)
    tyro.cli(fn, config=mjlab.TYRO_FLAGS, args=argv)
