"""
Load the notched-specimen k/sk fit coefficients from a plain text file.

Matches the original app's own "readme"-style export for these coefficients
(one value per line): 6 `k_value` entries, optionally followed by a
`Split = True`/`Split = False` marker line (whether the fit is split at a
displacement threshold) and 7 `sk_value` entries -- used only when that
split is enabled. Real exported files were found to come in both shapes:
just the 6 k-values (no split/sk section at all -- "sk" defaults to off and
was simply never part of that export), or the full 14 lines. Both are
accepted; only a length that matches neither (7-13 lines: an incomplete
split/sk section) is treated as an error, since that's genuinely ambiguous
rather than a recognized shorter format.

This is a different, simpler, specific format from core/param_io.py's bar/
specimen/platen config files (which are Python list literals
`['mean', 'dispersion', 'N'/'U']` per Monte-Carlo parameter) -- k/sk
coefficients are plain fitted constants, not Monte-Carlo parameters, so
there's no mean/dispersion/distribution to carry.
"""
from __future__ import annotations

import re

N_K = 6
N_SK = 7

_SPLIT_RE = re.compile(r"split\s*=\s*(true|false)", re.IGNORECASE)


def load_k_value_file(path: str) -> tuple[list[float], bool | None, list[float] | None]:
    """Reads a k-coefficient file and returns (k_value, sk_check, sk_value).

    `sk_check`/`sk_value` come back as `None` (rather than some made-up
    default) when the file only has the 6 `k_value` lines and no split/sk
    section -- the caller should then leave whatever split/sk settings are
    already on screen alone, not overwrite them with a guess.

    Expected shape (blank lines ignored): 6 lines of `k_value[0..5]`,
    optionally followed by one `Split = True`/`Split = False` line and then
    7 lines of `sk_value[0..6]`."""
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < N_K:
        raise ValueError(f"Expected at least {N_K} lines (k_value[0..{N_K - 1}]), found {len(lines)} in {path}")

    try:
        k_value = [float(v) for v in lines[:N_K]]
    except ValueError as exc:
        raise ValueError(f"Could not parse k_value[0..{N_K - 1}] as numbers in {path}: {exc}") from exc

    if len(lines) == N_K:
        return k_value, None, None

    expected = N_K + 1 + N_SK
    if len(lines) < expected:
        raise ValueError(
            f"Found {len(lines)} lines: {N_K} k-values (parsed OK) plus a partial split/sk "
            f"section. A split/sk section, if present, must be complete -- a 'Split = True/False' "
            f"line plus all {N_SK} sk-values ({expected} lines total), or omitted entirely "
            f"({N_K} lines total) in {path}"
        )

    split_line = lines[N_K]
    match = _SPLIT_RE.match(split_line)
    if not match:
        raise ValueError(
            f"Expected a 'Split = True' or 'Split = False' line at line {N_K + 1}, "
            f"found: {split_line!r} in {path}"
        )
    sk_check = match.group(1).lower() == "true"

    sk_lines = lines[N_K + 1: N_K + 1 + N_SK]
    try:
        sk_value = [float(v) for v in sk_lines]
    except ValueError as exc:
        raise ValueError(f"Could not parse sk_value[0..{N_SK - 1}] as numbers in {path}: {exc}") from exc

    return k_value, sk_check, sk_value


def save_k_value_file(path: str, k_value: list[float], sk_check: bool, sk_value: list[float]) -> None:
    """Writes a k-coefficient file in the same format `load_k_value_file` reads."""
    with open(path, "w") as f:
        for v in k_value:
            f.write(f"{v}\n")
        f.write(f"Split = {sk_check}\n")
        for v in sk_value:
            f.write(f"{v}\n")
