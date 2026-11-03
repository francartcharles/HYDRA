"""
Load/save Monte-Carlo parameter sets from the original file format.

Each line of these files is a Python list literal:
    ['<mean>', '<dispersion>', '<N or U>']
one line per parameter, in a fixed order specific to what's being
described (see `core.models.Bar.PARAM_ORDER`). 'N' = Normal, 'U' = Uniform.
"""
from __future__ import annotations

import ast

from core.mc import MCParameterSet, Distribution

_CODE_TO_DIST = {"N": Distribution.NORMAL, "U": Distribution.UNIFORM}
_DIST_TO_CODE = {Distribution.NORMAL: "N", Distribution.UNIFORM: "U"}


def load_param_file(path: str, pset: MCParameterSet, order: list[str]) -> None:
    """Read a parameter file and apply the values onto an existing
    MCParameterSet in-place, following `order` (one key per line)."""
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < len(order):
        raise ValueError(
            f"Expected {len(order)} parameter lines, found {len(lines)} in {path}"
        )

    for key, line in zip(order, lines):
        mean_str, dispersion_str, dist_code = ast.literal_eval(line)
        value = pset[key]
        value.mean = float(mean_str)
        value.dispersion = float(dispersion_str)
        value.distribution = _CODE_TO_DIST.get(dist_code.strip().upper(), Distribution.NORMAL)


def save_param_file(path: str, pset: MCParameterSet, order: list[str]) -> None:
    """Write a parameter file in the same format `load_param_file` reads."""
    with open(path, "w") as f:
        for key in order:
            value = pset[key]
            code = _DIST_TO_CODE[value.distribution]
            f.write(f"['{value.mean}', '{value.dispersion}', '{code}']\n")
