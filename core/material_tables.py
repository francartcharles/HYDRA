"""
Temperature-dependent bar material properties.

Ported from import_bar_tab.py. The original module kept nine independent
copy-pasted "import a column of numbers, build a cubic interpolant against
a shared temperature axis" blocks (T/e/rho/nu/lambda, once for the input
bar, once for the output bar). That is now one small class used twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.interpolate as sci


def _read_numeric_column(path: str | Path) -> np.ndarray:
    """Read a whitespace/line-delimited numeric text file, skipping blank
    lines and '#' comments -- same format used throughout the original app.
    """
    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(float(line))
    return np.asarray(values)


@dataclass
class MaterialTable:
    """Cubic-interpolated temperature-dependent bar properties.

    Call `load_temperature()` first (defines the shared T axis), then any
    of `load_e`, `load_rho`, `load_nu`, `load_lambda` with a matching-length
    column. Each becomes an interpolator `<prop>_interp(T) -> value`.
    """
    label: str = ""
    temperature: np.ndarray | None = None
    _interpolators: dict = field(default_factory=dict)
    _raw: dict = field(default_factory=dict)

    def load_temperature(self, path: str | Path) -> None:
        self.temperature = _read_numeric_column(path)

    def _load_property(self, prop: str, path: str | Path) -> None:
        if self.temperature is None:
            raise ValueError("Load the temperature column first.")
        values = _read_numeric_column(path)
        t_axis = np.linspace(self.temperature.min(), self.temperature.max(), num=len(self.temperature))
        self._raw[prop] = values
        self._interpolators[prop] = sci.interp1d(t_axis, values, kind="cubic")

    def load_e(self, path: str | Path) -> None:
        self._load_property("e", path)

    def load_rho(self, path: str | Path) -> None:
        self._load_property("rho", path)

    def load_nu(self, path: str | Path) -> None:
        self._load_property("nu", path)

    def load_lambda(self, path: str | Path) -> None:
        self._load_property("lambda", path)

    def has(self, prop: str) -> bool:
        return prop in self._interpolators

    def value(self, prop: str, temperature_k: float, default: float) -> float:
        """Interpolated property at temperature_k, or `default` if that
        property table was never loaded (mirrors the *_check fallbacks
        used throughout the original code)."""
        if not self.has(prop):
            return default
        return float(self._interpolators[prop](temperature_k))
