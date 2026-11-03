"""
Monte-Carlo parameter model.

Every physical input in the original application (bar properties, specimen
geometry, thermal properties...) was entered as a triplet:
    mean value / dispersion (std-dev or half-range) / distribution (Normal or Uniform)
and then expanded into an array of N Monte-Carlo draws (`*_pa` arrays in the
original code, e.g. `OB.D_pa`, `IB.rho_pa`).

That triplet + expansion logic was copy-pasted for every single parameter
(hundreds of times) in test_type.py / process_t.py. It is centralized here
once, and reused everywhere (see ui/widgets.py:MCParamRow for the matching
widget).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class Distribution(str, Enum):
    NORMAL = "Normal"
    UNIFORM = "Uniform"


@dataclass
class MCValue:
    """A single Monte-Carlo input parameter."""
    name: str
    unit: str = ""
    mean: float = 0.0
    dispersion: float = 0.0
    distribution: Distribution = Distribution.NORMAL

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Expand into an array of n Monte-Carlo draws (equivalent to *_pa)."""
        rng = rng or np.random.default_rng()
        if n <= 1 or self.dispersion == 0:
            return np.full(max(n, 1), float(self.mean))
        if self.distribution == Distribution.NORMAL:
            return rng.normal(self.mean, self.dispersion, size=n)
        # Uniform: dispersion is treated as a half-range around the mean,
        # matching the "Dispersion Value" convention used throughout the
        # original GUI.
        return rng.uniform(self.mean - self.dispersion, self.mean + self.dispersion, size=n)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "unit": self.unit,
            "mean": self.mean,
            "dispersion": self.dispersion,
            "distribution": self.distribution.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MCValue":
        return cls(
            name=d["name"],
            unit=d.get("unit", ""),
            mean=float(d.get("mean", 0.0)),
            dispersion=float(d.get("dispersion", 0.0)),
            distribution=Distribution(d.get("distribution", "Normal")),
        )


@dataclass
class MCParameterSet:
    """A named group of MCValue parameters (e.g. an Input Bar, a Specimen)."""
    label: str = ""
    values: dict[str, MCValue] = field(default_factory=dict)

    def add(self, key: str, name: str, unit: str = "", mean: float = 0.0) -> MCValue:
        v = MCValue(name=name, unit=unit, mean=mean)
        self.values[key] = v
        return v

    def sample_all(self, n: int, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        rng = rng or np.random.default_rng()
        return {k: v.sample(n, rng) for k, v in self.values.items()}

    def __getitem__(self, key: str) -> MCValue:
        return self.values[key]
