"""
Mechanical impedance / transmission coefficient wave prediction.

Predicts what the transmitted wave SHOULD look like if the specimen
responded purely elastically, from the measured incident+reflected wave,
using nothing but 1D mechanical impedance (Z = rho*C0*A) and the classic
transmission coefficient at each interface the wave crosses:
Bar_in -> [Platen_in] -> Specimen -> [Platen_out] -> Bar_out.

This ignores internal reverberations within the (thin) specimen and any
inelastic (plastic) response -- it's a first-order prediction good enough
to identify where the measured transmitted wave's onset SHOULD be and
roughly what amplitude to expect, for realigning it against the actual
recorded signal without needing to assume anything about the specimen's
real (nonlinear) behaviour beyond its initial elastic properties.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def mechanical_impedance(elastic_modulus: float, density: float, diameter: float) -> tuple[float, float]:
    """Returns (Z, C0) for a cylindrical rod: Z = rho * C0 * A, C0 = sqrt(E/rho)."""
    area = math.pi * (diameter / 2) ** 2
    c0 = math.sqrt(elastic_modulus / density)
    return density * c0 * area, c0


def transmission_coefficient(z1: float, z2: float) -> float:
    """Stress/force transmission coefficient for a wave going from a
    medium of impedance z1 into a medium of impedance z2."""
    return 2 * z2 / (z1 + z2)


@dataclass
class PredictedWaveResult:
    predicted_transmitted: np.ndarray
    total_transmission_coefficient: float
    impedances: dict  # {"input_bar", "platen", "specimen", "output_bar"} -> Z


def predicted_transmitted_wave(
    incident_plus_reflected_force,
    ib_modulus: float, ib_density: float, ib_diameter: float,
    specimen_modulus: float, specimen_density: float, specimen_diameter: float,
    ob_modulus: float, ob_density: float, ob_diameter: float,
    use_platens: bool = False, platen_modulus: float = 0.0, platen_density: float = 0.0, platen_diameter: float = 0.0,
) -> PredictedWaveResult:
    """Predicts the theoretical elastic transmitted FORCE wave from the
    incident+reflected (input bar) FORCE wave, via the product of
    transmission coefficients through every interface: input bar ->
    [platen] -> specimen -> [platen] -> output bar."""
    z_ib, _ = mechanical_impedance(ib_modulus, ib_density, ib_diameter)
    z_ob, _ = mechanical_impedance(ob_modulus, ob_density, ob_diameter)
    z_sp, _ = mechanical_impedance(specimen_modulus, specimen_density, specimen_diameter)
    impedances = {"input_bar": z_ib, "specimen": z_sp, "output_bar": z_ob}

    if use_platens:
        z_pl, _ = mechanical_impedance(platen_modulus, platen_density, platen_diameter)
        chain = [z_ib, z_pl, z_sp, z_pl, z_ob]
        impedances["platen"] = z_pl
    else:
        chain = [z_ib, z_sp, z_ob]

    t_total = 1.0
    for i in range(len(chain) - 1):
        t_total *= transmission_coefficient(chain[i], chain[i + 1])

    predicted = np.asarray(incident_plus_reflected_force) * t_total
    return PredictedWaveResult(predicted_transmitted=predicted, total_transmission_coefficient=t_total, impedances=impedances)


def _smooth_edge_padded(signal: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smooth, padding with edge values (not zeros) so
    there's no spurious artificial jump at the boundaries."""
    if window <= 1:
        return signal
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(signal, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def best_alignment_shift(measured, theoretical, max_shift_samples: int) -> int:
    """Finds the integer sample shift (within +/- max_shift_samples) of
    `measured` that best matches `theoretical`, by cross-correlation on
    their OVERALL (heavily smoothed) shapes -- comparing raw, unsmoothed
    signals point-by-point does not work well here, since the theoretical
    wave's local wiggles are just a scaled copy of the incident+reflected
    wave's own wiggles, which generally do NOT match the measured
    transmitted wave's own local wiggles (different physical origin --
    dispersion, reverberation, real material response). Only the broad
    rise/plateau/decay shape is comparable, so both curves are smoothed
    down to that broad trend before comparing."""
    measured = np.asarray(measured, dtype=float)
    theoretical = np.asarray(theoretical, dtype=float)
    n = min(len(measured), len(theoretical))
    measured, theoretical = measured[:n], theoretical[:n]
    if n < 2:
        return 0

    window = max(1, n // 8)
    measured_smooth = _smooth_edge_padded(measured, window)
    theoretical_smooth = _smooth_edge_padded(theoretical, window)

    # Normalize (zero mean, unit energy) so the correlation compares SHAPE,
    # not raw amplitude scale (which differs a lot between the two here).
    def _normalize(x):
        x = x - x.mean()
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x

    m_norm, t_norm = _normalize(measured_smooth), _normalize(theoretical_smooth)
    best_shift, best_score = 0, -np.inf
    max_shift_samples = min(max_shift_samples, n - 1)
    for shift in range(-max_shift_samples, max_shift_samples + 1):
        shifted = np.roll(m_norm, shift)
        if shift > 0:
            shifted[:shift] = 0
        elif shift < 0:
            shifted[shift:] = 0
        score = float(np.dot(shifted, t_norm))
        if score > best_score:
            best_score, best_shift = score, shift
    return best_shift
