"""
Wave-travel-time delay between a bar strain gage and a specimen-mounted
sensor (e.g. a strain gage glued directly on the specimen), for SHPB and
Direct Impact tests.

A signal recorded by a sensor mounted on the specimen doesn't start
"seeing" the test at the same instant as the bar-mounted strain gage --
the loading wave still has to travel from the gage to the specimen (and
through any platens, and half-way through the specimen itself, assuming
the sensor sits roughly at the specimen's mid-length) first. These
functions compute that delay so a specimen-mounted sensor's recording can
be cut/re-zeroed to start at the same physical instant as the bar-wave
signals.

    SHPB case:
        dt = Input_gage_position / C0_input_bar
           + Platens_length / C0_platens   (only if platens are used)
           + Sample_Length / (2 * C0_sample)

    Direct Impact case:
        dt = Output_gage_position / C0_output_bar
           + Platens_length / C0_platens   (only if platens are used)
           + Sample_Length / (2 * C0_sample)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DelayBreakdown:
    bar_term: float        # time spent travelling the bar, from the gage to the specimen
    platen_term: float     # time spent travelling through the platen (0 if platens aren't used)
    specimen_term: float   # time spent travelling through half the specimen
    total: float


def specimen_sensor_delay_shpb(
    input_gage_position: float, c0_input_bar: float,
    sample_length: float, c0_sample: float,
    use_platens: bool = False, platens_length: float = 0.0, c0_platens: float = 0.0,
) -> DelayBreakdown:
    bar_term = input_gage_position / c0_input_bar
    specimen_term = sample_length / (2 * c0_sample)
    platen_term = platens_length / c0_platens if use_platens and c0_platens else 0.0
    return DelayBreakdown(bar_term=bar_term, platen_term=platen_term, specimen_term=specimen_term,
                           total=bar_term + platen_term + specimen_term)


def specimen_sensor_delay_direct_impact(
    output_gage_position: float, c0_output_bar: float,
    sample_length: float, c0_sample: float,
    use_platens: bool = False, platens_length: float = 0.0, c0_platens: float = 0.0,
) -> DelayBreakdown:
    bar_term = output_gage_position / c0_output_bar
    specimen_term = sample_length / (2 * c0_sample)
    platen_term = platens_length / c0_platens if use_platens and c0_platens else 0.0
    return DelayBreakdown(bar_term=bar_term, platen_term=platen_term, specimen_term=specimen_term,
                           total=bar_term + platen_term + specimen_term)


def sample_wave_speed(elastic_modulus: float, density: float) -> float:
    """C0 = sqrt(E/rho) for the specimen material, used when the specimen
    doesn't otherwise have its own directly-measured wave speed."""
    return math.sqrt(elastic_modulus / density)


def specimen_sensor_cut_points_shpb(
    xd1: float,
    input_gage_position_pa: np.ndarray, c0_input_bar_pa: np.ndarray,
    sample_length_pa: np.ndarray, elastic_modulus_pa: np.ndarray, density_pa: np.ndarray,
    use_platens: bool = False,
    platens_length_pa: np.ndarray | None = None,
    platens_elastic_modulus_pa: np.ndarray | None = None,
    platens_density_pa: np.ndarray | None = None,
) -> np.ndarray:
    """Per-Monte-Carlo-draw version of `specimen_sensor_delay_shpb`'s cut
    point (`xd1 + dt.total`).

    The Processing page's "Auto-cut using wave travel time" button
    computes that delay ONCE, from the MEAN input-bar/platen/specimen
    properties -- reasonable as a quick default, but it throws away
    whatever uncertainty those same properties already carry everywhere
    else in the Monte-Carlo loop. This evaluates the exact same formula
    once per draw instead, using that draw's own sampled properties (the
    `*_pa` arrays `Bar.expand()`/`Specimen.expand()`/`PlatenSettings.
    expand()` already populate for every OTHER per-draw quantity in
    core/process_shpb.py) -- e.g. `specimen_C0_pa = sqrt(specimen.e_pa[d]
    / specimen.rho_pa[d])` is exactly process_shpb.py's own per-draw
    specimen wave-speed formula. This is what lets a specimen-mounted
    sensor's alignment be included in the Monte-Carlo loop (see
    CustomSensor.cut_in_montecarlo) instead of being fixed to one value
    shared by every draw."""
    n = len(input_gage_position_pa)
    cut_points = np.empty(n)
    for d in range(n):
        c0_sample = sample_wave_speed(elastic_modulus_pa[d], density_pa[d])
        if use_platens:
            c0_platens = sample_wave_speed(platens_elastic_modulus_pa[d], platens_density_pa[d])
            platens_length = platens_length_pa[d]
        else:
            c0_platens, platens_length = 0.0, 0.0
        dt = specimen_sensor_delay_shpb(
            input_gage_position=input_gage_position_pa[d], c0_input_bar=c0_input_bar_pa[d],
            sample_length=sample_length_pa[d], c0_sample=c0_sample,
            use_platens=use_platens, platens_length=platens_length, c0_platens=c0_platens,
        )
        cut_points[d] = xd1 + dt.total
    return cut_points


def specimen_sensor_cut_points_direct_impact(
    xd1_output: float,
    output_gage_position_pa: np.ndarray, c0_output_bar_pa: np.ndarray,
    sample_length_pa: np.ndarray, elastic_modulus_pa: np.ndarray, density_pa: np.ndarray,
    use_platens: bool = False,
    platens_length_pa: np.ndarray | None = None,
    platens_elastic_modulus_pa: np.ndarray | None = None,
    platens_density_pa: np.ndarray | None = None,
) -> np.ndarray:
    """Direct Impact counterpart of `specimen_sensor_cut_points_shpb` --
    see its docstring. Mirrors `_on_align_sensor`'s Direct Impact branch:
    the specimen started loading `dt.total` BEFORE the output-bar gage
    registered the transmitted wave, so the cut point is `xd1_output -
    dt.total`, not `+ dt.total`."""
    n = len(output_gage_position_pa)
    cut_points = np.empty(n)
    for d in range(n):
        c0_sample = sample_wave_speed(elastic_modulus_pa[d], density_pa[d])
        if use_platens:
            c0_platens = sample_wave_speed(platens_elastic_modulus_pa[d], platens_density_pa[d])
            platens_length = platens_length_pa[d]
        else:
            c0_platens, platens_length = 0.0, 0.0
        dt = specimen_sensor_delay_direct_impact(
            output_gage_position=output_gage_position_pa[d], c0_output_bar=c0_output_bar_pa[d],
            sample_length=sample_length_pa[d], c0_sample=c0_sample,
            use_platens=use_platens, platens_length=platens_length, c0_platens=c0_platens,
        )
        cut_points[d] = xd1_output - dt.total
    return cut_points
