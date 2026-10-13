"""Sea-state presets: wave spectrum, surface look and wind spray, applied to a shot.

    python3 -m factory.poses MESH.obj --sea rough -o shots/     # adds the preset to each pose

Keep ``direction_angular_spread_deg`` at 120 or more: with a narrow spread (50 or less)
the Water plugin's surface stops occluding geometry below the calm water level (seen
2026-10-10 with striped probe poles), so the render and the label disagree.

Keep the *summed* steepness below about 0.6 (``_waves`` takes totals and divides by the
wave count). In the plugin's Gerstner shader each wave's horizontal displacement has a
derivative equal to its steepness, so when the sum exceeds 1 the surface folds over itself
(sharp creases) and the water height above a point is no longer single-valued: the
label's inversion of the displacement cannot match the render (run01, 2026-10-11).

A preset sets three things in the shot JSON, all of which wf_shot.py applies without a
scene rebuild: ``waves`` (GerstnerWaterWaveGeneratorSimple parameters, cm and degrees),
and ``ocean`` (material instance parameters: surface detail, roughness). The plugin's
crest foam switch is left off: close to the camera it renders as beige blocks.
The label pass evaluates the same waves the water renders, so any state labels correctly.

Wave phase: every shot carries ``wave_time_s`` (factory/shotlist.py) and renders with the water
frozen at that time (r.Water.OverrideWavesTime, keyed per shot by wf_build_batch.py). Knowing
the phase, wf_build_batch re-evaluates the generated Gerstner waves in Python and lifts the
whale (with its camera) and the splash onto the actual surface: shotlist places both against
the mean plane, so without this a splash sat 1 m above a trough in a rough sea and a whale's
waterline moved with the phase. ``crest_height_m`` remains a safety margin
(body.wave_clearance_fraction) because the surface is only matched at one point of the body.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from factory import splash as splash_module


@dataclass(frozen=True)
class SeaState:
    description: str
    waves: dict
    ocean_scalars: dict = field(default_factory=dict)
    ocean_switches: dict = field(default_factory=dict)
    wind_spray: int = 0           # reserved: blown spray is not modelled yet (sprites looked wrong)


MAX_TOTAL_STEEPNESS = 0.6


def _waves(num, min_wl, max_wl, min_amp, max_amp, small_steep_total, large_steep_total, spread, wind=35.0, seed=7):
    """small/large_steep_total are the *summed* steepness over all waves (kept <= MAX_TOTAL_STEEPNESS)."""
    small_steep_total = min(small_steep_total, MAX_TOTAL_STEEPNESS)
    large_steep_total = min(large_steep_total, MAX_TOTAL_STEEPNESS)
    return {
        "num_waves": num, "seed": seed,
        "min_wavelength": min_wl, "max_wavelength": max_wl, "wavelength_falloff": 1.5,
        "min_amplitude": min_amp, "max_amplitude": max_amp, "amplitude_falloff": 2.0,
        "wind_angle_deg": wind, "direction_angular_spread_deg": spread,
        "small_wave_steepness": round(small_steep_total / num, 4), "large_wave_steepness": round(large_steep_total / num, 4),
        "steepness_falloff": 2.0,
    }


SEA_STATES: dict[str, SeaState] = {
    "glassy": SeaState("flat calm: long low swell, mirror-like surface",
                       _waves(12, 400.0, 3000.0, 0.2, 2.0, 0.3, 0.15, 150.0),
                       {"Default Near Normal Strength": 0.15, "Default Distant Normal Strength": 0.15, "Water Roughness": 0.015}),
    "calm": SeaState("light ripple, no whitecaps",
                     _waves(16, 150.0, 3500.0, 0.3, 4.0, 0.45, 0.2, 150.0),
                     {"Default Near Normal Strength": 0.4, "Default Distant Normal Strength": 0.3, "Water Roughness": 0.03}),
    # Amplitudes are per wave (cm); the plugin sums them into its "max wave height", so
    # 16 waves with a 2.0 falloff give crests of very roughly 2-3x the largest amplitude.
    # Rendered at full amplitude since the ocean's depth attenuation was switched off
    # (wf_build_scene.build_ocean); these values give the look approved on 2026-10-11.
    "light": SeaState("light breeze: chop over a low swell (the scene default)",
                      _waves(16, 120.0, 5000.0, 0.5, 9.0, 0.55, 0.3, 140.0),
                      {"Default Near Normal Strength": 0.7, "Default Distant Normal Strength": 0.5, "Water Roughness": 0.04}),
    "moderate": SeaState("moderate breeze: 1 m waves, scattered whitecaps",
                         _waves(16, 150.0, 8000.0, 1.0, 20.0, 0.6, 0.35, 130.0),
                         {"Default Near Normal Strength": 0.85, "Default Distant Normal Strength": 0.6, "Water Roughness": 0.05},
                         ),
    "rough": SeaState("fresh to strong breeze: 2-3 m waves, whitecaps and blown spray",
                      _waves(16, 200.0, 12000.0, 2.5, 50.0, 0.6, 0.4, 125.0),
                      {"Default Near Normal Strength": 1.0, "Default Distant Normal Strength": 0.7, "Water Roughness": 0.06},
                      ),
}
DEFAULT_SEA_STATE = "light"


def crest_height_m(name: str) -> float:
    """Rough height of the wave crests above the mean plane for a preset (metres): about 2.5x the
    largest per-wave amplitude, see the note on the presets. Used to lift low poses clear of the
    waves so that a sliver meant to be visible is not washed over in a heavy sea."""
    return 2.5 * float(SEA_STATES[name].waves["max_amplitude"]) / 100.0


MIN_SPREAD_DEG = 120.0


def apply_to_shot(shot: dict, name: str, seed: int = 0) -> dict:
    """Add a sea state to a shot dict (from factory.poses) in place and return it."""
    if name not in SEA_STATES:
        raise ValueError(f"unknown sea state {name!r}; choose from {list(SEA_STATES)}")
    state = SEA_STATES[name]
    if state.waves["direction_angular_spread_deg"] < MIN_SPREAD_DEG:
        raise ValueError(f"sea state {name!r}: direction spread must be >= {MIN_SPREAD_DEG} (see module doc)")
    total = state.waves["num_waves"] * max(state.waves["small_wave_steepness"], state.waves["large_wave_steepness"])
    if total > MAX_TOTAL_STEEPNESS + 1e-6:
        raise ValueError(f"sea state {name!r}: summed steepness {total:.2f} exceeds {MAX_TOTAL_STEEPNESS} (see module doc)")
    shot["sea_state"] = name
    shot["waves"] = dict(state.waves, seed=state.waves["seed"] + seed)
    ocean = shot.setdefault("ocean", {})
    ocean.setdefault("scalars", {}).update(state.ocean_scalars)
    if state.ocean_switches:
        ocean.setdefault("switches", {}).update(state.ocean_switches)
    return shot
