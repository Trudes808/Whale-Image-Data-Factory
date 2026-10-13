"""The Water plugin's Gerstner waves, in numpy: the same waves the ocean renders and the label
pass evaluates, generated from a sea state's spectrum exactly as GerstnerWaterWaveGeneratorSimple
does (its FRandomStream is a 32-bit LCG, replicated here), and the surface displacement from
GerstnerWaveFunctions.ush. Lets the factory know the water surface at a shot's frozen wave time
before the editor runs (whale/splash placement, white water along the contact line).

    waves = generate_waves(shot["waves"])            # list of dicts: wavelength, amplitude, steepness, direction
    z = surface_height(waves, x_cm, y_cm, t_s)       # cm above the mean plane at rest positions (x, y)
"""
from __future__ import annotations

import math

import numpy as np

GRAVITY_CM_S2 = 980.0


class RandomStream:
    """UE's FRandomStream: Seed = Seed * 196314165 + 907633515 (int32 wrap); fraction = mantissa bits of Seed >> 9."""

    def __init__(self, seed: int):
        self.seed = int(seed) & 0xFFFFFFFF

    def fraction(self) -> float:
        self.seed = (self.seed * 196314165 + 907633515) & 0xFFFFFFFF
        bits = 0x3F800000 | (self.seed >> 9)
        return float(np.frombuffer(np.uint32(bits).tobytes(), dtype=np.float32)[0]) - 1.0

    def rand_range(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.fraction()


def _f32(v: float) -> float:
    return float(np.float32(v))


def generate_waves(params: dict) -> list[dict]:
    """GerstnerWaterWaveGeneratorSimple.GenerateGerstnerWaves for a spectrum dict (the shot's "waves")."""
    n = int(params["num_waves"]); rng = RandomStream(int(params.get("seed", 0)))
    randomness = float(params.get("randomness", 0.0))
    wind = math.radians(float(params["wind_angle_deg"])); spread = float(params["direction_angular_spread_deg"])
    out = []
    for i in range(n):
        alpha = 1.0 - i / n + rng.rand_range(randomness * (-1.0 / n), randomness * (1.0 / n))
        alpha = min(max(alpha, 0.0), 1.0)
        dx, dy = math.cos(wind), math.sin(wind)                      # SinCos(&Y, &X, wind)
        if i > 0:
            a = math.radians(rng.rand_range(-spread, spread))         # RotateAngleAxis about +Z
            dx, dy = dx * math.cos(a) - dy * math.sin(a), dx * math.sin(a) + dy * math.cos(a)
        wavelength = _f32(float(params["min_wavelength"]) + (float(params["max_wavelength"]) - float(params["min_wavelength"])) * alpha ** float(params["wavelength_falloff"]))
        amplitude = max(_f32(float(params["min_amplitude"]) + (float(params["max_amplitude"]) - float(params["min_amplitude"])) * alpha ** float(params["amplitude_falloff"])), 0.0001)
        steepness = _f32(float(params["large_wave_steepness"]) + (float(params["small_wave_steepness"]) - float(params["large_wave_steepness"])) * (i / n) ** float(params["steepness_falloff"]))
        out.append({"wavelength": wavelength, "amplitude": amplitude, "steepness": steepness, "direction": [dx, dy]})
    return out


def displacement(waves: list[dict], x, y, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dx, dy, dz) in cm of the surface point whose rest position is (x, y) at time t (GerstnerWaveFunctions.ush)."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    dx = np.zeros_like(x); dy = np.zeros_like(x); dz = np.zeros_like(x)
    for w in waves:
        k = 2.0 * math.pi / w["wavelength"]; ux, uy = w["direction"]
        phase = (x * ux + y * uy) * k - math.sqrt(k * GRAVITY_CM_S2) * t
        s, c = np.sin(phase), np.cos(phase)
        qa = w["steepness"] / k
        dx -= qa * s * ux; dy -= qa * s * uy; dz += w["amplitude"] * c
    return dx, dy, dz


def surface_height(waves: list[dict], x, y, t: float, iterations: int = 8) -> np.ndarray:
    """Wave height (cm) above the mean plane at world (x, y): the horizontal displacement is inverted
    by fixed-point iteration, as the label shader does (summed steepness < 1)."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    qx, qy = x.copy(), y.copy()
    for _ in range(iterations):
        dx, dy, _ = displacement(waves, qx, qy, t)
        qx, qy = x - dx, y - dy
    return displacement(waves, qx, qy, t)[2]


def surface_points(waves: list[dict], x, y, t: float) -> np.ndarray:
    """Displaced surface points (N, 3) cm for rest-grid positions (x, y): the rendered water mesh's vertices."""
    dx, dy, dz = displacement(waves, x, y, t)
    return np.stack([np.asarray(x) + dx, np.asarray(y) + dy, dz], -1)
