"""Individual whales for the rigged 4k humpback: a library of "copies" of the model that differ in
body proportions, resting joint angles around the keypoints, fluke outline (span, sweep, notch,
trailing-edge wear) and colouring (skin variant + a Katona-style fluke underside class 1-5).

    python3 -m factory.individuals --count 12 --seed 1
        -> third_party_assets/humpback_4k/individuals.json      (the specs, by id "i01".."i12")
           third_party_assets/humpback_4k/skins/T_Humpback_BaseColor_i01.png ...   (their textures)

A shot carries ``"individual": <id>``; ``WhaleModel.posed`` applies the geometry part through the
rig (rest-vertex edits before linear blend skinning, world-space joint rotation offsets on top of
the clip), and the renderer uses the individual's skin texture (``skin`` = the individual's id).
The canonical body frame does not change: vertex i of every individual still maps to canonical
vertex i, so the body-coordinate labels stay comparable across individuals.

Proportions are kept inside the natural range of adult humpbacks (head 8 % either way, girth
7 %, fluke span 10 %); calves get their own, chubbier range via ``sample(rng, calf=True)``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from factory import skin as skin_mod
from factory.skinned import MODEL_TO_PROJECT

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "third_party_assets" / "humpback_4k"
LIBRARY = ASSET_DIR / "individuals.json"
SKINS_DIR = ASSET_DIR / "skins"
LEGACY_DORSAL = os.environ.get("WF_LEGACY_DORSAL", "") == "1"      # re-label run04 with its own (pre-2026-10-20) fin edits
BASE_TEXTURE = ASSET_DIR / "converted" / "T_Humpback4k_BaseColor.png"

# spine joints in nose -> fluke order (project-frame x decreasing); the rig's names
SPINE = ["joint51_00", "joint52_01", "joint53_02", "joint54_03", "joint55_04", "joint56_05", "joint57_06", "joint58_07",
         "joint59_08", "joint60_09", "joint61_010", "joint62_011", "joint63_012"]
PEC_LEFT = ["joint64_013", "joint65_014", "joint66_015", "joint67_016", "joint68_017", "joint69_018"]
PEC_RIGHT = ["joint70_019", "joint71_020", "joint72_021", "joint73_022", "joint74_023", "joint75_024"]

# Katona fluke classes: fraction of the underside that is white
FLUKE_CLASS_WHITE = {1: 0.92, 2: 0.74, 3: 0.52, 4: 0.30, 5: 0.10}


@dataclass
class Individual:
    id: str
    seed: int
    calf: bool = False
    head_len: float = 1.0           # rostrum length scale (vertices ahead of the jaw joint)
    girth: float = 1.0              # body cross-section scale about the spine
    peduncle: float = 1.0           # tail-stock cross-section scale
    fluke_span: float = 1.0         # fluke width scale
    fluke_sweep_cm: float = 0.0     # tips swept back (+) / forward (-)
    fluke_notch_cm: float = 0.0     # extra depth of the central notch
    fluke_edge_cm: float = 0.0      # trailing-edge wear amplitude (scallops, nicks)
    pec_len: float = 1.0            # pectoral length scale
    dorsal_height: float = 1.0      # dorsal fin height scale (the asset's fin is a small one; real fins run stubby to tall and falcate)
    dorsal_sweep_cm: float = 0.0    # dorsal fin tip moved back (+) or forward (-)
    dorsal_edge_cm: float = 0.0     # dorsal fin trailing-edge wear amplitude (scallops, notches): the researchers' first cue
    dorsal_nicks: int = 0           # sharp nicks on the dorsal trailing edge
    dorsal_hump_cm: float = 0.0     # run06: height of the broad hump the fin sits on (0 = the asset's straight ridge)
    dorsal_chord: float = 1.0       # run06 round 6: blade chord multiplier, growing toward the tip (broad, rounded blades; 1 = asset)
    dorsal_rake_cm: float = 0.0     # run06 round 6: tip raked back along a curve (falcate trailing edge), cm at the tip
    dorsal_round: float = 0.0       # run06 round 6: 0..1, how much the top of the blade is flattened into a rounded / knobby tip
    blade_template: int = -1        # run06 round 7: index of the real-fin outline template of the parametric blade (-1 none)
    blade_H_cm: float = 45.0        # run06 round 7: the blade's height above its base line (rest frame, cm)
    blade_nicks: int = 0            # run06 round 7: trailing-edge nicks on the blade
    blade_thickness: float = 1.0    # run06 round 7: blade / hump thickness multiplier
    pec_chord: float = 1.0          # run06 round 8: pectoral chord multiplier about its centre line (narrower blade; root kept)
    pec_knobs: int = 0              # run06 round 8: extra tubercles on the pectoral leading edge
    pec_knob_cm: float = 0.0        # run06 round 8: their height (cm)
    fluke_nicks: int = 0            # sharp nicks on the fluke trailing edge
    fluke_tip_loss_cm: float = 0.0  # a missing piece of one fluke tip (0 = intact)
    fluke_tip_loss_side: int = 1    # which tip (+1 right, -1 left)
    joint_deg: dict = field(default_factory=dict)   # joint name -> [roll, pitch, yaw] degrees (project axes x, y, z)
    skin_variant_seed: int = 0      # factory.skin colouring/scars seed
    fluke_class: int = 3            # 1 (white) .. 5 (black)
    fluke_pattern_seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Individual":
        return cls(**d)


def run06_variant(ind: "Individual", rng: np.random.Generator) -> "Individual":
    """The run06 realism copy of an individual: same id, texture and body, fins drawn toward what the field photos show
    (many low, stubby, hooked or worn fins, most on a broad hump), longer flippers."""
    from dataclasses import replace as _replace
    if ind.calf:
        return _replace(ind, dorsal_height=float(rng.uniform(0.5, 1.1)), dorsal_hump_cm=float(rng.uniform(0.0, 8.0)))
    kind = rng.choice(["stubby", "medium", "tall"], p=[0.4, 0.35, 0.25])
    height = {"stubby": rng.uniform(0.3, 0.6), "medium": rng.uniform(0.6, 1.1), "tall": rng.uniform(1.1, 1.6)}[str(kind)]
    hooked = rng.uniform() < 0.3
    # the sweep shears the fin in proportion to its height: on a low fin a large sweep is a spike, not a hook
    sweep = float(rng.uniform(20, 40) if hooked else rng.uniform(-10, 20)) * min(float(height), 1.0)
    return _replace(ind, dorsal_height=float(height), dorsal_sweep_cm=sweep,
                    dorsal_edge_cm=float(min(ind.dorsal_edge_cm * 1.5 + rng.uniform(0, 3), 14.0)), dorsal_nicks=int(rng.integers(0, 6)),
                    dorsal_hump_cm=float(rng.uniform(12.0, 32.0)) if rng.uniform() < 0.8 else 0.0,
                    pec_len=float(rng.uniform(1.0, 1.3)))


def sample(rng: np.random.Generator, id: str, calf: bool = False) -> Individual:
    seed = int(rng.integers(1, 2**31 - 1))
    ind = Individual(id=id, seed=seed, calf=calf)
    if calf:
        ind.head_len = float(rng.uniform(0.92, 1.0)); ind.girth = float(rng.uniform(1.08, 1.2)); ind.peduncle = float(rng.uniform(1.05, 1.15))
        ind.fluke_span = float(rng.uniform(0.9, 1.0)); ind.pec_len = float(rng.uniform(0.95, 1.05))
    else:
        ind.head_len = float(rng.uniform(0.92, 1.08)); ind.girth = float(rng.uniform(0.93, 1.07)); ind.peduncle = float(rng.uniform(0.9, 1.1))
        ind.fluke_span = float(rng.uniform(0.85, 1.2)); ind.pec_len = float(rng.uniform(0.9, 1.12))
    # run05: wider ranges on purpose, so fin / fluke landmarks must be found on fins of every height, sweep and wear
    ind.dorsal_height = float(rng.uniform(0.8, 1.3)) if calf else float(rng.uniform(0.5, 2.0)); ind.dorsal_sweep_cm = float(rng.uniform(-25, 40))
    ind.dorsal_edge_cm = float(rng.uniform(0, 10) ** 1.4 / 10 ** 0.4); ind.dorsal_nicks = int(rng.integers(0, 4))
    ind.fluke_sweep_cm = float(rng.uniform(-30, 40)); ind.fluke_notch_cm = float(rng.uniform(0, 30)); ind.fluke_edge_cm = float(rng.uniform(0, 20) ** 1.3 / 20 ** 0.3)
    ind.fluke_nicks = int(rng.integers(0, 7)); ind.fluke_tip_loss_cm = float(rng.uniform(15, 60)) if rng.random() < 0.15 else 0.0; ind.fluke_tip_loss_side = int(rng.choice([-1, 1]))
    deg = {}
    # resting angles around the keypoints: fluke pitch, tail-stock set, head set, pectoral set; small, per joint
    deg["joint62_011"] = [0.0, float(rng.normal(0, 4)), float(rng.normal(0, 1.5))]
    deg["joint61_010"] = [0.0, float(rng.normal(0, 2.5)), float(rng.normal(0, 1.0))]
    deg["joint59_08"] = [float(rng.normal(0, 1.0)), float(rng.normal(0, 1.5)), float(rng.normal(0, 1.0))]
    deg["joint52_01"] = [0.0, float(rng.normal(0, 1.5)), float(rng.normal(0, 1.0))]
    for root, mid in ((PEC_LEFT[0], PEC_LEFT[2]), (PEC_RIGHT[0], PEC_RIGHT[2])):
        deg[root] = [float(rng.normal(0, 5)), float(rng.normal(0, 3)), float(rng.normal(0, 4))]
        deg[mid] = [float(rng.normal(0, 3)), 0.0, float(rng.normal(0, 3))]
    ind.joint_deg = {k: [round(a, 2) for a in v] for k, v in deg.items()}
    ind.skin_variant_seed = int(rng.integers(1, 10**6))
    ind.fluke_class = int(rng.choice([1, 2, 3, 4, 5], p=[0.18, 0.24, 0.24, 0.2, 0.14]))
    ind.fluke_pattern_seed = int(rng.integers(1, 10**6))
    return ind


def load_library(path: Path | None = None) -> dict[str, Individual]:
    """The individual library; WF_INDIVIDUALS=<json> selects another file (e.g. individuals_run04.json to re-label
    a run rendered with an earlier library)."""
    path = Path(os.environ["WF_INDIVIDUALS"]) if (path is None and os.environ.get("WF_INDIVIDUALS")) else (path or LIBRARY)
    if not path.exists():
        return {}
    return {k: Individual.from_dict(v) for k, v in json.loads(path.read_text()).items()}


# ---- geometry: rest-vertex edits in the project frame ----------------------------------------------------
def _smooth(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0 below lo, 1 above hi, smoothstep between."""
    t = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _value_noise_1d(rng: np.random.Generator, n_knots: int, t: np.ndarray) -> np.ndarray:
    knots = rng.uniform(-1, 1, n_knots)
    x = np.linspace(0, 1, n_knots)
    return np.interp(np.clip(t, 0, 1), x, knots)


def _pec_planform(V: np.ndarray, w: np.ndarray, root: np.ndarray, tip: np.ndarray, chord: float, knobs: int, knob_cm: float, seed: int) -> np.ndarray:
    """run06 round 8: one pectoral narrowed about its centre line and given extra leading-edge tubercles (see apply_rest 3f)."""
    V = V.copy()
    axis = tip - root; Lx = float(np.linalg.norm(axis)); axis = axis / max(Lx, 1e-6)
    fin = w > 0.5
    if fin.sum() < 20:
        return V
    rel = V - root
    t = rel @ axis / Lx
    P = rel[fin] - np.outer(rel[fin] @ axis, axis)
    _, _, vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    cdir = vt[0] - (vt[0] @ axis) * axis; cdir = cdir / max(np.linalg.norm(cdir), 1e-6)
    if cdir[0] < 0:                                  # +x = toward the head: the leading edge
        cdir = -cdir
    c = rel @ cdir
    bins = np.linspace(-0.05, 1.3, 55)
    cc = np.full(len(bins), np.nan); hw = np.full(len(bins), np.nan)
    for i, b in enumerate(bins):
        s = fin & (np.abs(t - b) < 0.04)
        if s.sum() > 5:
            cc[i] = 0.5 * (c[s].max() + c[s].min()); hw[i] = 0.5 * (c[s].max() - c[s].min())
    ok = ~np.isnan(cc)
    if ok.sum() < 3:
        return V
    c0 = np.interp(t, bins[ok], cc[ok]); h0 = np.interp(t, bins[ok], hw[ok])
    ramp = _smooth(t, 0.12, 0.35) * w
    dc = (c - c0) * (chord - 1.0) * ramp
    if knobs:
        rng = np.random.default_rng(seed)
        t_end = float(np.percentile(t[fin], 99))
        bump = np.zeros_like(t)
        for tc in np.sort(rng.uniform(0.18, 0.92, int(knobs))) * t_end:
            bump += np.exp(-((t - tc) / (0.022 * t_end)) ** 2) * (1.0 - 0.5 * tc / t_end) * rng.uniform(0.7, 1.2)
        front = np.clip((c - c0) / np.maximum(h0, 1.0), 0.0, 1.0) ** 3
        dc = dc + knob_cm * bump * front * w
    return V + np.outer(dc, cdir)


def apply_rest(V: np.ndarray, ind: Individual, joints: dict, regions: dict) -> np.ndarray:
    """Edited rest vertices (project frame cm) for one individual. ``joints`` = rest joint positions,
    ``regions`` = {"pec_left": w, "pec_right": w} (per-vertex skin weight on the pectoral chains)."""
    V = np.array(V, float, copy=True)
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    # spine line z(x) from the spine joints, for scaling cross-sections about it
    sp = np.array([joints[n] for n in SPINE])
    order = np.argsort(sp[:, 0])
    z_spine = np.interp(x, sp[order, 0], sp[order, 2])
    # 1. head length: everything ahead of the jaw joint stretches along x from it
    xj = joints["joint52_01"][0]
    w = _smooth(x, xj - 60, xj + 60)
    V[:, 0] = x + w * (ind.head_len - 1.0) * (x - xj)
    x = V[:, 0]
    # 2. girth (whole body, fading out on the flukes) and peduncle (tail stock) about the spine
    x_fluke0 = joints["joint60_09"][0]
    g = 1.0 + (ind.girth - 1.0) * (1 - _smooth(-x, -x_fluke0, -x_fluke0 + 90))
    # tail stock only: full behind joint59, gone by joint56 (the dorsal fin sits between joint56 and joint55)
    if LEGACY_DORSAL:                                   # run04-era fade (reached into the fin region)
        pw = _smooth(x, joints["joint61_010"][0], joints["joint59_08"][0]) * (1 - _smooth(x, joints["joint56_05"][0], joints["joint54_03"][0]))
    else:
        pw = _smooth(x, joints["joint61_010"][0], joints["joint59_08"][0]) * (1 - _smooth(x, joints["joint57_06"][0], joints["joint56_05"][0]))
    g = g * (1.0 + (ind.peduncle - 1.0) * pw)
    V[:, 1] = y * g
    V[:, 2] = z_spine + (z - z_spine) * g
    y, z = V[:, 1], V[:, 2]
    # 3. flukes: span, sweep, notch, trailing-edge wear.  Fluke weight: behind joint60, off the midline
    x_te = x.min()
    fw = _smooth(-x, -joints["joint60_09"][0], -joints["joint60_09"][0] + 70)
    span_max = np.abs(y[fw > 0.5]).max() if (fw > 0.5).any() else 350.0
    V[:, 1] = y * (1.0 + (ind.fluke_span - 1.0) * fw)
    y = V[:, 1]
    frac = np.clip(np.abs(y) / span_max, 0, 1)
    V[:, 0] = x - fw * ind.fluke_sweep_cm * frac ** 2
    x = V[:, 0]
    # trailing edge band: within 40 cm of the hindmost point, scaled by how close to the edge
    te = _smooth(-x, -x_te - 40, -x_te - 5) * fw
    notch = ind.fluke_notch_cm * np.exp(-(y / 45.0) ** 2)
    rng = np.random.default_rng(ind.seed)
    wear = ind.fluke_edge_cm * _value_noise_1d(rng, 28, (y + span_max) / (2 * span_max))
    nick = np.zeros_like(x)
    for _ in range(int(ind.fluke_nicks if ind.fluke_nicks else rng.integers(0, 4))):          # sharp nicks (orca rake / entanglement)
        yc = float(rng.uniform(-span_max * 0.9, span_max * 0.9)); wd = float(rng.uniform(6, 18)); dp = float(rng.uniform(4, 14))
        nick += dp * np.exp(-((y - yc) / wd) ** 2)
    if ind.fluke_tip_loss_cm > 0:                      # a missing tip piece: the trailing edge pulled forward near one tip
        tip_w = np.clip((np.abs(y) - (span_max - ind.fluke_tip_loss_cm * 1.5)) / (ind.fluke_tip_loss_cm * 1.5), 0, 1) * (np.sign(y) == ind.fluke_tip_loss_side)
        nick += ind.fluke_tip_loss_cm * tip_w ** 2
    V[:, 0] = x + te * (notch + np.clip(wear, -ind.fluke_edge_cm, ind.fluke_edge_cm) + nick)
    # 3b. dorsal fin: height, sweep and trailing-edge wear on the fin proper. Membership comes from the canonical mesh
    # (WhaleModel.dorsal_region: height above the back-junction -> front-junction base line; vertex indices are shared
    # with the rig's rest mesh), directions from the rest frame: "up" = base midpoint -> tip, "along" = back junction ->
    # front junction (towards the head). The rig's rest pose is a curled, pitched whale, so neither z nor x would do.
    if LEGACY_DORSAL:
        # The run04-era block, kept verbatim so that run04 (rendered 2026-10-14..16 with it) can be re-labelled with
        # geometry that matches its frames: window behind the fin, height above a flank-percentile back line, z scaling.
        if abs(ind.dorsal_height - 1.0) > 1e-6 or abs(ind.dorsal_sweep_cm) > 1e-6:
            xr = (x > joints["joint59_08"][0] - 60) & (x < joints["joint56_05"][0] + 40)
            side = xr & (np.abs(y) > 25) & (np.abs(y) < 60)
            if side.any():
                bins = np.linspace(x[xr].min(), x[xr].max(), 12)
                centres = (bins[:-1] + bins[1:]) / 2
                back = np.array([np.percentile(z[side & (x >= a) & (x < b)], 90) if (side & (x >= a) & (x < b)).any() else np.nan for a, b in zip(bins[:-1], bins[1:])])
                okb = ~np.isnan(back)
                back_line = np.interp(x, centres[okb], back[okb])
                h = z - back_line
                wd = _smooth(h, 4.0, 18.0) * xr * (np.abs(y) < 30)
                hmax = max(float(np.clip(h, 0, None).max()), 1.0)
                V[:, 2] = z + (ind.dorsal_height - 1.0) * np.clip(h, 0, None) * wd
                V[:, 0] = x - ind.dorsal_sweep_cm * wd * np.clip(h, 0, None) / hmax
                x, z = V[:, 0], V[:, 2]
    elif "dorsal_w" in regions and (abs(ind.dorsal_height - 1.0) > 1e-6 or abs(ind.dorsal_sweep_cm) > 1e-6 or ind.dorsal_edge_cm > 0 or ind.dorsal_nicks > 0
                                    or abs(getattr(ind, "dorsal_chord", 1.0) - 1.0) > 1e-6 or getattr(ind, "dorsal_rake_cm", 0.0) or getattr(ind, "dorsal_round", 0.0)):
        wd = regions["dorsal_w"]; tip_i, back_i, front_i = regions["dorsal_idx"]
        scale = float(np.linalg.norm(V[front_i] - V[back_i])) / max(float(regions["dorsal_chord"]), 1e-6)
        hc = regions["dorsal_h"] * scale                                      # height above the base, rest-frame cm
        hmax = max(float(hc[tip_i]), 1.0)
        up = V[tip_i] - (V[back_i] + V[front_i]) / 2.0; up = up / max(np.linalg.norm(up), 1e-6)
        along = V[front_i] - V[back_i]; along = along / max(np.linalg.norm(along), 1e-6)
        V += np.outer((ind.dorsal_height - 1.0) * hc * wd, up)
        V -= np.outer(ind.dorsal_sweep_cm * wd * hc / hmax, along)
        chord, rake, rnd = getattr(ind, "dorsal_chord", 1.0), getattr(ind, "dorsal_rake_cm", 0.0), getattr(ind, "dorsal_round", 0.0)
        if abs(chord - 1.0) > 1e-6 or rake or rnd:
            # run06 round 6: the blade's outline. hz = height fraction (0 base, 1 tip) on the edited fin. The chord widens
            # about the blade's centre line by up to `chord` at the top (a broad blade instead of a spike), the top is raked
            # back along a curve (falcate trailing edge, tip over the rear), and the upper part is flattened (rounded / knobby tip)
            hz = np.clip(hc / hmax, 0.0, 1.0)
            fin = wd > 0.2
            s_al = (V - V[back_i]) @ along
            bins = np.linspace(0, 1, 9)
            cen = [np.median(s_al[fin & (np.abs(hz - t) < 0.08)]) if (fin & (np.abs(hz - t) < 0.08)).any() else np.median(s_al[fin]) for t in bins]
            sc = np.interp(hz, bins, cen)
            widen = 1.0 + (chord - 1.0) * np.sqrt(hz) * (1.0 - 0.92 * hz ** 2.5)        # broadest at mid height, tapering to a narrower tip
            # only the blade's own skin widens: weight by membership squared and fade out beyond the blade's own chord
            # (hump / back vertices far from the centre line would otherwise be thrown out as spikes)
            half = np.interp(hz, bins, [np.percentile(np.abs(s_al - sc)[fin & (np.abs(hz - t) < 0.08)], 90)
                                         if (fin & (np.abs(hz - t) < 0.08)).any() else 5.0 for t in bins])
            keep = np.clip(1.5 - np.abs(s_al - sc) / np.maximum(half, 1.0), 0.0, 1.0) * wd ** 2
            V += np.outer((s_al - sc) * (widen - 1.0) * keep, along)
            V -= np.outer(rake * hz ** 2 * wd, along)                                  # back = - along
            hcur = hc * ind.dorsal_height
            V -= np.outer(rnd * 0.45 * hcur * _smooth(hz, 0.55, 1.0) ** 1.5 * wd, up)
        # trailing-edge wear: vertices on the fin's rear side (behind the fin's centre at that height) move forward by a
        # height-indexed scallop profile plus sharp nicks, like the fluke's edge
        if (ind.dorsal_edge_cm > 0 or ind.dorsal_nicks > 0) and (wd > 0.2).any():
            fin = wd > 0.2; hz = hc / hmax
            s_along = (V - V[back_i]) @ along
            sc = np.interp(hz, np.linspace(0, 1, 8), [np.median(s_along[fin & (np.abs(hz - t) < 0.1)]) if (fin & (np.abs(hz - t) < 0.1)).any() else np.median(s_along[fin]) for t in np.linspace(0, 1, 8)])
            rear = _smooth(sc - s_along, 0.0, 12.0) * wd
            rng_d = np.random.default_rng(ind.seed + 7)
            scallop = ind.dorsal_edge_cm * _value_noise_1d(rng_d, 10, hz)
            nick_d = np.zeros(len(V))
            for _ in range(int(ind.dorsal_nicks)):
                hcn = float(rng_d.uniform(0.2, 0.95)); wn = float(rng_d.uniform(0.03, 0.08)); dp = float(rng_d.uniform(3, 9))
                nick_d += dp * np.exp(-((hz - hcn) / wn) ** 2)
            V += np.outer(rear * (np.clip(scallop, -ind.dorsal_edge_cm, ind.dorsal_edge_cm) + nick_d), along)
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
    # 3c. run06: the hump under and ahead of the fin (real fins sit on a broad hump; the asset's ridge runs straight on).
    # It rises from just behind the back base, carries the fin, and flattens into the back at the front base (the
    # researcher's dorsal_fin_front), within about one fin-region length either side of the midline.
    if getattr(ind, "dorsal_hump_cm", 0.0) > 0 and "dorsal_idx" in regions and "dorsal_front_base" in regions:
        tip_i, back_i, front_i = regions["dorsal_idx"]
        fb = regions["dorsal_front_base"]
        p_fb = V[fb].mean(0) if isinstance(fb, (list, tuple)) else V[fb]
        scale = float(np.linalg.norm(V[front_i] - V[back_i])) / max(float(regions["dorsal_chord"]), 1e-6)
        along = p_fb - V[back_i]; L = max(float(np.linalg.norm(along)), 1.0); along = along / L
        up = V[tip_i] - V[back_i]; up = up - (up @ along) * along; up = up / max(np.linalg.norm(up), 1e-6)
        side = np.cross(along, up)
        rel = V - V[back_i]
        s_al, h_up, lat = rel @ along / L, rel @ up / L, rel @ side / L
        prof = _smooth(s_al, -0.45, 0.0) * (1.0 - _smooth(s_al, 0.45, 1.05))            # rises behind the back base, gone at the front base
        lat_w = np.clip(np.cos(np.clip(np.abs(lat) / 1.0, 0, 1) * np.pi / 2), 0, None) ** 2
        top_w = _smooth(h_up, -1.4, -0.4)                                               # the top of the back only
        V = V + np.outer(ind.dorsal_hump_cm * scale * prof * lat_w * top_w, up)
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
    # 3d. run06 round 7 (parametric blade only, WhaleModel.enable_blade): the asset's own blade and any wear on it are
    # pressed down onto a smooth back (a quadratic fit to the back around it), so nothing of it shows beside the new blade
    if regions.get("dorsal_flatten") and "dorsal_idx" in regions:
        wd = np.asarray(regions["dorsal_w"]); tip_i, back_i, front_i = regions["dorsal_idx"]
        c0 = (V[back_i] + V[front_i]) / 2.0
        up = V[tip_i] - c0; up = up / max(np.linalg.norm(up), 1e-6)
        along = V[front_i] - V[back_i]; along = along - (along @ up) * up; along = along / max(np.linalg.norm(along), 1e-6)
        side = np.cross(along, up)
        rel = V - c0; s_, h_, l_ = rel @ along, rel @ up, rel @ side
        X = np.stack([np.ones(len(V)), s_, l_, s_ ** 2, l_ ** 2, s_ * l_], 1)
        near = (np.abs(s_) < 150.0) & (np.abs(l_) < 45.0) & (h_ > -40.0) & (wd < 0.02)
        if near.sum() > 20:
            coef = np.linalg.lstsq(X[near], h_[near], rcond=None)[0]
            excess = np.clip(h_ - X @ coef, 0.0, None)
            # (round 8: WhaleModel.enable_blade also removes the old blade's faces and closes the hole with a fan, so only
            # the ring around it and the fan's centre are used; projecting them onto the fitted back leaves no crater)
            V = V - np.outer(excess * np.clip(wd * 4.0, 0.0, 1.0), up)
            x, y, z = V[:, 0], V[:, 1], V[:, 2]
    # 3e. run06 round 8: the knobs along the ridge between the fin and the flukes, toned down to subtle knuckles. The
    # rest pose is curled, so heights are taken across the spine polyline (joints 55..60) in the x-z plane: per 4 cm of
    # arc length the ridge top (|y| < 12 cm), a robust smooth line through it, and the part of each bump above that line
    # scaled by (1 - ridge_soften) on the vertices near the top.
    if regions.get("ridge_soften") and "dorsal_idx" in regions:
        k_soft = float(regions["ridge_soften"])
        P = np.array([joints[n] for n in SPINE[4:10]], float)[:, [0, 2]]          # joint55_04 .. joint60_09 (x, z)
        seg = np.diff(P, axis=0); L = np.linalg.norm(seg, axis=1); cum = np.concatenate([[0.0], np.cumsum(L)])
        Q = V[:, [0, 2]]
        best = np.full(len(V), np.inf); s_arc = np.zeros(len(V)); h_up = np.zeros(len(V)); nrm = np.zeros((len(V), 2))
        for j in range(len(seg)):
            tdir = seg[j] / max(L[j], 1e-6); n = np.array([-tdir[1], tdir[0]])
            if n[1] < 0:
                n = -n
            u = np.clip((Q - P[j]) @ tdir, 0.0, L[j]); foot = P[j] + np.outer(u, tdir)
            d = np.linalg.norm(Q - foot, axis=1)
            take = d < best
            best[take] = d[take]; s_arc[take] = cum[j] + u[take]; h_up[take] = ((Q - foot) @ n)[take]; nrm[take] = n
        fin_s = float(s_arc[regions["dorsal_idx"][1]])
        s_lo, s_hi = fin_s + 15.0, cum[-1] - 20.0
        if s_hi > s_lo + 60.0:
            ss = np.arange(s_lo, s_hi, 4.0)
            topm = (np.abs(y) < 12.0) & (h_up > 0)
            ht = np.array([h_up[topm & (np.abs(s_arc - q) < 2.5)].max() if (topm & (np.abs(s_arc - q) < 2.5)).any() else np.nan for q in ss])
            okr = ~np.isnan(ht)
            if okr.sum() > 10:
                sr, hr = ss[okr], ht[okr]
                med = np.array([np.median(hr[max(0, j - 6): j + 7]) for j in range(len(hr))])
                line = np.convolve(np.pad(med, 3, mode="edge"), np.ones(7) / 7.0, mode="valid")
                bump = np.clip(hr - line, 0.0, None)
                line_v = np.interp(s_arc, sr, line); bump_v = np.interp(s_arc, sr, bump)
                inside = (s_arc > s_lo) & (s_arc < s_hi) & (np.abs(y) < 30.0)
                fall = np.clip((h_up - (line_v - 15.0)) / 15.0, 0.0, 1.0) * np.exp(-(y / 14.0) ** 2) * inside
                dz = k_soft * bump_v * fall
                V[:, 0] -= dz * nrm[:, 0]; V[:, 2] -= dz * nrm[:, 1]
                x, y, z = V[:, 0], V[:, 1], V[:, 2]
    # 3f. run06 round 8: pectoral planform: a narrower blade (chord about the fin's own centre line, the root and the
    # shoulder keep their width) and a few extra tubercles on the leading edge (the asset has small ones already)
    if (abs(getattr(ind, "pec_chord", 1.0) - 1.0) > 1e-6 or getattr(ind, "pec_knobs", 0)) and regions.get("pec_left") is not None:
        for side, root_name, tip_name, k_seed in (("pec_left", PEC_LEFT[0], PEC_LEFT[-1], 1), ("pec_right", PEC_RIGHT[0], PEC_RIGHT[-1], 2)):
            w = regions.get(side)
            if w is None:
                continue
            V = _pec_planform(V, np.asarray(w), np.array(joints[root_name], float), np.array(joints[tip_name], float),
                              float(ind.pec_chord), int(ind.pec_knobs), float(ind.pec_knob_cm), ind.seed * 7 + k_seed)
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
    # 4. pectoral length about the fin root along the fin axis
    for side, root_name, tip_name in (("pec_left", PEC_LEFT[0], PEC_LEFT[-1]), ("pec_right", PEC_RIGHT[0], PEC_RIGHT[-1])):
        w = regions.get(side)                       # per-vertex chain skin weight (smooth), or None for meshes without fins
        if w is None or abs(ind.pec_len - 1.0) < 1e-6:
            continue
        root = np.array(joints[root_name]); axis = np.array(joints[tip_name]) - root; axis /= np.linalg.norm(axis)
        along = (V - root) @ axis
        V += np.outer(w * np.clip(along, 0, None) * (ind.pec_len - 1.0), axis)
    return V


def joint_offsets_model(ind: Individual) -> dict:
    """{joint name: 3x3 rotation in the glTF model frame} from the individual's project-axes degrees."""
    P = MODEL_TO_PROJECT / 100.0                       # orthonormal (det -1); conjugation maps rotations between the frames
    out = {}
    for name, (roll, pitch, yaw) in ind.joint_deg.items():
        if abs(roll) + abs(pitch) + abs(yaw) < 1e-9:
            continue
        r, p, yw = np.radians([roll, pitch, yaw])
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(yw), -np.sin(yw), 0], [np.sin(yw), np.cos(yw), 0], [0, 0, 1]])
        R_proj = Rz @ Ry @ Rx
        out[name] = P.T @ R_proj @ P
    return out


# ---- texture: skin variant + fluke underside class --------------------------------------------------------
def _fbm(rng: np.random.Generator, shape: tuple, octaves: int = 5, base: int = 6) -> np.ndarray:
    from PIL import Image
    out = np.zeros(shape, np.float32); amp = 1.0; total = 0.0
    for o in range(octaves):
        n = base * 2 ** o
        small = rng.uniform(0, 1, (n, n)).astype(np.float32)
        img = Image.fromarray(small).resize((shape[1], shape[0]), Image.BICUBIC)
        out += amp * np.asarray(img); total += amp; amp *= 0.55
    return out / total


def fluke_underside_mask(rig, size: int) -> tuple[np.ndarray, np.ndarray]:
    """UV-space mask of the fluke underside and its leading-edge direction field (u, v image coords).
    Returns (mask HxW bool, grad HxW float in [0,1]: 0 at the trailing edge, 1 at the leading edge)."""
    from PIL import Image, ImageDraw
    V = rig.rest; x, y = V[:, 0], V[:, 1]; F = rig.faces; uv = rig.uv
    fl = x < -330                                             # the whole fluke underside, centre included
    fv = V[F]; n = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
    nz = -n[:, 2] / (np.linalg.norm(n, axis=1) + 1e-9)       # MODEL_TO_PROJECT mirrors the winding: flip
    under = fl[F].all(1) & (nz < -0.3)
    im = Image.new("L", (size, size), 0); d = ImageDraw.Draw(im)
    for tri in F[under]:
        d.polygon([(float(uv[i, 0] * size), float(uv[i, 1] * size)) for i in tri], fill=255)
    mask = np.asarray(im) > 0
    # leading-edge direction: regress project x on (u, v) over the underside vertices
    vid = np.unique(F[under])
    A = np.c_[uv[vid], np.ones(len(vid))]
    coef, *_ = np.linalg.lstsq(A, x[vid], rcond=None)
    uu, vv = np.meshgrid((np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size)
    xf = coef[0] * uu + coef[1] * vv + coef[2]
    lo, hi = x[vid].min(), x[vid].max()
    grad = np.clip((xf - lo) / max(hi - lo, 1e-6), 0, 1)
    return mask, grad.astype(np.float32)


def paint_fluke(tex: np.ndarray, mask: np.ndarray, grad: np.ndarray, fluke_class: int, seed: int) -> np.ndarray:
    """Katona class 1..5 underside: white/black blotches from thresholded noise biased toward pigment at the
    leading edge and the centre (as on real flukes), soft boundaries, scratches of the opposite tone."""
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed)
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = (slice(y0, y1), slice(x0, x1))
    m = mask[sub]; g = grad[sub]
    noise = _fbm(rng, m.shape, octaves=5, base=int(rng.integers(3, 6)))
    # pigment field (high = white): darker toward the leading edge (grad -> 1) with a per-individual mix;
    # the class sets the white fraction of the underside
    bias = -rng.uniform(0.2, 0.6) * g
    field = noise + bias
    white_frac = FLUKE_CLASS_WHITE[int(fluke_class)]
    thr = np.quantile(field[m], 1 - white_frac)
    k = 10.0 / max(field[m].std(), 1e-3)
    white = 1.0 / (1.0 + np.exp(-(field - thr) * k))    # 1 where white
    white = np.asarray(Image.fromarray((white * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(W / 1400.0))) / 255.0
    out = tex.astype(np.float32).copy()
    region = out[sub]
    white_col = np.array([rng.uniform(218, 240)] * 3, np.float32) * np.array([1.0, 0.985, 0.96], np.float32)
    black_col = np.array([rng.uniform(22, 40)] * 3, np.float32) * np.array([0.95, 0.97, 1.05], np.float32)
    # keep the base texture's shading detail: modulate by its luminance deviation
    base = region.mean(2, keepdims=True)
    detail = (base - np.median(base[m])) * 0.25
    colour = white[..., None] * white_col + (1 - white[..., None]) * black_col + detail
    # scratches: thin lines of the opposite tone, a few per fluke
    scratch = np.zeros(m.shape, np.float32)
    from PIL import ImageDraw
    sim = Image.new("L", (m.shape[1], m.shape[0]), 0); sd = ImageDraw.Draw(sim)
    for _ in range(int(rng.integers(3, 12))):
        cx, cy = rng.uniform(0, m.shape[1]), rng.uniform(0, m.shape[0]); ang = rng.uniform(0, np.pi); L = rng.uniform(0.05, 0.35) * m.shape[1]
        sd.line([(cx - np.cos(ang) * L, cy - np.sin(ang) * L), (cx + np.cos(ang) * L, cy + np.sin(ang) * L)], fill=255, width=max(1, int(W / 1500)))
    scratch = np.asarray(sim.filter(ImageFilter.GaussianBlur(W / 2000.0))).astype(np.float32) / 255.0
    sc = np.clip(scratch, 0, 1)[..., None]
    colour = colour * (1 - 0.7 * sc) + 0.7 * sc * ((1 - white[..., None]) * white_col + white[..., None] * black_col)
    region[m] = colour[m]
    out[sub] = region
    return np.clip(out, 0, 255).astype(np.uint8)


def write_texture(ind: Individual, rig, out_dir: Path = SKINS_DIR, base_texture: Path = BASE_TEXTURE) -> Path:
    from PIL import Image
    base = np.asarray(Image.open(base_texture).convert("RGB"))
    img, recipe = skin_mod.make_variant(base, ind.skin_variant_seed)
    mask, grad = fluke_underside_mask(rig, base.shape[0])
    img = paint_fluke(img, mask, grad, ind.fluke_class, ind.fluke_pattern_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"T_Humpback_BaseColor_{ind.id}.png"
    tmp = path.with_suffix(".tmp.png")                    # atomic: a render may import the file at any moment
    Image.fromarray(img).save(tmp)
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=12, help="adult individuals")
    parser.add_argument("--calves", type=int, default=4, help="calf individuals (ids c01..)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=LIBRARY)
    parser.add_argument("--no-textures", action="store_true")
    args = parser.parse_args(argv)
    from factory.skinned import Rig
    from factory.whalemodel import HUMPBACK4K_GLTF
    rng = np.random.default_rng(args.seed)
    lib = {}
    for k in range(args.count):
        ind = sample(rng, f"i{k + 1:02d}"); lib[ind.id] = ind
    for k in range(args.calves):
        ind = sample(rng, f"c{k + 1:02d}", calf=True); lib[ind.id] = ind
    args.out.write_text(json.dumps({k: v.to_dict() for k, v in lib.items()}, indent=1))
    print(f"{len(lib)} individuals -> {args.out}")
    if not args.no_textures:
        rig = Rig.load(HUMPBACK4K_GLTF)
        for ind in lib.values():
            print(write_texture(ind, rig), "fluke class", ind.fluke_class)
    return 0


if __name__ == "__main__":
    sys.exit(main())
