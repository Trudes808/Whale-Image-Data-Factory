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
    joint_deg: dict = field(default_factory=dict)   # joint name -> [roll, pitch, yaw] degrees (project axes x, y, z)
    skin_variant_seed: int = 0      # factory.skin colouring/scars seed
    fluke_class: int = 3            # 1 (white) .. 5 (black)
    fluke_pattern_seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Individual":
        return cls(**d)


def sample(rng: np.random.Generator, id: str, calf: bool = False) -> Individual:
    seed = int(rng.integers(1, 2**31 - 1))
    ind = Individual(id=id, seed=seed, calf=calf)
    if calf:
        ind.head_len = float(rng.uniform(0.92, 1.0)); ind.girth = float(rng.uniform(1.08, 1.2)); ind.peduncle = float(rng.uniform(1.05, 1.15))
        ind.fluke_span = float(rng.uniform(0.9, 1.0)); ind.pec_len = float(rng.uniform(0.95, 1.05))
    else:
        ind.head_len = float(rng.uniform(0.92, 1.08)); ind.girth = float(rng.uniform(0.93, 1.07)); ind.peduncle = float(rng.uniform(0.9, 1.1))
        ind.fluke_span = float(rng.uniform(0.9, 1.12)); ind.pec_len = float(rng.uniform(0.92, 1.1))
    ind.dorsal_height = float(rng.uniform(0.8, 1.2)) if calf else float(rng.uniform(0.7, 1.6)); ind.dorsal_sweep_cm = float(rng.uniform(-15, 15))
    ind.fluke_sweep_cm = float(rng.uniform(-20, 25)); ind.fluke_notch_cm = float(rng.uniform(0, 18)); ind.fluke_edge_cm = float(rng.uniform(0, 12) ** 1.3 / 12 ** 0.3)
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


def load_library(path: Path = LIBRARY) -> dict[str, Individual]:
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
    pw = _smooth(x, joints["joint61_010"][0], joints["joint59_08"][0]) * (1 - _smooth(x, joints["joint56_05"][0], joints["joint54_03"][0]))
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
    for _ in range(int(rng.integers(0, 4))):          # a few sharp nicks (orca rake / entanglement)
        yc = float(rng.uniform(-span_max * 0.9, span_max * 0.9)); wd = float(rng.uniform(6, 18)); dp = float(rng.uniform(4, 14))
        nick += dp * np.exp(-((y - yc) / wd) ** 2)
    V[:, 0] = x + te * (notch + np.clip(wear, -ind.fluke_edge_cm, ind.fluke_edge_cm) + nick)
    # 3b. dorsal fin: height about the local back line and tip sweep, on the thin midline part of the fin only
    if abs(ind.dorsal_height - 1.0) > 1e-6 or abs(ind.dorsal_sweep_cm) > 1e-6:
        xr = (x > joints["joint59_08"][0] - 60) & (x < joints["joint56_05"][0] + 40)      # the fin's stretch of back
        side = xr & (np.abs(y) > 25) & (np.abs(y) < 60)
        if side.any():
            bins = np.linspace(x[xr].min(), x[xr].max(), 12)
            centres = (bins[:-1] + bins[1:]) / 2
            back = np.array([np.percentile(z[side & (x >= a) & (x < b)], 90) if (side & (x >= a) & (x < b)).any() else np.nan for a, b in zip(bins[:-1], bins[1:])])
            okb = ~np.isnan(back)
            back_line = np.interp(x, centres[okb], back[okb])
            h = z - back_line
            wd = _smooth(h, 4.0, 18.0) * xr * (np.abs(y) < 30)
            V[:, 2] = z + (ind.dorsal_height - 1.0) * np.clip(h, 0, None) * wd
            V[:, 0] = x - ind.dorsal_sweep_cm * wd * np.clip(h, 0, None) / max(float(np.clip(h, 0, None).max()), 1.0)
            x, z = V[:, 0], V[:, 2]
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
