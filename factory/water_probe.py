"""Independent realism measure for water: frozen DINOv3 ViT-B/16 features of water-only patches (224 px windows at the
realism tool's frame scale, long side 1280), real photos against renders.

    PYTHONPATH=<factory>:/home/sat3737/NOAA_whales/Dinov3-WhaleID .venv/bin/python -m factory.water_probe \
        --split train --ref h32_tw_breeze [--ref other_tag ...]

Reports, per render set: a cross-validated linear-probe AUC real vs render (logistic regression on CLS + mean patch
token, folds hold out whole real days and whole render frames / batches), the same probe between two random halves of
the real days (the floor: what day-to-day differences alone give), and the Frechet distance (64 PCs of real
features) real half A -> renders over real half A -> real half B (ratio 1 = as close as other real days are).
Real patches come only from the training days (--split train, for tuning) or the validation days (--split val, read
once at the end); evaluation days are refused. Patch features are cached under /var/tmp/sat3737-parts-cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RG = Path("/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/realism_gap")
CACHE_IN = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/realism_gap/cache")
OUT = Path("/var/tmp/sat3737-parts-cache/water_engine/probe")
FRAMES = Path("/var/tmp/sat3737-parts-cache/water_engine/frames")
SPLITS = Path("/home/sat3737/NOAA_whales/Dinov3-WhaleID/model_fine_tuning/splits/days.json")
EVAL_DAYS = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}
P = 224          # patch side at the 1280 frame scale


def _rg():
    if str(RG) not in sys.path:
        sys.path.insert(0, str(RG))
    import common
    return common


def _windows(region_ok: np.ndarray, k: int, rng, side: int, tries: int = 60):
    """Up to k top-left corners of side x side windows fully inside region_ok (H, W bool)."""
    import cv2
    H, W = region_ok.shape
    if H < side or W < side:
        return []
    ii = cv2.integral(region_ok.astype(np.uint8))
    out = []
    for _ in range(tries):
        y = int(rng.integers(0, H - side + 1)); x = int(rng.integers(0, W - side + 1))
        s = ii[y + side, x + side] - ii[y, x + side] - ii[y + side, x] + ii[y, x]
        if s >= side * side * 0.995 and all(abs(y - a) > side // 2 or abs(x - b) > side // 2 for a, b in out):
            out.append((y, x))
            if len(out) >= k:
                break
    return out


def _to1280(rgb):
    import cv2
    H, W = rgb.shape[:2]
    s = 1280 / max(H, W)
    return cv2.resize(rgb, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)


def real_patches(split: str, per: int = 2, seed: int = 0):
    import cv2
    common = _rg()
    days = set(json.loads(SPLITS.read_text())[split])
    assert not days & EVAL_DAYS
    samples = json.loads((CACHE_IN / "real_v1" / "samples.json").read_text())
    rng = np.random.default_rng(seed)
    out = []
    for i, s in enumerate(samples):
        if s.get("day") not in days or s.get("day") in EVAL_DAYS:
            continue
        flab = cv2.imread(str(CACHE_IN / "real_v1" / "thumbs" / f"{i}_flab.png"), cv2.IMREAD_GRAYSCALE)
        if flab is None:
            continue
        img = cv2.cvtColor(cv2.imread(s["img"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        f = _to1280(img)
        lab = cv2.resize(flab, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
        for (y, x) in _windows(lab == 3, per, rng, P):
            out.append((f[y:y + P, x:x + P], s["day"], f"real|{i}"))
    return out


def run05_patches(per: int = 2, seed: int = 0, limit: int = 900):
    import cv2
    name = "render_run05_dataset_e5b544cc"
    samples = json.loads((CACHE_IN / name / "samples.json").read_text())
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(samples))[:limit]:
        s = samples[i]
        flab = cv2.imread(str(CACHE_IN / name / "thumbs" / f"{i}_flab.png"), cv2.IMREAD_GRAYSCALE)
        if flab is None:
            continue
        f = _to1280(cv2.cvtColor(cv2.imread(s["img"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        lab = cv2.resize(flab, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
        for (y, x) in _windows(lab == 3, per, rng, P):
            out.append((f[y:y + P, x:x + P], s.get("group", "?"), f"run05|{i}"))
    return out


def ref_patches(tag: str, per: int = 4, seed: int = 0, limit: int = 0):
    """Water-only windows of reference frames (or plates): rows below the horizon + 6 px."""
    import cv2
    d = FRAMES / tag if (FRAMES / tag).exists() else Path(tag)
    rng = np.random.default_rng(seed)
    out = []
    files = sorted(d.glob("*.jpg"))
    for jp in (files[:limit] if limit else files):
        f = _to1280(cv2.cvtColor(cv2.imread(str(jp), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        H, W = f.shape[:2]
        ok = np.ones((H, W), bool)
        hz = None
        aux = jp.with_name(jp.stem + "_aux.npz")
        js = jp.with_suffix(".json")
        if aux.exists():
            hz = np.load(aux)["horizon"]
        elif js.exists():
            m = json.loads(js.read_text())
            hz = None if m.get("horizon_row_max") is None else np.asarray([m["horizon_row_max"]], float)
        if hz is not None and (np.asarray(hz) >= 0).any():
            top = int(np.max(hz) * (H / 1707.0)) + 6           # horizon rows are in the 2560 x 1707 frame
            ok[:max(top, 0)] = False
        for (y, x) in _windows(ok, per, rng, P):
            out.append((f[y:y + P, x:x + P], f"{tag}|{jp.stem}", f"{tag}|{jp.stem}"))
    return out


_MODEL = None


def features(patches, batch: int = 64):
    import torch
    global _MODEL
    sys.path.insert(0, "/home/sat3737/NOAA_whales/Dinov3-WhaleID")
    from model_fine_tuning.common.backbone import IMAGENET_MEAN, IMAGENET_STD, load_backbone
    from factory.waterref import pick_device
    dev = pick_device(3.0)
    if _MODEL is None:
        _MODEL = load_backbone("vitb16").to(dev).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    out = []
    for b0 in range(0, len(patches), batch):
        x = torch.from_numpy(np.stack([p[0] for p in patches[b0:b0 + batch]])).to(dev).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            (pt, cls), = _MODEL.get_intermediate_layers((x - mean) / std, n=1, reshape=True, return_class_token=True, norm=True)
        out.append(torch.cat([cls.float(), pt.float().mean((2, 3))], 1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def cached(name: str, make):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return z["F"], list(z["G"]), z["thumbs"]
    patches = make()
    F = features(patches)
    G = np.array([p[1] for p in patches])
    th = np.stack([p[0][::4, ::4] for p in patches]) if patches else np.zeros((0, 56, 56, 3), np.uint8)
    np.savez(p, F=F, G=G, thumbs=th)
    return F, list(G), th


def probe(Fr, Gr, Fx, Gx, seed=0):
    if str(RG) not in sys.path:
        sys.path.insert(0, str(RG))
    import scorecard as SC
    X = np.concatenate([Fr, Fx]); y = np.r_[np.zeros(len(Fr)), np.ones(len(Fx))]
    groups = np.array([f"r|{g}" for g in Gr] + [f"x|{g}" for g in Gx])
    folds = SC.fold_of_groups(groups, 5, seed)
    res, _ = SC.image_level(X, y, folds)
    return res


def floor(Fr, Gr, seed=0):
    days = sorted(set(Gr)); rng = np.random.default_rng(seed)
    half = set(rng.permutation(days)[: len(days) // 2].tolist())
    a = np.array([g in half for g in Gr])
    return probe(Fr[~a], [g for g, m in zip(Gr, a) if not m], Fr[a], [g for g, m in zip(Gr, a) if m], seed), a


def report(split: str, refs: list[str], include_run05: bool = True) -> str:
    if str(RG) not in sys.path:
        sys.path.insert(0, str(RG))
    import scorecard as SC
    Fr, Gr, _ = cached(f"real_{split}", lambda: real_patches(split))
    lines = [f"water-patch DINOv3 probe, real = {split} days ({len(set(Gr))} days, {len(Fr)} patches of {P} px at the 1280 frame scale)"]
    if len(set(Gr)) >= 6:
        fl = [floor(Fr, Gr, s)[0]["auc"] for s in range(3)]
        lines.append(f"  floor  real days vs other real days: AUC {np.mean(fl):.3f} (3 random day splits: {', '.join(f'{x:.3f}' for x in fl)})")
    else:
        lines.append("  floor  not measurable with fewer than 6 days (use the training-day floor); Frechet floor = two random photo halves")
    days = sorted(set(Gr)); rng = np.random.default_rng(0)
    a = rng.random(len(Gr)) < 0.5 if len(days) < 6 else np.array([g in set(rng.permutation(days)[: len(days) // 2].tolist()) for g in Gr])
    sets = []
    if include_run05:
        F5, G5, _ = cached("run05", run05_patches); sets.append(("run05", F5, G5))
    for t in refs:
        t, _, lim = t.partition(":")                    # "dir:120" = the first 120 frames
        lim = int(lim or 0)
        nm = f"ref_{Path(t).name}" + (f"_{lim}" if lim else "")
        Fx, Gx, _ = cached(nm, lambda t=t, lim=lim: ref_patches(t, limit=lim)); sets.append((nm[4:], Fx, Gx))
    n_common = int(min([300] + [len(Fx) for _, Fx, _ in sets]))          # one sample size for every Frechet number
    lines.append(f"  (Frechet at n = {n_common} per side for every row; AUC folds hold out whole real days / render frames)")
    for name, Fx, Gx in sets:
        r = probe(Fr, Gr, Fx, Gx)
        fd = SC.fd_block(Fr[~a], Fr[a], Fx, n_fix=n_common)
        lines.append(f"  {name:<28} n={len(Fx):<5} AUC {r['auc']:.3f}  acc {r['acc']:.3f}  d' {r['dprime']}  "
                     f"Frechet real-real {fd['fd_real_real']:.2f} real-render {fd['fd_real_render']:.2f} ratio {fd['fd_ratio']:.2f} (n {fd['n']})")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--no-run05", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    txt = report(a.split, a.ref, not a.no_run05)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
