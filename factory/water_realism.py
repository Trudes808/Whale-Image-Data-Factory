"""Reference water frames measured with the realism-gap statistics (Dinov3-WhaleID/experiments/realism_gap), next to
the cached real photographs (train + val days only) and run05.

    PYTHONPATH=/home/sat3737/NOAA_whales/Whale-Image-Data-Factory:/home/sat3737/NOAA_whales/Dinov3-WhaleID \
      /home/sat3737/NOAA_whales/Dinov3-WhaleID/.venv/bin/python -m factory.water_realism frames --tag v1 --n 96
    ... -m factory.water_realism table --tag v1          # real / run05 / reference statistics (text + png)
    ... -m factory.water_realism sheets --tag v1         # reference frames next to real water at the same scale

Frames are drawn like field photographs: camera 2-4 m, focal 65-300 mm (22.3 mm sensor, 2560 x 1707), the aim
point 50-400 m out, one third of the frames with the horizon in view (as in the real set), the sun at its HDRI's
elevation and a random azimuth relative to the view, one of the baked sea states. Statistics are computed exactly
as the realism tool does for a frame (long side resized to 1280, far-water region = water below the horizon).
No real photograph is written anywhere but the scratch sheets.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

RG = Path("/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/realism_gap")
OUT_PICS = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/water_engine")
REAL_CACHE = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/realism_gap/cache")
REAL_NAME, RUN05_NAME = "real_v1", "render_run05_dataset_e5b544cc"
FRAMES = Path("/var/tmp/sat3737-parts-cache/water_engine/frames")
FAR_STATS = ["far.coherence", "far.vgrad_share", "far.hue_deg", "far.a_mean", "far.b_mean", "far.C_mean", "far.L_mean", "far.L_std",
             "far.S_hsv", "far.clip_hi", "far.foam", "far.tex_fine", "far.tex_medium", "far.tex_coarse", "far.psd_slope",
             "frame.clip_hi", "frame.L_p99", "frame.noise", "frame.has_horizon"]
BAND_STATS = ["band.psd_slope", "band.foam", "band.coherence", "band.tex_fine", "band.clip_hi", "band.hue_deg", "band.C_mean", "band.L_mean"]
EVAL_DAYS = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}


def _rg():
    if str(RG) not in sys.path:
        sys.path.insert(0, str(RG))
    import common  # noqa: F401
    import stats
    return common, stats


def rows(name: str) -> list[dict]:
    return [json.loads(l) for l in (REAL_CACHE / name / "stats.jsonl").read_text().splitlines() if l.strip()]


def frame_stats(rgb: np.ndarray, horizon_rows: np.ndarray | None, whale: np.ndarray | None = None) -> dict:
    """The realism tool's frame statistics (and crop statistics when a whale mask is given) for one frame."""
    import cv2
    common, S = _rg()
    H, W = rgb.shape[:2]
    sv = common.FRAME_LONG / max(H, W)
    fw, fh = int(round(W * sv)), int(round(H * sv))
    frgb = common.resize_any(rgb, (fw, fh))
    wh = np.zeros((H, W), bool) if whale is None else whale
    fwh = common.resize_any(wh, (fw, fh), nearest=True) > 0
    fig = np.zeros((fh, fw), bool)
    hz = None
    if horizon_rows is not None and (horizon_rows >= 0).any():
        hr = np.where(horizon_rows < 0, 0, horizon_rows)
        hz = (np.interp(np.arange(fw) / sv, np.arange(W), hr) * sv).astype(np.float32)
        if (hz <= 0).all():
            hz = None
    flab, fS = S.to_lab(frgb)
    if wh.any():
        ys, xs = np.nonzero(wh); box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    else:
        box = [W * 0.45, H * 0.45, W * 0.55, H * 0.55]
    fbox = [v * sv for v in box]
    band_f = float(np.clip(fbox[3] - fbox[1], 8, 0.2 * fw)) if wh.any() else 0.0
    lab_map = common.region_map(fwh, fig, np.ones((fh, fw), bool), hz, band_f)
    st = S.frame_stats(frgb, flab, fS, lab_map, np.ones((fh, fw), bool))
    st["frame.has_horizon"] = float(hz is not None)
    if wh.any():
        sx, sy, side = common.crop_params(box, W, H)
        C = common.CROP
        crgb, _ = common.warp_square(rgb, sx, sy, side, C, None)
        cwh = common.warp_square(wh.astype(np.uint8), sx, sy, side, C, cv2.INTER_NEAREST)[0] > 0
        cvalid = common.warp_square(np.ones((H, W), np.uint8), sx, sy, side, C, cv2.INTER_NEAREST)[0] > 0
        hzc = None
        if hz is not None:
            u = np.arange(C); xn = sx + (u + 0.5) * side / C
            yn = np.interp(xn * sv, np.arange(fw), hz) / sv
            hzc = ((yn - sy) * C / side).astype(np.float32)
        band_c = float(np.clip((box[3] - box[1]) * C / side, 4, 0.25 * C))
        clab_map = common.region_map(cwh, np.zeros_like(cwh), cvalid, hzc, band_c)
        clab, cS = S.to_lab(crgb)
        st.update(S.crop_stats(crgb, clab, cS, clab_map, cvalid))
    return {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in st.items()}


# ------------------------------------------------------------------------------------------------ frames
def draw_shots(n: int, seed: int, bakes: list[Path], skies: list[str], horizon_share: float = 0.32):
    from factory import waterref as W
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        f = float(np.exp(rng.uniform(math.log(65), math.log(300))))
        d = float(np.exp(rng.uniform(math.log(50), math.log(400))))
        h = float(rng.uniform(2.0, 4.0))
        vfov2 = math.atan(22.3 / 1.5 / 2 / f)
        off = float(rng.normal(0, 0.12)) * 2 * vfov2       # whale not always at the centre row
        dep = math.atan2(h, d)
        has_h = dep - off < vfov2
        want_h = len([s for s in out if s["has_h"]]) < horizon_share * (len(out) + 1)
        if has_h != want_h:                                # keep the real set's horizon share
            continue
        sky = skies[int(rng.integers(len(skies)))]
        out.append(dict(shot=dict(cam_height_m=h, focal_mm=f, distance_m=d, yaw_deg=float(rng.uniform(0, 360)),
                                  pitch_deg=-math.degrees(dep - off), cam_xy_m=(float(rng.uniform(0, 500)), float(rng.uniform(0, 500))),
                                  sky=sky, sun_azimuth_deg=0.0, seed=int(rng.integers(1 << 30))),
                        rel_sun_az=float(rng.uniform(-180, 180)), bake=str(bakes[int(rng.integers(len(bakes)))]), has_h=has_h))
    for s in out:
        s["shot"]["sun_azimuth_deg"] = s["shot"]["yaw_deg"] + s["rel_sun_az"]
    return out


def cmd_frames(a) -> int:
    import torch
    from PIL import Image
    from factory import oceanbake, waterref as W
    look = W.Look(**json.loads(a.look)) if a.look else W.Look()
    bakes = [Path(b) for b in a.bakes]
    shots = draw_shots(a.n, a.seed, bakes, a.skies or W.CLEAR_SKIES)
    out = FRAMES / a.tag; out.mkdir(parents=True, exist_ok=True)
    (out / "look.json").write_text(json.dumps({**look.__dict__, "geo": a.geo}, indent=1, default=list))
    dev = W.pick_device()
    cache_r, cache_s = {}, {}
    res_rows = []
    order = sorted(range(len(shots)), key=lambda i: shots[i]["bake"])
    for i in order:
        s = shots[i]
        if s["bake"] not in cache_r:
            cache_r.clear(); torch.cuda.empty_cache() if dev.type == "cuda" else None
            cache_r[s["bake"]] = W.Renderer(oceanbake.BakedSea.load(Path(s["bake"])), dev, look, geo=a.geo)
        R = cache_r[s["bake"]]
        sky_name = s["shot"]["sky"]
        if sky_name not in cache_s:
            cache_s[sky_name] = W.Sky(sky_name, dev)
        shot = W.Shot(**s["shot"])
        r = R.render(shot, cache_s[sky_name])
        img, info = W.develop(r["linear"], look, shot.seed)
        Image.fromarray(img).save(out / f"{i:04d}.jpg", quality=95)
        np.savez_compressed(out / f"{i:04d}_aux.npz", horizon=r["aux"]["horizon_rows"], foam=r["aux"]["foam"].astype(np.float16),
                            glint=np.minimum(r["aux"]["glint"], 6e4).astype(np.float16))
        st = frame_stats(img, r["aux"]["horizon_rows"])
        st.update({"_i": i, "_bake": Path(s["bake"]).name, "_focal": shot.focal_mm, "_dist": shot.distance_m, "_h": shot.cam_height_m,
                   "_rel_sun": s["rel_sun_az"], "_sky": sky_name, "_ev": info["ev"]})
        res_rows.append(st)
        print(i, Path(s["bake"]).name, f"f{shot.focal_mm:.0f} d{shot.distance_m:.0f} sun{s['rel_sun_az']:.0f}",
              {k: round(st.get(k) or float('nan'), 3) for k in ("far.coherence", "far.hue_deg", "far.C_mean", "far.L_mean", "far.clip_hi")}, flush=True)
    (out / "stats.jsonl").write_text("\n".join(json.dumps(r) for r in sorted(res_rows, key=lambda r: r["_i"])) + "\n")
    (out / "shots.json").write_text(json.dumps(shots, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ table
def table(ref_rows: list[dict], keys=FAR_STATS, title: str = "") -> str:
    from scipy.stats import ks_2samp
    real = rows(REAL_NAME); r05 = rows(RUN05_NAME)

    def col(rr, k):
        v = np.array([r.get(k) if r.get(k) is not None else np.nan for r in rr], float)
        return v[np.isfinite(v)]

    def fmt(v):
        if len(v) == 0:
            return "-"
        q = np.percentile(v, [25, 50, 75])
        f = (lambda x: f"{x:.3g}") if np.abs(q).max() < 1000 else (lambda x: f"{x:.0f}")
        return f"{f(q[1])} [{f(q[0])}, {f(q[2])}]"
    lines = [title, f"{'statistic':<18}{'real (n=%d)' % len(real):<28}{'run05':<28}{'reference':<28}{'KS run05':>9}{'KS ref':>8}{'in IQR':>7}"]
    for k in keys:
        a, b, c = col(real, k), col(r05, k), col(ref_rows, k)
        ks5 = ks_2samp(a, b).statistic if len(a) > 5 and len(b) > 5 else float("nan")
        ksr = ks_2samp(a, c).statistic if len(a) > 5 and len(c) > 5 else float("nan")
        q = np.percentile(a, [25, 75]) if len(a) else (np.nan, np.nan)
        inside = (q[0] <= np.median(c) <= q[1]) if len(c) else False
        lines.append(f"{k:<18}{fmt(a):<28}{fmt(b):<28}{fmt(c):<28}{ks5:>9.2f}{ksr:>8.2f}{'yes' if inside else 'no':>7}")
    return "\n".join(lines)


KEY_STATS = ["far.coherence", "far.vgrad_share", "far.hue_deg", "far.a_mean", "far.b_mean", "far.C_mean", "far.L_mean", "far.L_std",
             "far.clip_hi", "far.foam", "far.tex_fine", "far.tex_medium", "far.tex_coarse", "far.psd_slope"]


def cmd_summary(a) -> int:
    """One table: real / run05 / every reference tag (median and KS against real), and the mean KS over KEY_STATS."""
    from scipy.stats import ks_2samp
    real = rows(REAL_NAME); r05 = rows(RUN05_NAME)
    tags = a.tags
    sets = {t: [json.loads(l) for l in (FRAMES / t / "stats.jsonl").read_text().splitlines() if l.strip()] for t in tags}

    def col(rr, k):
        v = np.array([r.get(k) if r.get(k) is not None else np.nan for r in rr], float)
        return v[np.isfinite(v)]
    names = ["run05"] + [t.replace("final_", "") for t in tags]
    lines = [f"{'statistic':<17}{'real median':>12}" + "".join(f"{n:>21}" for n in names), " " * 29 + "".join(f"{'median (KS)':>21}" for _ in names)]
    ks_sum = {n: [] for n in names}
    for k in KEY_STATS + ["frame.clip_hi", "frame.has_horizon"]:
        a_ = col(real, k)
        cells = []
        for n, rr in zip(names, [r05] + [sets[t] for t in tags]):
            c = col(rr, k)
            if len(c) > 3:
                ks = ks_2samp(a_, c).statistic
                if k in KEY_STATS:
                    ks_sum[n].append(ks)
                cells.append(f"{np.median(c):>12.4g} ({ks:.2f})")
            else:
                cells.append(f"{'-':>21}")
        lines.append(f"{k:<17}{np.median(a_):>12.4g}" + "".join(f"{c:>21}" for c in cells))
    lines.append(f"{'mean KS (far)':<29}" + "".join(f"{np.mean(ks_sum[n]):>21.3f}" for n in names))
    txt = "\n".join(lines)
    print(txt)
    (OUT_PICS / f"stats_summary_{a.out}.txt").write_text(txt + "\n")
    text_png(txt, OUT_PICS / f"stats_summary_{a.out}.png")
    return 0


def text_png(text: str, path: Path, scale: int = 1):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    ls = text.splitlines()
    w = max(len(l) for l in ls) * 9 + 20; h = len(ls) * 19 + 20
    im = Image.new("RGB", (min(w, 1990), h), "white"); d = ImageDraw.Draw(im)
    for i, l in enumerate(ls):
        d.text((10, 10 + 19 * i), l, fill="black", font=font)
    im.save(path)


def cmd_table(a) -> int:
    rr = [json.loads(l) for l in (FRAMES / a.tag / "stats.jsonl").read_text().splitlines() if l.strip()]
    txt = table(rr, FAR_STATS, f"far water: real photos (train+val) / run05 / reference renderer '{a.tag}' (n={len(rr)}); median [IQR]")
    by = {}
    for r in rr:
        by.setdefault(r["_bake"].split("_s")[0], []).append(r)
    txt += "\n\nper sea state (KS against real, far.coherence / hue / chroma / L / clip / tex_fine / psd):\n"
    from scipy.stats import ks_2samp
    real = rows(REAL_NAME)
    for st, group in sorted(by.items()):
        parts = []
        for k in ("far.coherence", "far.vgrad_share", "far.hue_deg", "far.C_mean", "far.L_mean", "far.clip_hi", "far.foam", "far.tex_fine", "far.tex_medium", "far.psd_slope"):
            a_ = np.array([r[k] for r in real if r.get(k) is not None]); c_ = np.array([r[k] for r in group if r.get(k) is not None])
            parts.append(f"{k.split('.')[1]} {np.median(c_):.3g} (KS {ks_2samp(a_, c_).statistic:.2f})" if len(c_) > 3 else f"{k} -")
        txt += f"  {st:<14} n={len(group):<3} " + "; ".join(parts) + "\n"
    print(txt)
    OUT_PICS.mkdir(parents=True, exist_ok=True)
    (OUT_PICS / f"stats_{a.tag}.txt").write_text(txt)
    text_png(txt, OUT_PICS / f"stats_{a.tag}.png")
    return 0


# ------------------------------------------------------------------------------------------------ sheets
def _real_far_patches(k: int, side: int, scale_long: int, seed: int = 0):
    """k far-water patches (side x side px) from real photos resized to a long side of scale_long (2560 = native)."""
    import cv2
    common, _ = _rg()
    samples = json.loads((REAL_CACHE / REAL_NAME / "samples.json").read_text())
    rr = rows(REAL_NAME)
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(samples)):
        s = samples[i]
        if s.get("day") in EVAL_DAYS or s["world"] != "real":
            continue
        rgb, whale, ign, box = common.load_sample(s)
        H, W = rgb.shape[:2]
        sc = scale_long / max(H, W)
        if abs(sc - 1) > 1e-3:
            rgb = cv2.resize(rgb, (int(round(W * sc)), int(round(H * sc))), interpolation=cv2.INTER_AREA)
            whale = cv2.resize(whale.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            ign = cv2.resize(ign.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        H, W = rgb.shape[:2]
        bad = cv2.dilate((whale | ign).astype(np.uint8), np.ones((side // 2, side // 2), np.uint8)) > 0
        for bb in [s["bbox"]] + [o["bbox"] for o in s.get("others", [])]:      # every instance box, grown by 60 %
            kb = W / 2560.0
            x0, y0, x1, y1 = [v * kb for v in bb]; gx, gy = 0.6 * (x1 - x0), 0.6 * (y1 - y0)
            bad[max(int(y0 - gy), 0):int(y1 + gy), max(int(x0 - gx), 0):int(x1 + gx)] = True
        # take a square below the horizon (the lower 60 % of the frame) without whale
        for _ in range(30):
            y = int(rng.integers(int(0.4 * H), H - side)) if H - side > int(0.4 * H) else 0
            x = int(rng.integers(0, W - side))
            if not bad[y:y + side, x:x + side].any():
                out.append((rgb[y:y + side, x:x + side], f"real {s['day']} far L*{rr[i].get('far.L_mean', 0):.0f}"))
                break
        if len(out) >= k:
            break
    return out


def cmd_sheets(a) -> int:
    import cv2
    from PIL import Image, ImageDraw
    d = FRAMES / a.tag
    stats = [json.loads(l) for l in (d / "stats.jsonl").read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(1)
    # 1) frames at the tool's scale (1280 long side): full frames, 8 reference / 8 real thumbs
    T = 480
    pick = rng.permutation(len(stats))[:8]
    refs = []
    for i in pick:
        r = stats[i]
        im = cv2.cvtColor(cv2.imread(str(d / f"{r['_i']:04d}.jpg")), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (T, int(round(T * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
        refs.append((im, f"ref {r['_bake'].split('_s')[0]} f{r['_focal']:.0f} d{r['_dist']:.0f} sun{r['_rel_sun']:.0f}"))
    thumbs = REAL_CACHE / REAL_NAME / "thumbs"
    reals = []
    samples = json.loads((REAL_CACHE / REAL_NAME / "samples.json").read_text())
    for i in rng.permutation(len(samples))[:8]:
        if samples[i].get("day") in EVAL_DAYS:
            continue
        im = cv2.cvtColor(cv2.imread(str(thumbs / f"{i}_frame.jpg")), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (T, int(round(T * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
        reals.append((im, f"real {samples[i]['day']}"))
    _grid([refs[:4], reals[:4], refs[4:8], reals[4:8]], OUT_PICS / f"frames_{a.tag}.jpg")
    # 2) far-water patches at the realism tool's scale (frame long side 1280): 256 px squares
    P = 256
    ref_p = []
    for i in pick:
        r = stats[i]
        im = cv2.cvtColor(cv2.imread(str(d / f"{r['_i']:04d}.jpg")), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (1280, int(round(1280 * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
        hz = np.load(d / f"{r['_i']:04d}_aux.npz")["horizon"]
        top = int(max(hz.max(), 0) / 2) + 10
        y = int(rng.integers(max(top, im.shape[0] // 2 - 100), im.shape[0] - P)); x = int(rng.integers(0, 1280 - P))
        ref_p.append((im[y:y + P, x:x + P], f"ref {r['_bake'].split('_s')[0]} f{r['_focal']:.0f}"))
    real_p = _real_far_patches(8, P, 1280, seed=a.seed)
    _grid([ref_p[:4] + real_p[:4], ref_p[4:] + real_p[4:]], OUT_PICS / f"patches1280_{a.tag}.jpg")
    # 3) native 2560 scale: 3 reference frames' 640 px windows next to real 640 px windows at native resolution
    Q = 480
    nat_r = []
    for i in pick[:3]:
        r = stats[i]
        im = cv2.cvtColor(cv2.imread(str(d / f"{r['_i']:04d}.jpg")), cv2.COLOR_BGR2RGB)
        H = im.shape[0]
        nat_r.append((im[H - Q - 100:H - 100, 1000:1000 + Q], f"ref 2560 f{r['_focal']:.0f} d{r['_dist']:.0f}"))
    nat_real = _real_far_patches(3, Q, 2560, seed=a.seed + 5)
    _grid([[nat_r[0], nat_real[0]], [nat_r[1], nat_real[1]], [nat_r[2], nat_real[2]]], OUT_PICS / f"native2560_{a.tag}.jpg")
    return 0


# real-photo camera distribution (coordinator, 2026-11-17): focal percentiles 5/25/50/75/95 % = 65/160/220/300/300 mm
FOCAL_PCT = ([0.0, 0.05, 0.25, 0.50, 0.75, 1.0], [65.0, 65.0, 160.0, 220.0, 300.0, 300.0])


def draw_plate_shots(n: int, seed: int, bakes: list[Path], skies: list[str], horizon_share: float = 1 / 3):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        f = float(np.interp(rng.uniform(), *FOCAL_PCT))
        d = float(np.exp(rng.uniform(math.log(50), math.log(400))))
        h = float(rng.uniform(2.0, 4.0))
        vfov2 = math.atan(22.3 / 1.5 / 2 / f)
        off = float(rng.normal(0, 0.12)) * 2 * vfov2
        dep = math.atan2(h, d)
        has_h = dep - off < vfov2
        if has_h != (sum(s["has_h"] for s in out) < horizon_share * (len(out) + 1)):
            continue
        # sun elevation 25-67 deg, median 46: a triangular-ish draw (two uniforms averaged), clipped
        el = float(np.clip(46.0 + 21.0 * (rng.uniform() + rng.uniform() - 1.0), 25.0, 67.0))
        yaw = float(rng.uniform(0, 360)); rel = float(rng.uniform(-180, 180))
        out.append(dict(shot=dict(cam_height_m=h, focal_mm=f, distance_m=d, yaw_deg=yaw, pitch_deg=-math.degrees(dep - off),
                                  cam_xy_m=(float(rng.uniform(0, 1000)), float(rng.uniform(0, 1000))), sky=skies[int(rng.integers(len(skies)))],
                                  sun_azimuth_deg=yaw + rel, sun_elevation_deg=el, seed=int(rng.integers(1 << 30))),
                        rel_sun_az=rel, bake=str(bakes[int(rng.integers(len(bakes)))]), has_h=has_h))
    return out


def cmd_plates(a) -> int:
    """Water-only plates for a training test: JPEG q92 at 2560 x 1707 + one JSON per frame."""
    import time
    import torch
    from PIL import Image
    from factory import oceanbake, waterref as W
    look = W.Look(**json.loads(a.look)) if a.look else W.Look()
    look.camera_fx = dict(look.camera_fx, jpeg_quality=[100, 100])      # the one JPEG compression is the q92 save
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    shots = draw_plate_shots(a.n, a.seed, [Path(b) for b in a.bakes], a.skies or W.CLEAR_SKIES)
    dev = W.pick_device(a.gpu_gb)
    cache_r, cache_s = {}, {}
    order = sorted(range(len(shots)), key=lambda i: (i // 30, shots[i]["bake"]))   # every block of 30 mixes the sea states
    for i in order:
        stem = f"plate_{i:04d}"
        if (out / f"{stem}.json").exists():
            continue
        t0 = time.time()
        s = shots[i]
        if s["bake"] not in cache_r:
            cache_r.clear()
            cache_r[s["bake"]] = W.Renderer(oceanbake.BakedSea.load(Path(s["bake"])), dev, look, geo=a.geo)
        R = cache_r[s["bake"]]
        if s["shot"]["sky"] not in cache_s:
            cache_s[s["shot"]["sky"]] = W.Sky(s["shot"]["sky"], dev)
        shot = W.Shot(**s["shot"])
        try:
            r = R.render(shot, cache_s[shot.sky])
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(stem, "out of GPU memory, skipped (re-run resumes it)", flush=True)
            continue
        img, info = W.develop(r["linear"], look, shot.seed)
        Image.fromarray(img).save(out / f"{stem}.jpg", quality=92)
        hz = r["aux"]["horizon_rows"]
        lay = R.sea.layout
        meta = {"id": stem, "width": int(img.shape[1]), "height": int(img.shape[0]),
                "horizon_row": None if not (hz >= 0).any() else round(float(np.median(hz[hz >= 0])), 2),
                "horizon_row_min": None if not (hz >= 0).any() else round(float(hz[hz >= 0].min()), 2),
                "horizon_row_max": None if not (hz >= 0).any() else round(float(hz[hz >= 0].max()), 2),
                "focal_mm": round(shot.focal_mm, 2), "sensor_width_mm": shot.sensor_w_mm, "camera_height_m": round(shot.cam_height_m, 3),
                "distance_m": round(shot.distance_m, 2), "pitch_deg": round(r["aux"]["pitch_deg"], 4), "yaw_deg": round(shot.yaw_deg, 3),
                "sun_azimuth_deg": round(shot.sun_azimuth_deg % 360, 3), "sun_relative_azimuth_deg": round(s["rel_sun_az"], 3),
                "sun_elevation_deg": round(shot.sun_elevation_deg, 3), "sky": shot.sky, "sea_state": lay["spec"]["state"],
                "wind_mps": lay["spec"]["wind"], "wind_dir_deg": round(lay["spec"]["wind_dir_deg"], 2), "bake": Path(s["bake"]).name,
                "bake_seed": lay["spec"]["seed"], "seed": shot.seed, "cam_xy_m": [round(v, 3) for v in shot.cam_xy_m],
                "develop": info, "render_s": round(time.time() - t0, 1), "renderer": "factory.waterref (round 2)", "geo_cascades": R.geo,
                "stats": frame_stats(img, hz)}
        (out / f"{stem}.json").write_text(json.dumps(meta, indent=1))
        print(stem, meta["sea_state"], f"f{shot.focal_mm:.0f} d{shot.distance_m:.0f} el{shot.sun_elevation_deg:.0f} rel{s['rel_sun_az']:.0f}",
              f"{meta['render_s']}s", flush=True)
    return 0


def plates_to_frames(plates: Path, tag: str, limit: int = 0) -> Path:
    """Expose plate stats as a frames tag (FRAMES/<tag>/stats.jsonl) for `table` / `summary`."""
    rows_ = []
    for js in sorted(Path(plates).glob("plate_*.json"))[: limit or None]:
        m = json.loads(js.read_text())
        st = dict(m["stats"]); st.update({"_i": m["id"], "_bake": m["bake"], "_focal": m["focal_mm"], "_dist": m["distance_m"],
                                          "_rel_sun": m["sun_relative_azimuth_deg"], "_sky": m["sky"], "_plate": str(js.with_suffix(".jpg"))})
        rows_.append(st)
    d = FRAMES / tag; d.mkdir(parents=True, exist_ok=True)
    (d / "stats.jsonl").write_text("\n".join(json.dumps(r) for r in rows_) + "\n")
    return d


def cmd_plate_sheets(a) -> int:
    """The three sheets for plates: frames at the tool's scale, far-water patches at 1280, native 2560 windows."""
    import cv2
    plates = sorted(Path(a.plates).glob("plate_*.json"))
    rng = np.random.default_rng(a.seed)
    pick = [plates[i] for i in rng.permutation(len(plates))[:8]]
    metas = [json.loads(p.read_text()) for p in pick]
    ims = [cv2.cvtColor(cv2.imread(str(p.with_suffix(".jpg"))), cv2.COLOR_BGR2RGB) for p in pick]
    T = 480
    refs = [(cv2.resize(im, (T, int(round(T * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA),
             f"ref {m['sea_state']} f{m['focal_mm']:.0f} d{m['distance_m']:.0f} sun{m['sun_relative_azimuth_deg']:.0f}") for im, m in zip(ims, metas)]
    samples = json.loads((REAL_CACHE / REAL_NAME / "samples.json").read_text())
    reals = []
    for i in rng.permutation(len(samples)):
        if samples[i].get("day") in EVAL_DAYS:
            continue
        im = cv2.cvtColor(cv2.imread(str(REAL_CACHE / REAL_NAME / "thumbs" / f"{i}_frame.jpg")), cv2.COLOR_BGR2RGB)
        reals.append((cv2.resize(im, (T, int(round(T * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA), f"real {samples[i]['day']}"))
        if len(reals) == 8:
            break
    _grid([refs[:4], reals[:4], refs[4:8], reals[4:8]], OUT_PICS / f"frames_{a.tag}.jpg")
    P = 256
    ref_p = []
    for im, m in zip(ims, metas):
        f = cv2.resize(im, (1280, int(round(1280 * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
        top = int((m["horizon_row_max"] or -20) / 2) + 10
        y = int(rng.integers(max(top, f.shape[0] // 2 - 100), f.shape[0] - P)); x = int(rng.integers(0, 1280 - P))
        ref_p.append((f[y:y + P, x:x + P], f"ref {m['sea_state']} f{m['focal_mm']:.0f}"))
    real_p = _real_far_patches(8, P, 1280, seed=a.seed)
    _grid([ref_p[:4] + real_p[:4], ref_p[4:] + real_p[4:]], OUT_PICS / f"patches1280_{a.tag}.jpg")
    Q = 480
    nat_r = []
    for im, m in zip(ims[:3], metas[:3]):
        H = im.shape[0]
        nat_r.append((im[H - Q - 100:H - 100, 1000:1000 + Q], f"ref 2560 {m['sea_state']} f{m['focal_mm']:.0f} d{m['distance_m']:.0f}"))
    nat_real = _real_far_patches(3, Q, 2560, seed=a.seed + 5)
    _grid([[nat_r[0], nat_real[0]], [nat_r[1], nat_real[1]], [nat_r[2], nat_real[2]]], OUT_PICS / f"native2560_{a.tag}.jpg")
    return 0


DEMO_SHOTS = {"surfacing": "000005", "fluke_up_dive": "000012", "pec_slap": "000164"}
RUN05 = Path("/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders/run05")


def _run05_shot(shot_id: str) -> dict:
    for line in (RUN05 / "shotlist.jsonl").read_text().splitlines():
        if line.startswith('{"id": "%s"' % shot_id):
            return json.loads(line)
    raise KeyError(shot_id)


def cmd_contact(a) -> int:
    """Contact layer on three run05 poses (read only), rendered by the reference renderer with the whale as a dark
    matte shape, plus top views of the contact texture."""
    import time
    import torch
    from PIL import Image
    from factory import contact, oceanbake, project, waterref as W
    from factory.whalemodel import WhaleModel
    look = W.Look(**json.loads(a.look)) if a.look else W.Look()
    model = WhaleModel("humpback4k")
    sea = oceanbake.BakedSea.load(Path(a.bake))
    if not a.layers_only:
        dev = W.pick_device()
        R = W.Renderer(sea, dev, look, geo=a.geo)
        sky = W.Sky(a.sky, dev)
    out = FRAMES / f"contact_{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    tiles, tops, st_rows = [], [], []
    for k, (name, sid) in enumerate(DEMO_SHOTS.items()):
        shot = _run05_shot(sid)
        V = project.transform_mesh(model.posed(shot), shot["whale"]) / 100.0
        lift = float((shot.get("water_surface_z_cm") or {}).get("whale", 0.0)) / 100.0
        loc = np.asarray(shot["whale"]["location"], float) / 100.0
        target = np.array([150.0 + 37.0 * k, 120.0 + 11.0 * k])
        aim = np.asarray(shot["stats"]["aim_point"], float) / 100.0
        h_aim = float(sea.height(np.array([target[0] + aim[0] - loc[0]]), np.array([target[1] + aim[1] - loc[1]]))[0])
        V[:, 0] += target[0] - loc[0]; V[:, 1] += target[1] - loc[1]; V[:, 2] += h_aim - lift
        heading = shot["whitewater"]["heading"] if shot.get("whitewater") else [1.0, 0.0]
        ex = out / f"{name}_contact.exr"
        if ex.exists() and a.reuse:
            layer = contact.read_exr(ex); dt_contact = float(layer["seconds"])
            layer["pierce_mask"] = np.load(out / f"{name}_pierce.npy")
        else:
            t0 = time.time()
            layer = contact.run(V, model.faces, shot["category"], heading, seed=k, water_z=h_aim)
            dt_contact = time.time() - t0
            contact.write_exr(layer, ex); np.save(out / f"{name}_pierce.npy", layer["pierce_mask"])
        if a.layers_only:
            print(name, shot["category"], f"{dt_contact:.1f}s", layer.get("stats"), flush=True)
            continue
        R.set_contact(layer["tex"], layer["origin_m"], layer["size_m"])
        # camera: from the side of the heading (rear quarter for the dive), 110-140 m away
        hd = np.asarray(heading, float); side = np.array([-hd[1], hd[0]])
        view = {"surfacing": side, "fluke_up_dive": -0.6 * hd + 0.8 * side, "pec_slap": side}[name]
        view = view / np.linalg.norm(view)
        dist = {"surfacing": 120.0, "fluke_up_dive": 110.0, "pec_slap": 130.0}[name]
        cen = V[V[:, 2] > h_aim - 0.3][:, :2].mean(0)
        cam_xy = cen + view * dist
        yaw = math.degrees(math.atan2(cen[1] - cam_xy[1], cen[0] - cam_xy[0]))
        f = {"surfacing": 200.0, "fluke_up_dive": 180.0, "pec_slap": 200.0}[name]
        sh = W.Shot(cam_height_m=3.0, focal_mm=f, distance_m=dist, yaw_deg=yaw, cam_xy_m=(float(cam_xy[0]), float(cam_xy[1])), sky=a.sky,
                    sun_azimuth_deg=yaw + a.rel_sun, seed=11 + k)
        sel = (V[model.faces][:, :, 2] > h_aim - 0.6).any(1)
        r = R.render(sh, sky, whale_mesh_m=V, whale_faces=model.faces[sel])
        img, info = W.develop(r["linear"], look, sh.seed)
        Image.fromarray(img).save(out / f"{name}.jpg", quality=95)
        wh = r["aux"]["whale"]
        st = frame_stats(img, r["aux"]["horizon_rows"], wh)
        st.update({"_name": name, "_contact_s": round(dt_contact, 1)}); st_rows.append(st)
        # crop around the whale at native resolution for the sheet
        # crop centred on the visible waterline (lowest whale pixel per column), where the contact layer is
        ys, xs = np.nonzero(wh)
        cols = np.unique(xs); bottom = np.array([ys[xs == c].max() for c in cols[:: max(1, len(cols) // 200)]])
        cy, cx = int(np.median(bottom)), int(np.median(cols))
        hw = 640
        y0 = int(np.clip(cy - hw * 0.45, 0, img.shape[0] - hw)); x0 = int(np.clip(cx - hw, 0, img.shape[1] - 2 * hw))
        tiles.append((img[y0:y0 + hw, x0:x0 + 2 * hw], f"{name} ({shot['category']}, run05 {sid}) f{f:.0f} d{dist:.0f}, contact {dt_contact:.0f}s"))
        # top view of the texture: height (diverging), foam (white), damping (cyan)
        t = layer["tex"]
        hgt = np.clip(t[..., 0] / 0.15, -1, 1)
        rgb = np.stack([0.5 + 0.5 * hgt, 0.5 + 0.0 * hgt, 0.5 - 0.5 * hgt], -1)
        rgb = rgb * (1 - 0.6 * t[..., 2:3]) + np.array([0.0, 0.8, 0.8]) * 0.6 * t[..., 2:3]
        rgb = rgb * (1 - np.clip(t[..., 1:2], 0, 1)) + np.clip(t[..., 1:2], 0, 1)
        rgb[layer["pierce_mask"]] = 0.05
        tops.append(((np.flipud(rgb) * 255).astype(np.uint8), f"{name}: height (red +/blue -, +-15 cm), foam white, slick cyan, 72 m"))
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    if a.layers_only:
        return 0
    R.set_contact(None, (0, 0), 1)
    _grid([[t] for t in tiles], OUT_PICS / f"contact_frames_{a.tag}.jpg")
    _grid([tops], OUT_PICS / f"contact_textures_{a.tag}.jpg")
    (out / "stats.jsonl").write_text("\n".join(json.dumps(r) for r in st_rows) + "\n")
    txt = table(st_rows, BAND_STATS + ["whale.dL_vs_band"], f"contact-layer frames '{a.tag}' (n={len(st_rows)}): band = water within one whale height (crop)")
    print(txt)
    (OUT_PICS / f"contact_stats_{a.tag}.txt").write_text(txt)
    return 0


def _grid(rows_, path: Path, pad: int = 4):
    from PIL import Image, ImageDraw
    hs = [max(t[0].shape[0] for t in r) for r in rows_]
    ws = [sum(t[0].shape[1] for t in r) + pad * (len(r) - 1) for r in rows_]
    W = max(ws); H = sum(hs) + pad * (len(rows_) - 1)
    canvas = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
    y = 0
    for r, h in zip(rows_, hs):
        x = 0
        for im, label in r:
            canvas.paste(Image.fromarray(np.ascontiguousarray(im)), (x, y))
            dr.rectangle([x, y, x + 8 * len(label) + 6, y + 16], fill=(0, 0, 0)); dr.text((x + 3, y + 2), label, fill=(255, 255, 0))
            x += im.shape[1] + pad
        y += h + pad
    if max(canvas.size) > 2000:
        s = 2000 / max(canvas.size)
        canvas = canvas.resize((int(canvas.size[0] * s), int(canvas.size[1] * s)), Image.LANCZOS)
    canvas.save(path, quality=92)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames"); f.add_argument("--tag", required=True); f.add_argument("--n", type=int, default=96)
    f.add_argument("--seed", type=int, default=0); f.add_argument("--bakes", nargs="+", required=True); f.add_argument("--skies", nargs="*")
    f.add_argument("--look", default=""); f.add_argument("--geo", type=int, default=None)
    t = sub.add_parser("table"); t.add_argument("--tag", required=True)
    s = sub.add_parser("sheets"); s.add_argument("--tag", required=True); s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("contact"); c.add_argument("--tag", required=True); c.add_argument("--bake", required=True)
    c.add_argument("--sky", default="qwantani_noon_puresky"); c.add_argument("--rel-sun", type=float, default=150.0); c.add_argument("--look", default=""); c.add_argument("--geo", type=int, default=None)
    c.add_argument("--reuse", action="store_true"); c.add_argument("--layers-only", action="store_true")
    pl = sub.add_parser("plates"); pl.add_argument("--out", default="/var/tmp/sat3737-parts-cache/water_engine/plates")
    pl.add_argument("--n", type=int, default=400); pl.add_argument("--seed", type=int, default=11); pl.add_argument("--bakes", nargs="+", required=True)
    pl.add_argument("--skies", nargs="*"); pl.add_argument("--look", default=""); pl.add_argument("--geo", type=int, default=4)
    pl.add_argument("--gpu-gb", type=float, default=3.0)
    ps = sub.add_parser("plate_sheets"); ps.add_argument("--plates", default="/var/tmp/sat3737-parts-cache/water_engine/plates")
    ps.add_argument("--tag", default="plates_r2"); ps.add_argument("--seed", type=int, default=3)
    m = sub.add_parser("summary"); m.add_argument("--tags", nargs="+", required=True); m.add_argument("--out", default="all")
    a = ap.parse_args(argv)
    return {"frames": cmd_frames, "table": cmd_table, "sheets": cmd_sheets, "contact": cmd_contact, "summary": cmd_summary, "plates": cmd_plates, "plate_sheets": cmd_plate_sheets}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
