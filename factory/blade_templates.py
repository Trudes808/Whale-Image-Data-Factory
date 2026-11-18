"""run06 round 7: dorsal blade outline templates from real fins (the parametric blade, factory.blade).

Each real fin (the researcher's clicked tip T, front base F and back base B on 2025-01-02 and 2025-03-18, Dinov3
experiments) is cut from its live whale mask and normalised to its own triangle frame: a = position along the base line
from the back base (0) toward the front base, in units of the fin height H (the tip's height above the line); h = height
above the line over H. For NH heights h in [0.1, 1] the template holds the leading edge a_lead(h) and the trailing edge
a_trail(h) of the outline's run that contains the tip, plus the tip position and the base length (|F - B| / H).
Templates only: no photo pixels are stored.

    python3 -m factory.blade_templates OUT.json      (needs the Dinov3-WhaleID checkout; run once)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

NH = 19
HEIGHTS = np.linspace(0.1, 1.0, NH)


def outline(mask: np.ndarray, T, F, B) -> dict | None:
    T, F, B = (np.asarray(v, float) for v in (T, F, B))
    d = F - B; L = float(np.linalg.norm(d))
    if L < 5:
        return None
    u = d / L; n = np.array([-u[1], u[0]])
    if (T - B) @ n < 0:
        n = -n
    H = float((T - B) @ n)
    if H < 8:
        return None
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(float) - B
    a, h = P @ u / H, P @ n / H
    ta = float((T - B) @ u) / H
    win = (a > -1.5) & (a < L / H + 0.5)
    lead, trail = [], []
    for hh in HEIGHTS:
        sel = win & (np.abs(h - hh) < max(0.6 / H, 0.025))
        if sel.sum() < 2:
            if hh > 0.95:                      # the very top: a point at the tip
                lead.append(ta); trail.append(ta); continue
            return None
        aa = np.sort(a[sel])
        gaps = np.nonzero(np.diff(aa) > max(2.0 / H, 0.06))[0]
        segs = np.split(aa, gaps + 1)
        seg = min(segs, key=lambda s: 0 if s[0] - 0.05 <= ta <= s[-1] + 0.05 else min(abs(s[0] - ta), abs(s[-1] - ta)))
        lead.append(float(seg[-1])); trail.append(float(seg[0]))
    lead, trail = np.array(lead), np.array(trail)
    if (lead - trail).min() < 0 or (lead - trail)[0] < 0.15:
        return None
    return {"base_over_height": L / H, "tip_along": ta, "lead": np.round(lead, 4).tolist(), "trail": np.round(trail, 4).tolist(),
            "H_px": round(H, 1)}


def main(argv=None) -> int:
    out = Path((argv or sys.argv[1:])[0])
    dino = Path("/home/sat3737/NOAA_whales/Dinov3-WhaleID")
    sys.path.insert(0, str(dino))
    from model_fine_tuning.parts_weak import LAB, DAYS
    from PIL import Image
    temps = []
    for day in ("2025-01-02", "2025-03-18"):
        labs = json.loads((LAB / day / "pair_marks.json").read_text()); labs = labs["labels"] if isinstance(labs, dict) else labs
        kp = {}
        for l in labs:
            if l.get("verdict") == "skip":
                continue
            for sd in ("a", "b"):
                k = (l.get("keypoints") or {}).get(sd) or {}
                if k and l.get(f"key_{sd}"):
                    kp.setdefault(l[f"key_{sd}"], {}).update(k)
        for key, k in sorted(kp.items()):
            if not all(n in k for n in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")):
                continue
            stem = key.rsplit("~", 1)[0]
            mp, ip = DAYS / day / "canon" / f"{stem}_mask.png", DAYS / day / "images" / f"{stem}.jpg"
            if not mp.exists() or not ip.exists():
                continue
            W, Hh = Image.open(ip).size
            m = np.asarray(Image.open(mp).convert("L")) > 0
            sx, sy = m.shape[1] / W, m.shape[0] / Hh
            P = {n: (k[n]["x"] * W * sx, k[n]["y"] * Hh * sy) for n in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")}
            t = outline(m, P["dorsal_fin_tip"], P["dorsal_fin_front"], P["dorsal_fin_back"])
            if t:
                t["source"] = f"{day}"
                temps.append(t)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"heights": HEIGHTS.round(4).tolist(), "templates": temps}, indent=0))
    tmp.replace(out)
    print(len(temps), "templates ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
