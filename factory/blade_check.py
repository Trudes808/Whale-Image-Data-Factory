"""run06 round 11: fold check of the posed parametric dorsal blade (factory.blade / WhaleModel.enable_blade).

For every whale of a run's shot list (and, with --sweep K, every animal of the library in K representative poses taken
from that list), on the POSED blade:
  backwards   a ring point running backwards along the chord (beyond TOL_CM), counted per ring
  column_down a ring lower than the ring below it in the same column (along the posed blade's up), counted per point
  sides_cross the two sides of a ring crossing each other (a right-side point beyond its left mirror point), per ring
  stretched   a blade edge longer than 2 x its length on the rest (placed, scaled) blade and by more than 1 cm
  junction    a point of the blade's lowest, sunk ring whose nearest crossing with the posed body surface (along the
              blade's up) lies more than JUNCTION_CM below it: the blade's bottom is outside the skin, an open gap. 6 cm: the back's posed vertex spacing beside the fin is about 3-5 cm, so a
              nearest-vertex offset below ~1.5 spacings is mesh resolution, not a visible step (a 6 cm step on a 15 m
              whale is ~2 px at the side views' typical scale)
Exit status 1 if any count is above zero (so a build cannot start with a folding blade); 0 without a blade.

    python3 -m factory.blade_check RUN_DIR [--sweep 40] [--workers 8] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOL_CM = 0.3
STRETCH = 2.0
STRETCH_MIN_CM = 1.0      # and longer by at least 1 cm: the knife-thin trailing-edge strip has sub-cm edges
JUNCTION_CM = 6.0
_M = None


def _model(config: dict):
    global _M
    if _M is None:
        from factory.whalemodel import WhaleModel
        _M = WhaleModel.from_config(config)
    return _M


def check_animal(m, animal: dict) -> dict:
    """Fold counts of one posed whale's blade."""
    from factory import blade as bl
    ind = animal.get("individual")
    scale = float(animal.get("body_scale", 1.0))
    V = np.asarray(m.posed(animal), float)
    rig, Pb = V[: m.n_rig], V[m.n_rig:]
    local = m._blade_local(ind)
    a = m._blade_base_anchors(ind, local) if hasattr(m, "_blade_base_anchors") else m._blade_anchors
    Rm, t = bl.kabsch(m.rest[a] * scale, rig[a])
    _, along, up = m._blade_frame_for(local)
    alp, upp, latp = Rm @ along, Rm @ up, Rm @ np.array([0.0, 1.0, 0.0])
    R2 = 2 * bl.NC
    ring = Pb[:-1].reshape(bl.NL, R2, 3)
    x, h, y = ring @ alp, ring @ upp, ring @ latp
    tol = TOL_CM * scale
    right_back = (np.diff(x[:, : bl.NC], axis=1) < -tol).any(1)            # TE -> LE must go forward
    left_back = (np.diff(x[:, bl.NC:], axis=1) > tol).any(1)               # LE -> TE must go back
    backwards = int((right_back | left_back).sum())
    column_down = int((np.diff(h, axis=0) < -tol).sum())
    mirror = y[:, : bl.NC] - y[:, bl.NC:][:, ::-1]                         # right point minus its left mirror
    sides_cross = int((mirror < -tol).any(1).sum())
    # edges against the rest (placed, scaled) blade
    rest_b = m._blade_place(local) * scale
    F = np.asarray(local["faces"])
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    lp = np.linalg.norm(Pb[E[:, 0]] - Pb[E[:, 1]], axis=1); lr = np.linalg.norm(rest_b[E[:, 0]] - rest_b[E[:, 1]], axis=1)
    stretched = int(((lp > STRETCH * np.maximum(lr, 1e-3)) & (lp - lr > STRETCH_MIN_CM * scale)).sum())
    # junction: the blade's lowest, sunk ring (h = -0.35 H) must stay inside the body, or the blade stands open on the
    # skin (a gap under its foot). The line through each point along the blade's up is intersected with the body's
    # triangles near the blade; if the nearest crossing lies BELOW the point by more than JUNCTION_CM, the point hangs
    # outside the skin. (The foot ring at h = 0 may stand a little above the skin on the flanks: the wall below it runs
    # on into the body, which is the fin's own base, not a gap.)
    foot = ring[0]                                   # the lowest (sunk) ring: if it is outside the skin, the blade is open
    Fb = np.asarray(m.faces[: m.n_rig_faces])
    c = foot.mean(0)
    fsel = Fb[(np.linalg.norm(rig[Fb].mean(1) - c, axis=1) < 300.0 * scale)]
    junction = 0
    if len(fsel):
        A, Bv, C = rig[fsel[:, 0]], rig[fsel[:, 1]], rig[fsel[:, 2]]
        e1, e2 = Bv - A, C - A
        pv = np.cross(upp[None, :], e2); det = (e1 * pv).sum(1)
        okd = np.abs(det) > 1e-9
        def crossing(p):
            tv = p - A
            u = (tv * pv).sum(1) / np.where(okd, det, 1.0)
            qv = np.cross(tv, e1)
            v = (upp[None, :] * qv).sum(1) / np.where(okd, det, 1.0)
            s = (e2 * qv).sum(1) / np.where(okd, det, 1.0)          # signed distance along up from p to the crossing
            hit = okd & (u >= 0) & (v >= 0) & (u + v <= 1)
            return s[hit][np.argmin(np.abs(s[hit]))] if hit.any() else None
        # three parallel lines 1.5 cm apart across the blade: a line along the dorsal midline can slip through the
        # mesh's seam there; the point is outside only if every line says so
        for p in foot:
            outs = [(lambda sn: sn is not None and sn < -JUNCTION_CM * scale)(crossing(p + d * latp)) for d in (-1.5, 0.0, 1.5)]
            if all(outs):
                junction += 1
    return {"backwards": backwards, "column_down": column_down, "sides_cross": sides_cross, "stretched": stretched,
            "junction": junction}


def _job(args):
    config, label, animal = args
    m = _model(config)
    try:
        r = check_animal(m, animal)
    except Exception as exc:                                                 # a failure to pose is a failure
        r = {"error": str(exc)}
    return label, r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--sweep", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    config = json.loads((run / "run_config.json").read_text())["config"]
    if not (config.get("whale") or {}).get("blade_templates"):
        print("blade_check: no parametric blade in this run"); return 0
    shots = [json.loads(l) for l in (run / "shotlist.jsonl").read_text().splitlines() if l.strip()]
    if a.limit:
        shots = shots[: a.limit]
    jobs = []
    for s in shots:
        if s.get("individual"):
            jobs.append((config, f"{s['id']}a1 {s['individual']}", s))
        for c in s.get("companions") or []:
            if c.get("individual"):
                jobs.append((config, f"{s['id']}a{c.get('instance_id')} {c['individual']}", c))
    if a.sweep:
        from factory import individuals as indiv
        from factory.whalemodel import WhaleModel
        lib = indiv.load_library(WhaleModel.from_config(config).library_path)
        poses = [s for s in shots if s.get("individual")]
        step = max(1, len(poses) // a.sweep)
        for s in poses[::step][: a.sweep]:
            for k in sorted(lib):
                jobs.append((config, f"sweep {s['id']} {k}", {**s, "individual": k}))
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        res = pool.map(_job, jobs, chunksize=8)
    keys = ("backwards", "column_down", "sides_cross", "stretched", "junction")
    tot = {k: 0 for k in keys}; bad = []
    errors = [(l, r["error"]) for l, r in res if "error" in r]
    for l, r in res:
        if "error" in r:
            continue
        for k in keys:
            tot[k] += r[k]
        if any(r[k] for k in keys):
            bad.append((l, r))
    out = {"run": str(run), "whales_checked": len(res), "totals": tot, "whales_failing": len(bad), "errors": len(errors),
           "failing": [{"whale": l, **r} for l, r in bad][:400], "error_list": errors[:20]}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("run", "whales_checked", "totals", "whales_failing", "errors")}))
    for l, r in bad[:25]:
        print("  ", l, r)
    return 1 if (bad or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
