"""run06 round 7: the individuals library with parametric blades (individuals_run06d.json) from an earlier run06 library:
every animal gets a real-fin outline template (factory.blade_templates), a blade height, nicks and thickness; the asset's
own blade is flattened (dorsal_height 0.08, no chord / sweep / rake / round edits) and the asset hump is off (the
template's outline carries the hump's front slope). Deterministic per individual id.

    python3 -m factory.blade_library SRC.json OUT.json TEMPLATES.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    src, out, temps = (Path(a) for a in (argv or sys.argv[1:])[:3])
    lib = json.loads(src.read_text())
    T = json.loads(temps.read_text())["templates"]
    n = len(T)
    # the templates spread evenly over the real distribution (sorted by base / height, evenly spaced picks), so a library
    # of a few dozen animals has the real quartiles, not a lucky draw
    order = np.argsort([t["base_over_height"] for t in T])
    keys = sorted(lib)
    picks = order[np.linspace(0, n - 1, len(keys)).round().astype(int)]
    picks = picks[np.random.default_rng(7).permutation(len(keys))]
    assigned = dict(zip(keys, (int(i) for i in picks)))
    for k, v in lib.items():
        rng = np.random.default_rng(int(hashlib.sha1(f"blade7:{k}".encode()).hexdigest()[:8], 16))
        v.update(dorsal_height=0.08, dorsal_chord=1.0, dorsal_sweep_cm=0.0, dorsal_rake_cm=0.0, dorsal_round=0.0, dorsal_hump_cm=0.0,
                 blade_template=assigned[k],
                 blade_H_cm=float(np.clip(np.exp(rng.normal(np.log(22.0 if v.get("calf") else 38.0), 0.35)), 10.0, 75.0)),
                 blade_nicks=int(rng.choice([0, 0, 1, 2, 3])), blade_thickness=float(rng.uniform(0.85, 1.2)))
    tmp = out.with_suffix(".json.tmp"); tmp.write_text(json.dumps(lib, indent=1)); tmp.replace(out)
    print(len(lib), "individuals ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
