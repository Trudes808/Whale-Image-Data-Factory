"""run06 round 8: individual library with the pectoral planform (narrower chord, a few extra leading-edge tubercles),
made from the round-7 library (blade fields kept).

    python3 -m factory.library_round8 third_party_assets/humpback_4k/individuals_run06d.json third_party_assets/humpback_4k/individuals_run06e.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    src, out = (argv or sys.argv[1:])[:2]
    lib = json.loads(Path(src).read_text())
    rng = np.random.default_rng(8)
    for k in sorted(lib):
        d = lib[k]
        calf = bool(d.get("calf"))
        d["pec_chord"] = float(rng.uniform(0.72, 0.85) if calf else rng.uniform(0.62, 0.8))
        d["pec_knobs"] = int(rng.integers(2, 6))
        d["pec_knob_cm"] = float(rng.uniform(2.0, 4.0) if calf else rng.uniform(3.0, 6.0))
    tmp = Path(out).with_suffix(".json.tmp"); tmp.write_text(json.dumps(lib, indent=1)); tmp.replace(out)
    print(len(lib), "individuals ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
