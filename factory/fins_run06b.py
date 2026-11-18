"""run06 round 5: the individuals_run06b library (third_party_assets/humpback_4k/individuals_run06b.json): the run06
library with new dorsal fins, drawn per individual id (deterministic): height scale log-normal round HMED (sd HSPREAD),
sweep uniform in [SWEEP_LO, SWEEP_HI] cm (negative = tip forward), hump uniform in [HUMP_LO, HUMP_HI] cm on 85 %. With
config whale.dorsal_front_ratio 7.0 (front base where the hump's slope starts) the made library is
    python3 factory/fins_run06b.py third_party_assets/humpback_4k/individuals_run06.json \
        third_party_assets/humpback_4k/individuals_run06b.json 1.0 0.6 -60 -30 5 35
and gives base / height 2.38 / 2.58 / 2.95 (q25 / median / q75) and tip position 0.04 / 0.14 / 0.16 from the back base
(real: 2.2-3.8 around 2.3-3.1, tip 0.03-0.2 around 0.12; Dinov3 experiments/parts_weak/fin_definition.json)."""
import json, sys, hashlib
import numpy as np
src, out = sys.argv[1], sys.argv[2]
hmed, hsp, s0, s1, h0, h1 = map(float, sys.argv[3:9])
lib = json.load(open(src))
for k, v in lib.items():
    if v.get("calf"):
        continue
    rng = np.random.default_rng(int(hashlib.sha1(f"fin5:{k}".encode()).hexdigest()[:8], 16))
    v["dorsal_height"] = float(hmed * np.exp(rng.normal(0, hsp)))
    v["dorsal_sweep_cm"] = float(rng.uniform(s0, s1))
    v["dorsal_hump_cm"] = float(rng.uniform(h0, h1)) if rng.uniform() < 0.85 else 0.0
json.dump(lib, open(out, "w"), indent=1)
