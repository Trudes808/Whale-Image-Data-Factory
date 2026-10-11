"""Seeded shot sampler: config + mesh -> shotlist.jsonl and render batches.

    python3 -m factory.shotlist factory/config/poc_boat.json MESH.obj -o data/renders/run01 --count 2000 --seed 1

Every shot is a dict that ``wf_shot.apply_shot`` understands (whale, camera, splash, sun,
fog, exposure, waves, ocean) plus bookkeeping (id, pose category, sea state, look). Shots
sharing a sea state and water look are grouped into batches (``batches/batch_NNN.json``),
each rendered in one editor session by ``wf_build_batch.py``.

Variance covered: pose category and continuous pitch/roll/yaw/height jitter (so the same
category appears at many submersion depths), camera bearing all around the whale,
distance, boat and elevated heights, focal length chosen from how much of the frame the
whale should fill, aim offset, roll, sea state, water colour, sun elevation and azimuth,
overcast or clear, haze, exposure, spray, and negatives (submerged, or whale out of frame).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from factory import poses, seastate, splash
from factory.project import SENSOR_WIDTH_MM

WHALE_APPARENT_WIDTH_M = 13.0  # typical on-screen extent of a 16 m whale seen obliquely


def weighted_choice(rng: np.random.Generator, weights: dict) -> str:
    names = list(weights)
    probabilities = np.array([weights[n] for n in names], float)
    return names[rng.choice(len(names), p=probabilities / probabilities.sum())]


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_pose(rng: np.random.Generator, config: dict) -> tuple[str, poses.Pose]:
    name = weighted_choice(rng, config["pose_weights"])
    base = poses.POSES[name]
    jitter = config["pose_jitter"]
    roll_jitter = jitter["breach_roll_deg"] if name == "breach" else jitter["roll_deg"]
    lo, hi = jitter["clearance_scale"]
    clearance = base.clearance_m * rng.uniform(lo, hi) if base.clearance_m > 0 else base.clearance_m * rng.uniform(0.5, 3.0)
    pose = replace(base,
                   pitch_deg=base.pitch_deg + rng.uniform(-jitter["pitch_deg"], jitter["pitch_deg"]),
                   roll_deg=base.roll_deg + rng.uniform(-roll_jitter, roll_jitter),
                   clearance_m=clearance,
                   yaw_deg=float(rng.uniform(0.0, 360.0)))
    return name, pose


def sample_camera(rng: np.random.Generator, config: dict, pose: poses.Pose, placed: dict, water_z_cm: float) -> dict:
    cam = config["camera"]
    distance_m = log_uniform(rng, *cam["distance_m"])
    elevated = rng.uniform() < cam["elevated_fraction"]
    height_m = rng.uniform(*(cam["elevated_height_m"] if elevated else cam["boat_height_m"]))
    bearing = rng.uniform(0.0, 360.0)
    aim = np.array(placed["stats"]["aim_point"], float)
    angle = math.radians(pose.yaw_deg + bearing)
    location = aim + np.array([math.cos(angle) * distance_m * 100.0, math.sin(angle) * distance_m * 100.0, 0.0])
    location[2] = water_z_cm + height_m * 100.0
    fill = rng.uniform(*cam["fill_fraction"])
    focal = fill * SENSOR_WIDTH_MM * distance_m / WHALE_APPARENT_WIDTH_M
    focal = float(np.clip(focal, *cam["focal_mm"]))
    rotation = poses.look_at_rotation(location, aim)
    # Aim jitter: shift the view by a fraction of the frame, in angle.
    hfov = 2 * math.degrees(math.atan(SENSOR_WIDTH_MM / 2 / focal))
    rotation["yaw"] = round(rotation["yaw"] + rng.uniform(-1, 1) * cam["aim_jitter_frame_fraction"] * hfov / 2, 4)
    rotation["pitch"] = round(rotation["pitch"] + rng.uniform(-1, 1) * cam["aim_jitter_frame_fraction"] * hfov * 9 / 16 / 2, 4)
    rotation["roll"] = round(float(rng.uniform(-cam["roll_deg"], cam["roll_deg"])), 3)
    return {"location": location.round(2).tolist(), "rotation": rotation, "focal_mm": round(focal, 2),
            "bearing_deg": round(bearing, 2), "distance_m": round(distance_m, 2), "height_m": round(height_m, 2),
            "elevated": bool(elevated), "fill_fraction": round(fill, 3)}


def sample_lighting(rng: np.random.Generator, config: dict, camera_yaw: float) -> tuple[dict, dict, float, bool]:
    light = config["lighting"]
    overcast = rng.uniform() < light["overcast_fraction"]
    elevation = rng.uniform(*light["sun_elevation_deg"])
    # Azimuth relative to the camera's view direction: 0 = sun behind the camera, 180 = backlit.
    relative = rng.uniform(0.0, 360.0)
    sun_yaw = (camera_yaw + 180.0 + relative) % 360.0  # the light's forward vector points *from* the sun
    intensity = rng.uniform(*(light["overcast_intensity"] if overcast else light["clear_intensity"]))
    sun = {"pitch": round(-elevation, 2), "yaw": round(sun_yaw, 2), "intensity": round(intensity, 2),
           "elevation_deg": round(elevation, 2), "relative_azimuth_deg": round(relative, 2)}
    fog_lo, fog_hi = light["fog_density"]
    fog = {"density": round(log_uniform(rng, fog_lo * (4 if overcast else 1), fog_hi), 5)}
    exposure = round(float(rng.uniform(*light["exposure_ev"])), 3)
    return sun, fog, exposure, overcast


def make_shot(index: int, rng: np.random.Generator, config: dict, vertices_cm: np.ndarray) -> dict:
    category, pose = sample_pose(rng, config)
    spray_cfg = config["splash"]
    if pose.splash and rng.uniform() > spray_cfg["probability"].get(category, 1.0):
        pose = replace(pose, splash=None)                      # this time, no white water
    elif pose.splash:
        lo, hi = spray_cfg["scale_jitter"]
        pose = replace(pose, splash_scale=pose.splash_scale * float(rng.uniform(lo, hi)),
                       splash_amount=pose.splash_amount * float(rng.uniform(*spray_cfg["amount_jitter"])))
    elif pose.clearance_m > 0 and rng.uniform() < spray_cfg["extra_wash_fraction"]:
        pose = replace(pose, splash="wash", splash_scale=float(rng.uniform(0.3, 0.8)), splash_amount=float(rng.uniform(0.3, 0.8)))
    placed = poses.place(vertices_cm, pose)
    camera = sample_camera(rng, config, pose, placed, 0.0)
    no_whale = rng.uniform() < config["no_whale_in_frame_fraction"]
    if no_whale:
        camera["rotation"]["yaw"] = round(camera["rotation"]["yaw"] + rng.choice([-1, 1]) * rng.uniform(30.0, 70.0), 4)
    sun, fog, exposure, overcast = sample_lighting(rng, config, camera["rotation"]["yaw"])
    look = weighted_choice(rng, {name: spec["weight"] for name, spec in config["looks"].items()})
    sea = weighted_choice(rng, config["sea_state_weights"])
    splash_spec = placed["splash"]
    shot = {
        "id": f"{index:06d}",
        "name": f"{index:06d}_{category}_{sea}",
        "category": category,
        "negative": bool(no_whale or placed["stats"]["vertex_fraction_above_water"] == 0.0),
        "no_whale_in_frame": bool(no_whale),
        "overcast": bool(overcast),
        "look": look,
        "pose": {k: v for k, v in poses.asdict(pose).items() if k != "camera"},
        "whale": placed["whale"],
        "camera": camera,
        "splash": splash_spec,
        "sun": sun,
        "fog": fog,
        "exposure_ev": exposure,
        "stats": placed["stats"],
        "ocean": {"vectors": dict(config["looks"][look]["vectors"])},
    }
    seastate.apply_to_shot(shot, sea, seed=index)
    return shot


def make_batches(shots: list[dict], batch_size: int) -> list[dict]:
    groups: dict[tuple, list] = {}
    for shot in shots:
        groups.setdefault((shot["sea_state"], shot["look"]), []).append(shot)
    batches = []
    for (sea, look), members in sorted(groups.items()):
        for start in range(0, len(members), batch_size):
            chunk = members[start:start + batch_size]
            batches.append({
                "index": len(batches),
                "sea_state": sea,
                "look": look,
                "waves": chunk[0]["waves"],
                "ocean": chunk[0]["ocean"],
                "shots": chunk,
            })
    return batches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    vertices = poses.load_obj_vertices(args.mesh)
    rng = np.random.default_rng(args.seed)
    shots = [make_shot(i, rng, config, vertices) for i in range(args.count)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        splash.ensure_obj(shot.get("splash"), args.out_dir / "splash", shot["id"])
    (args.out_dir / "run_config.json").write_text(json.dumps({"config": config, "seed": args.seed, "count": args.count,
                                                              "mesh": str(args.mesh)}, indent=2))
    with open(args.out_dir / "shotlist.jsonl", "w") as handle:
        for shot in shots:
            handle.write(json.dumps(shot) + "\n")
    batch_dir = args.out_dir / "batches"
    batch_dir.mkdir(exist_ok=True)
    batches = make_batches(shots, config["batch_size"])
    for batch in batches:
        batch["resolution"] = config["resolution"]
        batch["frames_per_shot"] = config["frames_per_shot"]
        (batch_dir / f"batch_{batch['index']:03d}.json").write_text(json.dumps(batch))
    categories = {}
    for shot in shots:
        categories[shot["category"]] = categories.get(shot["category"], 0) + 1
    print(f"{len(shots)} shots in {len(batches)} batches -> {args.out_dir}")
    print("categories:", dict(sorted(categories.items())))
    print("negatives:", sum(s["negative"] for s in shots), "no whale in frame:", sum(s["no_whale_in_frame"] for s in shots))
    return 0


if __name__ == "__main__":
    sys.exit(main())
