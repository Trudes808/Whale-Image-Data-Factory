"""Surface poses for a rigid whale mesh, plus a boat camera that frames them.

The humpback asset is a static mesh in a cruising pose (see factory.humpback_asset), so a
"pose" here is a rigid transform: a rotation (Unreal roll/pitch/yaw, degrees) and a height
chosen so that the highest point of the rotated mesh sits ``clearance_m`` above calm water.
That is enough for the classic field-photo views: logging at the surface, the blow, the
arched back before a dive, a fluke-up dive, a pectoral fin raised, and a spyhop.
Non-rigid motion (swim cycle, fluke curl) needs a rig and comes later.

    python3 -m factory.poses MESH.obj [POSE ...] -o shots/        # one JSON per pose

Each JSON is what ``wf_apply_shot.py`` applies in the editor: whale location/rotation,
camera location/rotation/focal length, and some statistics for QA.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from factory import seastate
from factory import splash as splash_module


@dataclass(frozen=True)
class Camera:
    """Boat camera relative to the whale: bearing (deg, clockwise from the whale's nose
    direction when seen from above), distance and height above water (m), focal length."""

    bearing_deg: float = 100.0
    distance_m: float = 45.0
    height_m: float = 3.0
    focal_mm: float = 85.0


@dataclass(frozen=True)
class Pose:
    pitch_deg: float
    roll_deg: float
    clearance_m: float
    description: str
    camera: Camera = field(default_factory=Camera)
    yaw_deg: float = 20.0
    # White water: "crown" (impact splash at the waterline), "wash" (white water along the
    # waterline), or None; splash_scale sets size, splash_amount how much. See factory/splash.py.
    splash: str | None = None
    splash_scale: float = 1.0
    splash_amount: float = 1.0




POSES: dict[str, Pose] = {
    # Camera: the CineCamera's default 36 mm-wide filmback, so 50 mm at 60 m frames about
    # 43 m of water, enough to show the whole 16 m whale with sea around it.
    "logging": Pose(0.0, 0.0, 1.0, "resting at the surface: back, dorsal fin and blowhole exposed",
                    Camera(100.0, 60.0, 3.0, 50.0)),
    "blow": Pose(7.0, 0.0, 1.2, "surfacing to breathe: head and blowhole up, back rolling through",
                 Camera(60.0, 60.0, 3.0, 50.0), splash="wash", splash_scale=0.5, splash_amount=0.5),
    "arch": Pose(-16.0, 0.0, 1.5, "arched back and dorsal fin high just before a dive",
                 Camera(110.0, 55.0, 3.0, 50.0), splash="wash", splash_scale=0.5, splash_amount=0.5),
    "fluke_up": Pose(-55.0, 0.0, 3.0, "terminal dive: peduncle and flukes clear of the water",
                     Camera(180.0, 60.0, 3.0, 70.0), splash="crown", splash_scale=0.45, splash_amount=0.6),
    "breach": Pose(65.0, 25.0, 9.0, "breach: most of the body out of the water, twisting, with a crown of spray",
                   Camera(120.0, 75.0, 3.0, 70.0), splash="crown", splash_scale=1.0, splash_amount=1.2),
    "pec_up": Pose(0.0, 75.0, 1.6, "rolled on one side, a pectoral fin raised (pec slap)",
                   Camera(90.0, 55.0, 3.0, 50.0), splash="crown", splash_scale=0.4, splash_amount=0.6),
    "spyhop": Pose(70.0, 0.0, 3.0, "head raised vertically out of the water",
                   Camera(30.0, 50.0, 3.0, 70.0), splash="wash", splash_scale=0.6, splash_amount=0.6),
    "awash": Pose(0.0, 0.0, 0.25, "barely breaking the surface: only the hump shows between waves",
                  Camera(100.0, 60.0, 3.0, 50.0)),
    "submerged": Pose(0.0, 0.0, -0.5, "just below the surface: a negative (no visible whale)",
                      Camera(100.0, 60.0, 3.0, 50.0)),
    # run03: close-ups where only a part of the whale shows, and rolled whales (the hard views in field photos)
    "blow_sliver": Pose(6.0, 0.0, 0.45, "just the blowhole and the top of the back break the surface",
                        Camera(70.0, 60.0, 3.0, 85.0), splash="wash", splash_scale=0.35, splash_amount=0.4),
    "fin_tip": Pose(0.0, 82.0, 0.7, "rolled on one side, only the pectoral fin tip above the water",
                    Camera(90.0, 55.0, 3.0, 85.0), splash="crown", splash_scale=0.3, splash_amount=0.4),
    "fluke_edge": Pose(-72.0, 0.0, 1.1, "end of the dive: only the fluke tips are still above the water",
                       Camera(180.0, 60.0, 3.0, 85.0), splash="crown", splash_scale=0.35, splash_amount=0.5),
    "peduncle": Pose(-35.0, 0.0, 2.0, "tail stock arched high before the flukes come up",
                     Camera(140.0, 60.0, 3.0, 85.0), splash="wash", splash_scale=0.5, splash_amount=0.5),
    "spyhop_low": Pose(72.0, 0.0, 1.3, "only the rostrum out of the water",
                       Camera(30.0, 50.0, 3.0, 85.0), splash="wash", splash_scale=0.4, splash_amount=0.4),
    "roll_belly": Pose(0.0, 160.0, 1.2, "rolled belly-up at the surface, fins in the air",
                       Camera(90.0, 55.0, 3.0, 85.0), splash="crown", splash_scale=0.5, splash_amount=0.6),
    # run04: the commonest field shot. The whale breathes, then arches: blowhole, back and dorsal fin clearly out
    # of the water as it slides through and resubmerges. The body shape comes from the start of the rig's clip;
    # the clearance is measured on the back (flukes excluded), so how much back shows varies while the flukes stay under.
    "back_arch": Pose(-5.0, 0.0, 1.1, "breath then arch: blowhole to dorsal fin and back out of the water, sliding through",
                      Camera(100.0, 60.0, 3.0, 50.0), splash="wash", splash_scale=0.5, splash_amount=0.5),
}
# categories whose clearance is measured on the back only (body-frame x > BACK_X_MIN_CM: no flukes / tail stock)
BACK_HEIGHT_CATEGORIES = {"back_arch"}
BACK_X_MIN_CM = -350.0


def load_obj_vertices(path: str | Path) -> np.ndarray:
    vertices = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(t) for t in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"no vertices in {path}")
    return np.asarray(vertices, dtype=float)


def rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Unreal's FRotator as a matrix whose rows are the rotated X, Y, Z axes.

    world = local @ M. Positive pitch raises the nose (+X toward +Z), positive yaw turns the
    nose toward +Y (clockwise from above), positive roll drops the right side (+Y toward -Z).
    """
    sr, cr = math.sin(math.radians(roll_deg)), math.cos(math.radians(roll_deg))
    sp, cp = math.sin(math.radians(pitch_deg)), math.cos(math.radians(pitch_deg))
    sy, cy = math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg))
    return np.array([
        [cp * cy, cp * sy, sp],
        [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
        [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp],
    ])


def place(vertices_cm: np.ndarray, pose: Pose, water_z_cm: float = 0.0, top_mask: np.ndarray | None = None) -> dict:
    """Whale transform for a pose, and the camera that frames it. All values in cm/deg. ``top_mask`` selects
    the vertices whose highest point sits at the clearance (default: the whole whale)."""
    rotated = vertices_cm @ rotation_matrix(pose.roll_deg, pose.pitch_deg, pose.yaw_deg)
    top = rotated[top_mask] if top_mask is not None and top_mask.any() else rotated
    location = np.array([0.0, 0.0, water_z_cm + pose.clearance_m * 100.0 - top[:, 2].max()])
    world = rotated + location
    above = world[world[:, 2] > water_z_cm]
    above_fraction = float(len(above) / len(world))
    # Aim at what shows above the water; for a submerged whale, at its back.
    aim = above.mean(axis=0) if len(above) else np.array([0.0, 0.0, water_z_cm])
    bearing = math.radians(pose.yaw_deg + pose.camera.bearing_deg)
    cam_location = aim + np.array([math.cos(bearing) * pose.camera.distance_m * 100.0,
                                   math.sin(bearing) * pose.camera.distance_m * 100.0, 0.0])
    cam_location[2] = water_z_cm + pose.camera.height_m * 100.0
    cam_rotation = look_at_rotation(cam_location, aim)
    splash = make_splash(pose, rotated, location, water_z_cm)
    return {
        "splash": splash,
        "whale": {
            "location": location.round(2).tolist(),
            "rotation": {"roll": pose.roll_deg, "pitch": pose.pitch_deg, "yaw": pose.yaw_deg},
        },
        "camera": {
            "location": cam_location.round(2).tolist(),
            "rotation": cam_rotation,
            "focal_mm": pose.camera.focal_mm,
        },
        "stats": {
            "highest_point_cm": float(world[:, 2].max()),
            "lowest_point_cm": float(world[:, 2].min()),
            "vertex_fraction_above_water": above_fraction,
            "aim_point": aim.round(2).tolist(),
        },
    }


def make_splash(pose: Pose, rotated: np.ndarray, location: np.ndarray, water_z_cm: float, seed: int = 0) -> dict | None:
    """White water for the pose, centred where the body crosses the surface."""
    if not pose.splash:
        return None
    world = rotated + location
    band = world[np.abs(world[:, 2] - water_z_cm) < 80.0]
    if len(band) < 4:
        return None
    centre = np.array([band[:, 0].mean(), band[:, 1].mean(), water_z_cm])
    radius = float(np.clip(np.linalg.norm(band[:, :2] - centre[:2], axis=1).max(), 200.0, 700.0))
    kind = pose.splash
    height = {"crown": 0.6 * radius, "wash": 0.2 * radius}[kind] * pose.splash_scale
    return splash_module.splash_spec(kind, centre, radius * pose.splash_scale, height, seed, amount=pose.splash_amount)


def look_at_rotation(eye: np.ndarray, target: np.ndarray) -> dict:
    """Unreal rotator (roll 0) that points +X from eye toward target."""
    d = np.asarray(target, float) - np.asarray(eye, float)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return {"roll": 0.0, "pitch": round(pitch, 4), "yaw": round(yaw, 4)}


def make_shot(name: str, vertices_cm: np.ndarray, pose: Pose | None = None, water_z_cm: float = 0.0) -> dict:
    pose = pose or POSES[name]
    shot = {"name": name, "pose": asdict(pose)}
    shot.update(place(vertices_cm, pose, water_z_cm))
    return shot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", type=Path, help="whale OBJ in Unreal units (cm, +X nose, +Z up)")
    parser.add_argument("poses", nargs="*", default=list(POSES), help=f"pose names (default: all: {', '.join(POSES)})")
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("--water-z-cm", type=float, default=0.0)
    parser.add_argument("--sea", choices=list(seastate.SEA_STATES), default=None,
                        help="sea-state preset to add (file names get a _<sea> suffix)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    unknown = [p for p in args.poses if p not in POSES]
    if unknown:
        parser.error(f"unknown poses: {unknown}; choose from {list(POSES)}")
    vertices = load_obj_vertices(args.mesh)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.poses:
        shot = make_shot(name, vertices, water_z_cm=args.water_z_cm)
        if args.sea:
            seastate.apply_to_shot(shot, args.sea, args.seed)
            shot["name"] = f"{name}_{args.sea}"
        splash_module.ensure_obj(shot.get("splash"), args.out_dir / "splash", shot["name"])
        path = args.out_dir / f"{shot['name']}.json"
        path.write_text(json.dumps(shot, indent=2))
        stats = shot["stats"]
        print(f"{path}: highest {stats['highest_point_cm'] / 100:.2f} m, "
              f"{stats['vertex_fraction_above_water']:.0%} of vertices above water")
    return 0


if __name__ == "__main__":
    sys.exit(main())
