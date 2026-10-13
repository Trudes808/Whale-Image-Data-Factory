"""Convert the purchased skinned humpback (third_party_assets/humpback_4k, glTF) into the project's
asset layout, like factory.humpback_asset does for the K9239 model: a rest-pose OBJ in the project
frame (+X nose, +Y right, +Z up, cm; Y written mirrored for Unreal's importer; groups "body" and
"eye"), importer-friendly textures (base colour, roughness from the metallic-roughness map, normal
with the green channel flipped, occlusion), and an info JSON. Per-shot posed meshes come from
factory.skinned at render time.

    python3 -m factory.humpback4k third_party_assets/humpback_4k/src_gltf third_party_assets/humpback_4k/converted
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from factory.skinned import Rig

NAME = "SM_Humpback4k"


def write_obj(path: Path, groups: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], comment: str) -> list[str]:
    """groups: (name, vertices cm project frame, faces, uv). Faces wound so the outward normal matches."""
    lines = [f"# {comment}", f"mtllib {path.stem}.mtl"]
    base = 0; written = []
    for name, v, faces, uv in groups:
        tri = v[faces]; centre = v.mean(0)
        geometric = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        outward = np.einsum("ij,ij->i", geometric, tri.mean(1) - centre)
        if (outward < 0).mean() > 0.5:                       # outward winding in the project frame, as the K9239 OBJ is written
            faces = faces[:, [0, 2, 1]]
        lines.append(f"usemtl {name}"); written.append(name)
        lines.extend(f"v {x:.4f} {-y:.4f} {z:.4f}" for x, y, z in v)
        lines.extend(f"vt {u:.6f} {1.0 - t:.6f}" for u, t in uv)
        for a, b, c in faces + base + 1:
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
        base += len(v)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".mtl").write_text("".join(f"newmtl {g}\n" for g in written))
    return written


def neutralise_throat(tex: "np.ndarray") -> "np.ndarray":
    """The purchased texture paints the underside of the head and the chest pink, which reads as an open red
    mouth whenever the chin shows (breaches, spyhops, rolled whales). Real humpbacks have a pale pleated
    throat there: those texels are turned into the ventral white, keeping their shading as luminance."""
    from PIL import Image, ImageFilter
    t = tex.astype(np.float32)
    R, G, B = t[..., 0], t[..., 1], t[..., 2]
    pink = (R > G + 25) & (R > B + 15) & (R > 90)
    soft = np.asarray(Image.fromarray((pink * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))).astype(np.float32) / 255.0
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    target = np.clip(lum * (205.0 / 125.0), 0, 235)[..., None] * np.array([0.98, 0.97, 0.97], np.float32)
    out = t * (1 - soft[..., None]) + target * soft[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def convert(src: Path, out: Path) -> dict:
    rig = Rig.load(src / "scene.gltf")
    out.mkdir(parents=True, exist_ok=True)
    body = rig.rest
    centre = np.array([(body[:, 0].max() + body[:, 0].min()) / 2, 0.0, (body[:, 2].max() + body[:, 2].min()) / 2])
    groups = [("body", body - centre, rig.faces, rig.uv)]
    eyes = rig.extra_meshes(None)
    if eyes:
        ev = np.concatenate([e["vertices"] for e in eyes]) - centre
        ef = np.concatenate([e["faces"] + sum(len(x["vertices"]) for x in eyes[:k]) for k, e in enumerate(eyes)])
        eu = np.concatenate([e["uv"] for e in eyes])
        groups.append(("eye", ev, ef, eu))
    written = write_obj(out / f"{NAME}.obj", groups, "purchased 4k humpback (skinned), rest pose, converted by factory.humpback4k")
    tex = src / "textures"
    Image.open(tex / "Mesh_Humpback_Whale_LOD0_baseColor.png").convert("RGB").save(out / "T_Humpback4k_BaseColor.png")
    mr = np.asarray(Image.open(tex / "Mesh_Humpback_Whale_LOD0_metallicRoughness.png").convert("RGB"))
    Image.fromarray(mr[..., 1]).save(out / "T_Humpback4k_Roughness.png")
    normal = np.asarray(Image.open(tex / "Mesh_Humpback_Whale_LOD0_normal.png").convert("RGB")).copy()
    normal[..., 1] = 255 - normal[..., 1]
    Image.fromarray(normal).save(out / "T_Humpback4k_Normal.png")
    occl = src.parent / "src_zip" / "textures" / "Humpback5H90_Mesh_Humpback_Whale_LOD0_Occl.png"
    if occl.exists():
        Image.open(occl).convert("L").save(out / "T_Humpback4k_Occlusion.png")
    info = {"source": "purchased 4k humpback whale (Fab), skinned, 25 joints, one 29 s clip", "centre_shift_cm": centre.tolist(),
            "length_cm": float(body[:, 0].max() - body[:, 0].min()), "bounds_min_cm": (body - centre).min(0).tolist(), "bounds_max_cm": (body - centre).max(0).tolist(),
            "vertices": int(len(body)), "faces": int(len(rig.faces)), "groups": written, "clip_s": rig.clip_end, "gltf": str(src / "scene.gltf")}
    (out / "humpback4k_info.json").write_text(json.dumps(info, indent=2))
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path); p.add_argument("out", type=Path)
    a = p.parse_args(argv)
    info = convert(a.src, a.out); print(json.dumps({k: v for k, v in info.items() if k != "gltf"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
