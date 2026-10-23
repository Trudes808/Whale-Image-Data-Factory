"""Pinhole projection of the whale mesh with the shot's camera, outside Unreal.

Used by the post-processing step for what the renderer does not give us directly: the
amodal silhouette (the whole whale, ignoring water), per-body-part masks, and the whale's
orientation relative to the camera. The camera model is Unreal's CineCamera: a filmback
of SENSOR_WIDTH_MM by SENSOR_HEIGHT_MM behind a lens of ``focal_mm``, output resolution
cropping to the image aspect. Validate against a render with ``check_against_mask``.

Frames: Unreal world is +X forward, +Y right, +Z up (left-handed); camera looks down its
local +X with +Y right and +Z up in the image.
"""

from __future__ import annotations

import math

import numpy as np

from factory.poses import rotation_matrix

# Unreal 5 CineCameraActor default filmback ("16:9 Digital Film"); calibrated 2026-10-11
# against rendered masks (98.6-98.9 % of visible pixels inside the projection, bounding
# boxes within a few pixels). 36 x 20.25 (the DSLR preset) was clearly wrong.
SENSOR_WIDTH_MM = 23.76
SENSOR_HEIGHT_MM = 13.365

# Body parts by position along the mesh's X axis (fraction of length from the tail, 0,
# to the snout, 1), for the K9239 humpback mesh in its converted frame.
BODY_PARTS = {
    "fluke": (0.0, 0.14),
    "peduncle": (0.14, 0.32),
    "dorsal_back": (0.32, 0.55),
    "mid_body": (0.55, 0.72),
    "head": (0.72, 1.0),
}
PART_IDS = {name: index + 1 for index, name in enumerate(BODY_PARTS)}


class Camera:
    def __init__(self, location, rotation: dict, focal_mm: float, width: int, height: int,
                 sensor_width_mm: float = SENSOR_WIDTH_MM, sensor_height_mm: float = SENSOR_HEIGHT_MM):
        self.location = np.asarray(location, float)
        self.basis = rotation_matrix(rotation.get("roll", 0.0), rotation.get("pitch", 0.0), rotation.get("yaw", 0.0))
        self.focal_mm = float(focal_mm)
        self.width, self.height = int(width), int(height)
        # Unreal fits the horizontal FOV to the sensor width; the vertical FOV follows the
        # output aspect ratio (the filmback is cropped if the aspects differ).
        self.fx = self.focal_mm / sensor_width_mm * self.width
        self.fy = self.fx
        self.cx, self.cy = self.width / 2.0, self.height / 2.0

    def to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """World -> camera coordinates (x forward, y right, z up)."""
        return (np.asarray(points_world, float) - self.location) @ self.basis.T

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates (u right, v down) and depth (cm) for each point."""
        cam = self.to_camera(points_world)
        depth = cam[:, 0]
        safe = np.where(depth > 1e-3, depth, 1e-3)
        u = self.cx + self.fx * cam[:, 1] / safe
        v = self.cy - self.fy * cam[:, 2] / safe
        return np.stack([u, v], axis=1), depth


def rasterize(uv: np.ndarray, depth: np.ndarray, faces: np.ndarray, width: int, height: int,
              face_values: np.ndarray | None = None, vertex_attributes: np.ndarray | None = None,
              return_extra: bool = False):
    """Scanline-fill triangles into an int image (0 = empty). face_values gives the value
    per face (default 1); nearer faces win where they overlap (z-buffered). With
    vertex_attributes (N, K) the attributes are interpolated with barycentric weights;
    return_extra also returns the depth buffer (cm, inf where empty) and the attribute image."""
    image = np.zeros((height, width), np.int32)
    zbuf = np.full((height, width), np.inf)
    attrs = None if vertex_attributes is None else np.zeros((height, width, vertex_attributes.shape[1]), np.float32)
    values = np.ones(len(faces), np.int32) if face_values is None else np.asarray(face_values, np.int32)
    tri_uv = uv[faces]
    tri_depth = depth[faces]
    keep = (tri_depth > 1.0).all(axis=1)
    order = np.argsort(-tri_depth.mean(axis=1))  # far to near
    for i in order:
        if not keep[i]:
            continue
        pts = tri_uv[i]
        x0, y0 = int(max(0, np.floor(pts[:, 0].min()))), int(max(0, np.floor(pts[:, 1].min())))
        x1, y1 = int(min(width - 1, np.ceil(pts[:, 0].max()))), int(min(height - 1, np.ceil(pts[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = pts
        det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(det) < 1e-9:
            continue
        l1 = ((bx - xs) * (cy - ys) - (cx - xs) * (by - ys)) / det
        l2 = ((cx - xs) * (ay - ys) - (ax - xs) * (cy - ys)) / det
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not inside.any():
            continue
        z = l1 * tri_depth[i, 0] + l2 * tri_depth[i, 1] + l3 * tri_depth[i, 2]
        region = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        nearer = inside & (z < zbuf[region])
        zbuf[region][nearer] = z[nearer]
        image[region][nearer] = values[i]
        if attrs is not None:
            a, b, c = vertex_attributes[faces[i]]
            interp = l1[..., None] * a + l2[..., None] * b + l3[..., None] * c
            attrs[region][nearer] = interp[nearer]
    if return_extra:
        return image, zbuf, attrs
    return image


def transform_mesh(vertices_cm: np.ndarray, whale: dict) -> np.ndarray:
    rot = whale["rotation"]
    return vertices_cm @ rotation_matrix(rot.get("roll", 0.0), rot.get("pitch", 0.0), rot.get("yaw", 0.0)) + np.asarray(whale["location"], float)


def part_of_vertices(vertices_cm: np.ndarray) -> np.ndarray:
    """Body-part id per mesh vertex, by position along the body (mesh frame)."""
    x = vertices_cm[:, 0]
    frac = (x - x.min()) / max(x.max() - x.min(), 1e-6)
    ids = np.zeros(len(x), np.int32)
    for name, (lo, hi) in BODY_PARTS.items():
        ids[(frac >= lo) & (frac < hi if hi < 1.0 else frac <= hi)] = PART_IDS[name]
    return ids


def body_coordinates(vertices_cm: np.ndarray) -> np.ndarray:
    """Canonical body coordinates per vertex: (fraction along the body from tail 0 to snout 1,
    angle around the body axis in degrees 0..360 with 0 = top/back, 90 = the whale's right
    side, 180 = belly, 270 = left side). Stable across poses, for comparing surface features
    (scars, barnacles, pigmentation) at the same body location between photos."""
    x = vertices_cm[:, 0]
    frac = (x - x.min()) / max(x.max() - x.min(), 1e-6)
    angle = (np.degrees(np.arctan2(vertices_cm[:, 1], vertices_cm[:, 2])) + 360.0) % 360.0
    return np.stack([frac, angle], axis=1)


def amodal_and_parts(vertices_cm: np.ndarray, faces: np.ndarray, whale: dict, camera: Camera, extra: bool = False,
                     canonical_cm: np.ndarray | None = None, world_z: bool = False, canonical_xyz: bool = False,
                     deformed_xyz: bool = False):
    """Image of body-part ids (0 = no whale) for the whole whale, ignoring the water. With
    extra=True also the depth buffer (cm) and the body-coordinate image (fraction, angle);
    with world_z=True as well, a further array: the world z (cm) of the whale surface at each
    pixel (height above the mean water plane), nan where there is no whale; with
    canonical_xyz=True, a further (H, W, 3) array: the point on the *undeformed* mesh (cm, mesh
    frame: x tail -> snout, y right, z up) seen at each pixel, nan where there is no whale.
    Return order: image, zbuf, body[, height][, canon].
    vertices_cm is the geometry to project (possibly deformed); canonical_cm (default: the
    same array) is the undeformed mesh that body coordinates and parts are measured on."""
    if canonical_cm is None:
        canonical_cm = vertices_cm
    world = transform_mesh(vertices_cm, whale)
    uv, depth = camera.project(world)
    parts = part_of_vertices(canonical_cm)
    face_parts = np.array([np.bincount(parts[f]).argmax() for f in faces], np.int32)
    if not extra:
        return rasterize(uv, depth, faces, camera.width, camera.height, face_parts)
    coords = body_coordinates(canonical_cm)
    # Interpolating an angle across the 0/360 seam is wrong; interpolate its sin/cos.
    attributes = np.concatenate([np.stack([coords[:, 0], np.sin(np.radians(coords[:, 1])), np.cos(np.radians(coords[:, 1])), world[:, 2]], axis=1),
                                 np.asarray(canonical_cm, float), np.asarray(vertices_cm, float)], axis=1)
    image, zbuf, attrs = rasterize(uv, depth, faces, camera.width, camera.height, face_parts, attributes, return_extra=True)
    angle = (np.degrees(np.arctan2(attrs[..., 1], attrs[..., 2])) + 360.0) % 360.0
    body = np.stack([attrs[..., 0], angle], axis=-1)
    out = [image, zbuf, body]
    if world_z:
        out.append(np.where(image > 0, attrs[..., 3], np.nan))
    if canonical_xyz:
        out.append(np.where((image > 0)[..., None], attrs[..., 4:7], np.nan).astype(np.float32))
    if deformed_xyz:   # the point on the posed (scaled, bent) mesh in the whale frame, for pose fitting
        out.append(np.where((image > 0)[..., None], attrs[..., 7:10], np.nan).astype(np.float32))
    return tuple(out)


def landmarks(vertices_cm: np.ndarray, groups: dict | None = None) -> dict:
    """Named points on the mesh (mesh frame, cm), found geometrically so they hold for any
    rigidly posed copy of the model. Pectoral fins hang down on this model."""
    v = vertices_cm
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    centre = np.abs(y) < 30
    length = x.max() - x.min()
    tail = x < x.min() + 0.15 * length
    def pick(selection, values, best):
        idx = np.where(selection)[0]
        return int(idx[best(values[idx])]) if len(idx) else None

    indices = {
        "snout": pick(centre, x, np.argmax),
        "fluke_notch": pick(centre, x, np.argmin),
        "fluke_tip_left": pick(tail, y, np.argmin),
        "fluke_tip_right": pick(tail, y, np.argmax),
        "dorsal_fin_tip": int(np.argmax(z)),
        "pectoral_tip_left": pick(y < 0, z, np.argmin),
        "pectoral_tip_right": pick(y > 0, z, np.argmin),
    }
    # run03 additions: the valleys and bases that pin the parts (chin, blowhole, dorsal fin front/back,
    # tail base top/bottom, pectoral bases, belly). All geometric picks on the canonical mesh.
    def band(x0, half):
        return np.abs(x - x0) < half
    xm = x.min()
    extra = {
        "chin": pick(centre & (x > xm + 0.9 * length), z, np.argmin),
        "blowhole": pick(centre & band(xm + 0.80 * length, 0.025 * length), z, np.argmax),
        "belly": pick(centre & band(xm + 0.55 * length, 0.03 * length), z, np.argmin),
        "tail_base_top": pick(centre & band(xm + 0.21 * length, 0.02 * length), z, np.argmax),
        "tail_base_bottom": pick(centre & band(xm + 0.21 * length, 0.02 * length), z, np.argmin),
    }
    # spine_end: the dorsal midline point at the fluke root, i.e. the narrowest cross-section of the tail stock before
    # the blades spread (the end of the spine's dorsal ridge). With the notch it gives the fluke's midline axis, and its
    # visibility tells the fluke's upper side from its underside.
    xs = np.arange(xm + 0.08 * length, xm + 0.25 * length, 0.005 * length)
    widths = [np.abs(y[(x >= a) & (x < a + 0.005 * length)]).max() if ((x >= a) & (x < a + 0.005 * length)).any() else np.inf for a in xs]
    x_root = float(xs[int(np.argmin(widths))] + 0.0025 * length)
    extra["spine_end"] = pick((np.abs(y) < 8) & band(x_root, 0.01 * length), z, np.argmax)
    if indices.get("dorsal_fin_tip") is not None:
        xt = x[indices["dorsal_fin_tip"]]
        extra["dorsal_fin_front"] = pick(centre & band(xt + 0.06 * length, 0.015 * length), z, np.argmax)
        extra["dorsal_fin_back"] = pick(centre & band(xt - 0.05 * length, 0.015 * length), z, np.argmax)
    try:
        from factory import deform as _deform
        for side_name, key in (("left", "pectoral_base_left"), ("right", "pectoral_base_right")):
            region = _deform.fin_regions(v).get(side_name)
            if region is not None and len(region["indices"]):
                idx = region["indices"]
                extra[key] = int(idx[np.argmin(np.abs(y[idx]))])
    except Exception:
        pass
    indices.update({k: i for k, i in extra.items() if i is not None})
    # run05: edge chains as extra named points, so the fin and fluke trailing edges are labelled per shot like keypoints
    # (fin_te_00.. tip -> base along the rear of the dorsal fin, fin_le_00.. tip -> base along its front,
    #  fluke_te_00.. left tip -> notch -> right tip along the hindmost edge)
    indices.update(edge_chains(v, indices))
    if groups and "eye" in groups:
        eyes = np.asarray(groups["eye"])
        left, right = eyes[v[eyes, 1] < 0], eyes[v[eyes, 1] >= 0]
        if len(left):
            indices["eye_left"] = [int(i) for i in left]
        if len(right):
            indices["eye_right"] = [int(i) for i in right]
    return {name: i for name, i in indices.items() if i is not None}


def edge_chains(v: np.ndarray, indices: dict, n_fin: int = 12, n_fluke: int = 24) -> dict:
    """Vertex indices along the dorsal fin's trailing and leading edges (height bins from the fin base to the tip) and
    along the fluke's trailing edge (span bins from the left tip to the right tip), picked geometrically on the
    canonical mesh so that they deform with it."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    out = {}
    tip, front, back = indices.get("dorsal_fin_tip"), indices.get("dorsal_fin_front"), indices.get("dorsal_fin_back")
    if tip is not None and front is not None and back is not None:
        # The fin's side silhouette: the upper envelope z_max(x) of the vertices near the midline between the back
        # junction and the front junction, above the base line (back -> front). The trailing edge is the envelope
        # from the back junction up to the tip, the leading edge from the tip down to the front junction; each is
        # resampled at n_fin positions along its length (nearest envelope vertex). Vertex indices, so the chains
        # follow an individual's fin (height, sweep, trailing-edge wear) and the pose.
        xb, zb = x[back], z[back]; xf, zf = x[front], z[front]; xt = x[tip]
        base = lambda xx: zb + (xx - xb) * (zf - zb) / max(xf - xb, 1e-6)
        band = (x >= xb - 2.0) & (x <= xf + 2.0) & (np.abs(y) < 30.0)
        above = band & (z > base(x) + 2.0)
        sel = np.nonzero(above)[0]
        if len(sel) > 20:
            bins = np.floor((x[sel] - xb) / 2.0).astype(int)                       # 2 cm x bins -> the highest vertex each
            env = {}
            for i, b in zip(sel, bins):
                if b not in env or z[i] > z[env[b]]:
                    env[b] = int(i)
            env_idx = np.array([env[b] for b in sorted(env)])                       # back -> front
            k_tip = int(np.argmin(np.abs(x[env_idx] - xt)))

            def walk(chain):
                """Follow the envelope away from the tip while it keeps descending: stop at the first saddle (the
                envelope rises again by > 3 cm: the back's hump in front of the fin) or below the base line."""
                keep, zmin = [], np.inf
                for i in chain:
                    if z[i] > zmin + 3.0 or z[i] < base(x[i]) + 2.0:
                        break
                    keep.append(int(i)); zmin = min(zmin, float(z[i]))
                return np.array(keep, dtype=int)
            trailing = walk(env_idx[: k_tip + 1][::-1])                             # tip -> back junction
            leading = walk(env_idx[k_tip:])                                         # tip -> front junction
            for prefix, chain in (("fin_te", trailing), ("fin_le", leading)):
                if len(chain) < 3:
                    continue
                pts = v[chain][:, [0, 2]]
                arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
                for k, t in enumerate(np.linspace(0.0, arc[-1], n_fin)):
                    out[f"{prefix}_{k:02d}"] = int(chain[int(np.argmin(np.abs(arc - t)))])   # 00 = tip, n-1 = junction
    lt, rt = indices.get("fluke_tip_left"), indices.get("fluke_tip_right")
    if lt is not None and rt is not None:
        length = x.max() - x.min(); tail = x < x.min() + 0.15 * length
        yl, yr = y[lt], y[rt]
        bins = np.linspace(min(yl, yr), max(yl, yr), n_fluke + 1)
        for k in range(n_fluke):
            sel = np.nonzero(tail & (y >= bins[k]) & (y < bins[k + 1]))[0]
            if len(sel):
                out[f"fluke_te_{k:02d}"] = int(sel[np.argmin(x[sel])])                       # hindmost at this span position
    return out


def landmark_positions(indices: dict, vertices_cm: np.ndarray) -> dict:
    """Landmark points (cm) on a (possibly deformed) copy of the mesh, from landmark indices."""
    out = {}
    for name, i in indices.items():
        out[name] = vertices_cm[i].mean(axis=0) if isinstance(i, list) else vertices_cm[i]
    return out


def project_landmarks(points: dict, whale: dict, camera: Camera, zbuf: np.ndarray, visible_mask: np.ndarray | None,
                      tolerance_cm: float = 40.0, water_z_cm: float | None = None, canonical: dict | None = None) -> dict:
    """Pixel positions of the landmarks; ``occluded`` if the body is in front of the point,
    ``in_label`` if the rendered visible mask covers the pixel (above water, not sprayed).
    With water_z_cm, a ``state`` per keypoint: visible (in frame, not occluded, in the label),
    under_water (in frame, not occluded, below the mean water plane), hidden (in frame, not in
    the label but above the water: spray, wave or mask edge), occluded (behind the body),
    out_of_frame. Also the point's whale-frame position (cm, deformed) and, from ``canonical``,
    its canonical (undeformed, unit-size) position."""
    out = {}
    for name, point in points.items():
        world = transform_mesh(point[None, :], whale)
        uv, depth = camera.project(world)
        u, v = float(uv[0, 0]), float(uv[0, 1])
        inside = 0 <= u < camera.width and 0 <= v < camera.height and depth[0] > 0
        entry = {"uv": [round(u, 1), round(v, 1)], "depth_m": round(float(depth[0]) / 100.0, 3), "in_frame": bool(inside),
                 "xyz_whale_cm": [round(float(c), 1) for c in point]}
        if canonical is not None and name in canonical:
            entry["xyz_canonical_cm"] = [round(float(c), 1) for c in canonical[name]]
        if water_z_cm is not None:
            entry["height_above_water_m"] = round(float(world[0, 2] - water_z_cm) / 100.0, 3)
        if inside:
            col, row = int(u), int(v)
            entry["occluded"] = bool(zbuf[row, col] < depth[0] - tolerance_cm)
            if visible_mask is not None:   # a thin tip can fall a pixel outside the mask: look at a 5x5 neighbourhood
                r0, r1, c0, c1 = max(row - 2, 0), min(row + 3, visible_mask.shape[0]), max(col - 2, 0), min(col + 3, visible_mask.shape[1])
                entry["in_label"] = bool(visible_mask[r0:r1, c0:c1].any())
            else:
                entry["in_label"] = None
        if water_z_cm is not None:
            if not inside:
                state = "out_of_frame"
            elif entry["occluded"]:
                state = "occluded"
            elif entry["in_label"]:
                state = "visible"
            elif world[0, 2] < water_z_cm:
                state = "under_water"
            else:
                state = "hidden"
            entry["state"] = state
        out[name] = entry
    return out


def pose_in_camera(whale: dict, camera: Camera) -> dict:
    """The whale's rigid pose in the camera frame (x forward, y right, z up): rotation
    matrix rows are the whale's own axes expressed in camera coordinates; translation is
    the whale origin in cm."""
    rot = whale["rotation"]
    whale_basis = rotation_matrix(rot.get("roll", 0.0), rot.get("pitch", 0.0), rot.get("yaw", 0.0))
    rotation = whale_basis @ camera.basis.T
    translation = camera.to_camera(np.asarray(whale["location"], float)[None, :])[0]
    return {"rotation": np.round(rotation, 6).tolist(), "translation_cm": np.round(translation, 2).tolist()}


def orientation(whale: dict, camera: Camera) -> dict:
    """How the whale is seen: heading relative to the camera and viewing angles.

    relative_bearing_deg: 0 = whale seen from directly behind (swimming away), 90 = from
    its left side... measured clockwise from the whale's nose direction to the direction
    from whale to camera, in the horizontal plane. view_elevation_deg: camera elevation
    above the whale's horizontal plane. Plus the whale's own pitch/roll/yaw and the
    nose direction's component toward the camera.
    """
    rot = whale["rotation"]
    basis = rotation_matrix(rot.get("roll", 0.0), rot.get("pitch", 0.0), rot.get("yaw", 0.0))
    nose = basis[0]
    to_camera = camera.location - np.asarray(whale["location"], float)
    horizontal = to_camera.copy()
    horizontal[2] = 0.0
    distance = float(np.linalg.norm(to_camera))
    bearing = math.degrees(math.atan2(horizontal[1], horizontal[0]) - math.atan2(nose[1], nose[0])) % 360.0
    elevation = math.degrees(math.atan2(to_camera[2], np.linalg.norm(horizontal)))
    return {
        "pitch_deg": float(rot.get("pitch", 0.0)),
        "roll_deg": float(rot.get("roll", 0.0)),
        "yaw_deg": float(rot.get("yaw", 0.0)),
        "relative_bearing_deg": round(bearing, 2),
        "view_elevation_deg": round(elevation, 2),
        "distance_m": round(distance / 100.0, 2),
        "nose_toward_camera": round(float(nose @ to_camera / max(distance, 1e-6)), 4),
    }


def load_obj(path, with_groups: bool = False):
    """Vertices and faces of an OBJ; with_groups also returns {usemtl name: vertex indices}."""
    vertices, faces, groups, current = [], [], {}, None
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(t) for t in line.split()[1:4]])
                if current is not None:
                    groups.setdefault(current, []).append(len(vertices) - 1)
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
            elif line.startswith("usemtl "):
                current = line.split()[1]
    result = (np.asarray(vertices, float), np.asarray(faces, np.int64))
    return result + ({k: np.asarray(v) for k, v in groups.items()},) if with_groups else result


def check_against_mask(amodal: np.ndarray, visible_mask: np.ndarray) -> dict:
    """The rendered visible-whale mask should lie inside the projected amodal silhouette."""
    amodal_b = amodal > 0
    inside = float((visible_mask & amodal_b).sum() / max(visible_mask.sum(), 1))
    return {"visible_pixels": int(visible_mask.sum()), "amodal_pixels": int(amodal_b.sum()),
            "visible_inside_amodal": round(inside, 4),
            "visible_fraction": round(float(visible_mask.sum() / max(amodal_b.sum(), 1)), 4)}


def spine_curve(posed_cm: np.ndarray, canonical_cm: np.ndarray, n_slices: int = 13) -> dict:
    """A model-independent description of how the body is bent: the posed centroids of n_slices slices of
    the body along its canonical length (tail -> snout), in the whale frame (cm), with the pitch (xz) and
    yaw (xy) angle of every segment, degrees, relative to the straight canonical body. Lets a model learn
    the body's bend directly from the picture (whatever produced it: clip, hump, rig offsets)."""
    x = canonical_cm[:, 0]; lo, hi = float(x.min()), float(x.max())
    edges = np.linspace(lo, hi, n_slices + 1)
    pts = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (x >= a) & (x < b) if b < hi else (x >= a) & (x <= b)
        pts.append(posed_cm[sel].mean(0) if sel.any() else np.array([np.nan] * 3))
    pts = np.array(pts)
    seg = np.diff(pts, axis=0)
    pitch = np.degrees(np.arctan2(seg[:, 2], np.maximum(seg[:, 0], 1e-6)))
    yaw = np.degrees(np.arctan2(seg[:, 1], np.maximum(seg[:, 0], 1e-6)))
    return {"slices": n_slices, "points_cm": np.round(pts, 1).tolist(), "segment_pitch_deg": np.round(pitch, 2).tolist(),
            "segment_yaw_deg": np.round(yaw, 2).tolist(), "fraction": np.round((edges[:-1] + edges[1:]) / 2.0, 1).tolist()}
