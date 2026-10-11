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


def amodal_and_parts(vertices_cm: np.ndarray, faces: np.ndarray, whale: dict, camera: Camera, extra: bool = False):
    """Image of body-part ids (0 = no whale) for the whole whale, ignoring the water. With
    extra=True also the depth buffer (cm) and the body-coordinate image (fraction, angle)."""
    world = transform_mesh(vertices_cm, whale)
    uv, depth = camera.project(world)
    parts = part_of_vertices(vertices_cm)
    face_parts = np.array([np.bincount(parts[f]).argmax() for f in faces], np.int32)
    if not extra:
        return rasterize(uv, depth, faces, camera.width, camera.height, face_parts)
    coords = body_coordinates(vertices_cm)
    # Interpolating an angle across the 0/360 seam is wrong; interpolate its sin/cos.
    attributes = np.stack([coords[:, 0], np.sin(np.radians(coords[:, 1])), np.cos(np.radians(coords[:, 1]))], axis=1)
    image, zbuf, attrs = rasterize(uv, depth, faces, camera.width, camera.height, face_parts, attributes, return_extra=True)
    angle = (np.degrees(np.arctan2(attrs[..., 1], attrs[..., 2])) + 360.0) % 360.0
    body = np.stack([attrs[..., 0], angle], axis=-1)
    return image, zbuf, body


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
        return v[idx[best(values[idx])]] if len(idx) else None

    points = {
        "snout": pick(centre, x, np.argmax),
        "fluke_notch": pick(centre, x, np.argmin),
        "fluke_tip_left": pick(tail, y, np.argmin),
        "fluke_tip_right": pick(tail, y, np.argmax),
        "dorsal_fin_tip": v[np.argmax(z)],
        "pectoral_tip_left": pick(y < 0, z, np.argmin),
        "pectoral_tip_right": pick(y > 0, z, np.argmin),
    }
    if groups and "eye" in groups:
        eyes = v[groups["eye"]]
        points["eye_left"] = eyes[eyes[:, 1] < 0].mean(axis=0)
        points["eye_right"] = eyes[eyes[:, 1] >= 0].mean(axis=0)
    return {name: np.asarray(p, float) for name, p in points.items() if p is not None}


def project_landmarks(points: dict, whale: dict, camera: Camera, zbuf: np.ndarray, visible_mask: np.ndarray | None,
                      tolerance_cm: float = 40.0) -> dict:
    """Pixel positions of the landmarks; ``occluded`` if the body is in front of the point,
    ``in_label`` if the rendered visible mask covers the pixel (above water, not sprayed)."""
    out = {}
    for name, point in points.items():
        world = transform_mesh(point[None, :], whale)
        uv, depth = camera.project(world)
        u, v = float(uv[0, 0]), float(uv[0, 1])
        inside = 0 <= u < camera.width and 0 <= v < camera.height and depth[0] > 0
        entry = {"uv": [round(u, 1), round(v, 1)], "depth_m": round(float(depth[0]) / 100.0, 3), "in_frame": bool(inside)}
        if inside:
            col, row = int(u), int(v)
            entry["occluded"] = bool(zbuf[row, col] < depth[0] - tolerance_cm)
            entry["in_label"] = bool(visible_mask[row, col]) if visible_mask is not None else None
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
