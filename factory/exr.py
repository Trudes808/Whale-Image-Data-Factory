"""Minimal OpenEXR scanline writer / reader in numpy + zlib (no OpenEXR package needed).

Writes single-part scanline images with HALF (float16) or FLOAT (float32) channels, NO or ZIP compression
(zlib on blocks of 16 scanlines with the OpenEXR byte reordering and delta predictor), which Unreal's EXR
importer and any OpenEXR reader read. The reader handles what the writer writes (and ZIPS / NO files from other
tools with the same channel types), enough for the factory to read back exactly the texels it baked.

    write("T_Mom_c3.exr", {"R": a, "G": b, "B": c, "A": d}, half=True)     # arrays (H, W)
    ch = read("T_Mom_c3.exr")                                               # dict name -> float32 (H, W)

Channel order on disk is alphabetical (A, B, G, R), as the format requires; Unreal maps R, G, B, A by name.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

MAGIC = 20000630
NO_COMPRESSION, ZIPS_COMPRESSION, ZIP_COMPRESSION = 0, 2, 3
HALF, FLOAT = 1, 2
_LINES = {NO_COMPRESSION: 1, ZIPS_COMPRESSION: 1, ZIP_COMPRESSION: 16}


def _attr(name: str, typ: str, data: bytes) -> bytes:
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(data)) + data


def _predict_reorder(raw: bytes) -> bytes:
    b = np.frombuffer(raw, np.uint8)
    t = np.concatenate([b[0::2], b[1::2]])
    d = t.astype(np.int16)
    d[1:] = (t[1:].astype(np.int16) - t[:-1].astype(np.int16) + 128 + 256) % 256
    return d.astype(np.uint8).tobytes()


def _unpredict_reorder(data: bytes) -> bytes:
    d = np.frombuffer(data, np.uint8).astype(np.int64)
    d[1:] -= 128
    t = (np.cumsum(d) % 256).astype(np.uint8)
    n = len(t); half = (n + 1) // 2
    out = np.empty(n, np.uint8)
    out[0::2] = t[:half]; out[1::2] = t[half:]
    return out.tobytes()


def write(path, channels: dict, half: bool = True, compression: int = ZIP_COMPRESSION) -> Path:
    """Write channels (name -> (H, W) array) as a scanline EXR; returns the path."""
    path = Path(path)
    names = sorted(channels)
    arrs = [np.asarray(channels[n]) for n in names]
    H, W = arrs[0].shape
    dt = np.float16 if half else np.float32
    ptype = HALF if half else FLOAT
    arrs = [a.astype(dt) for a in arrs]
    chlist = b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", ptype, 0, 1, 1) for n in names) + b"\0"
    header = struct.pack("<ii", MAGIC, 2)
    header += _attr("channels", "chlist", chlist)
    header += _attr("compression", "compression", bytes([compression]))
    header += _attr("dataWindow", "box2i", struct.pack("<iiii", 0, 0, W - 1, H - 1))
    header += _attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, W - 1, H - 1))
    header += _attr("lineOrder", "lineOrder", bytes([0]))
    header += _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\0"
    lines = _LINES[compression]
    nblocks = (H + lines - 1) // lines
    chunks = []
    for b in range(nblocks):
        y0, y1 = b * lines, min(H, (b + 1) * lines)
        raw = b"".join(a[y].astype("<" + a.dtype.str[1:]).tobytes() for y in range(y0, y1) for a in arrs)
        if compression == NO_COMPRESSION:
            data = raw
        else:
            data = zlib.compress(_predict_reorder(raw), 6)
            if len(data) >= len(raw):
                data = raw
        chunks.append(struct.pack("<ii", y0, len(data)) + data)
    offset = len(header) + 8 * nblocks
    table = []
    for c in chunks:
        table.append(offset); offset += len(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header); f.write(struct.pack(f"<{nblocks}Q", *table))
        for c in chunks:
            f.write(c)
    return path


def read(path) -> dict:
    """Read a scanline EXR written by `write` (HALF / FLOAT channels, NO / ZIPS / ZIP) -> name -> float32 (H, W)."""
    buf = Path(path).read_bytes()
    magic, version = struct.unpack_from("<ii", buf, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: not an EXR file")
    if version & 0x200:
        raise ValueError(f"{path}: tiled EXR not supported")
    p = 8; attrs = {}
    while buf[p] != 0:
        e = buf.index(b"\0", p); name = buf[p:e].decode(); p = e + 1
        e = buf.index(b"\0", p); typ = buf[p:e].decode(); p = e + 1
        size, = struct.unpack_from("<i", buf, p); p += 4
        attrs[name] = (typ, buf[p:p + size]); p += size
    p += 1
    ch = []; q = 0; cl = attrs["channels"][1]
    while cl[q] != 0:
        e = cl.index(b"\0", q); n = cl[q:e].decode(); q = e + 1
        ptype, = struct.unpack_from("<i", cl, q); q += 16
        ch.append((n, ptype))
    comp = attrs["compression"][1][0]
    x0, y0, x1, y1 = struct.unpack("<iiii", attrs["dataWindow"][1])
    W, H = x1 - x0 + 1, y1 - y0 + 1
    lines = _LINES.get(comp)
    if lines is None:
        raise ValueError(f"{path}: compression {comp} not supported")
    nblocks = (H + lines - 1) // lines
    table = struct.unpack_from(f"<{nblocks}Q", buf, p)
    out = {n: np.zeros((H, W), np.float32) for n, _ in ch}
    for off in table:
        yb, size = struct.unpack_from("<ii", buf, off)
        data = buf[off + 8: off + 8 + size]
        nl = min(lines, y1 + 1 - yb)
        rawlen = sum(W * (2 if t == HALF else 4) for _, t in ch) * nl
        if comp != NO_COMPRESSION and size < rawlen:
            data = _unpredict_reorder(zlib.decompress(data))
        pos = 0
        for y in range(yb - y0, yb - y0 + nl):
            for n, t in ch:
                dt = np.dtype("<f2") if t == HALF else np.dtype("<f4")
                k = W * dt.itemsize
                out[n][y] = np.frombuffer(data[pos:pos + k], dt).astype(np.float32); pos += k
    return out
