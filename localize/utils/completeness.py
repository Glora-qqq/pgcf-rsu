"""按 CSI 缺失表把无效帧的特征置零。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import torch

_SRC_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_JSON = _SRC_ROOT / "data_analysis" / "results" / "completeness.json"
_DEFAULT_TXT = _SRC_ROOT / "data_analysis" / "results-train" / "completeness_output.txt"

_CACHE: Dict[str, Dict[str, Dict[str, Set[int]]]] = {}
_LOADED_FROM: Optional[str] = None

def default_completeness_path() -> Path:
    if _DEFAULT_JSON.is_file():
        return _DEFAULT_JSON
    if _DEFAULT_TXT.is_file():
        return _DEFAULT_TXT
    raise FileNotFoundError(
        f"找不到 CSI 完备性文件:\n  {_DEFAULT_JSON}\n  {_DEFAULT_TXT}"
    )

def _ranges_to_set(ranges: Iterable) -> Set[int]:
    out: Set[int] = set()
    for item in ranges:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            lo, hi = int(item[0]), int(item[1])
        else:
            continue
        if hi < lo:
            lo, hi = hi, lo
        out.update(range(lo, hi + 1))
    return out

def _load_from_json(path: Path) -> Dict[str, Dict[str, Dict[str, Set[int]]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    root = data.get("by_scene_vehicle_rsf", data)
    out: Dict[str, Dict[str, Dict[str, Set[int]]]] = {}
    for scene, vehicles in root.items():
        out[scene] = {}
        for vehicle, rsfs in vehicles.items():
            out[scene][vehicle] = {}
            for rsf, meta in rsfs.items():
                ranges = meta.get("empty_ranges") or []
                out[scene][vehicle][rsf] = _ranges_to_set(ranges)
    return out

def _parse_range_token(tok: str) -> Tuple[int, int]:
    tok = tok.strip().strip("[]")
    if "-" in tok:
        a, b = tok.split("-", 1)
        return int(a), int(b)
    v = int(tok)
    return v, v

def _load_from_txt(path: Path) -> Dict[str, Dict[str, Dict[str, Set[int]]]]:

    text = path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, Dict[str, Dict[str, Set[int]]]] = {}
    scene = None
    rsf = None
    vehicle = None
    collecting = False
    buf: list[str] = []

    scene_re = re.compile(r"^(Scene\d+)\s*:")
    rsf_re = re.compile(r"^\s*(RSF\d+)\s*:")
    veh_re = re.compile(r"^\s*(\S+)\s+\((RSF\d+)\)\s*:")
    empty_re = re.compile(r"Empty/Zero ranges", re.I)

    def flush_ranges():
        nonlocal collecting, buf, scene, vehicle, rsf
        if not (collecting and scene and vehicle and rsf and buf):
            collecting = False
            buf = []
            return
        joined = " ".join(buf)
        tokens = re.findall(r"\[\d+(?:-\d+)?\]", joined)
        s: Set[int] = set()
        for t in tokens:
            lo, hi = _parse_range_token(t)
            s.update(range(lo, hi + 1))
        out.setdefault(scene, {}).setdefault(vehicle, {})[rsf] = s
        collecting = False
        buf = []

    for line in text.splitlines():
        m = scene_re.match(line)
        if m:
            flush_ranges()
            scene = m.group(1)
            rsf = None
            vehicle = None
            continue
        m = rsf_re.match(line)
        if m:
            flush_ranges()
            rsf = m.group(1)
            vehicle = None
            continue
        m = veh_re.match(line)
        if m:
            flush_ranges()
            vehicle = m.group(1)
            rsf = m.group(2)
            continue
        if empty_re.search(line):
            flush_ranges()
            collecting = True
            buf = []
            continue
        if collecting:
            if line.strip() == "" or scene_re.match(line) or rsf_re.match(line) or veh_re.match(line):
                flush_ranges()
                m = scene_re.match(line)
                if m:
                    scene = m.group(1)
                m = rsf_re.match(line)
                if m:
                    rsf = m.group(1)
                m = veh_re.match(line)
                if m:
                    vehicle = m.group(1)
                    rsf = m.group(2)
                continue
            if "[" in line:
                buf.append(line)
    flush_ranges()
    return out

def load_csi_invalid_map(path: Optional[Path] = None) -> Dict[str, Dict[str, Dict[str, Set[int]]]]:
    global _CACHE, _LOADED_FROM
    path = Path(path) if path else default_completeness_path()
    key = str(path.resolve())
    if _CACHE and _LOADED_FROM == key:
        return _CACHE
    if path.suffix.lower() == ".json":
        _CACHE = _load_from_json(path)
    else:
        _CACHE = _load_from_txt(path)
    _LOADED_FROM = key
    n = sum(len(rs) for sc in _CACHE.values() for vh in sc.values() for rs in vh.values())
    print(f"[CSI-completeness] loaded {path.name}: {len(_CACHE)} scenes, invalid-frame sets={n}")
    return _CACHE

def invalid_frame_set(scene: str, vehicle: str, rsf: str, path: Optional[Path] = None) -> Set[int]:
    m = load_csi_invalid_map(path)
    scene_map = m.get(scene) or m.get(scene.strip())
    if not scene_map:
        return set()
    veh = vehicle
    if veh not in scene_map:
        for k in scene_map:
            if k.lower() == vehicle.lower():
                veh = k
                break
    rsf_map = scene_map.get(veh) or {}
    if rsf in rsf_map:
        return rsf_map[rsf]
    for k, v in rsf_map.items():
        if k.upper() == rsf.upper():
            return v
    return set()

def zero_csi_where_invalid(
    csi_features: torch.Tensor,
    frame_ids: torch.Tensor,
    scene: str,
    vehicle: str,
    rsf: str,
    path: Optional[Path] = None,
) -> Tuple[torch.Tensor, int]:

    invalid = invalid_frame_set(scene, vehicle, rsf, path)
    if not invalid or csi_features.numel() == 0:
        return csi_features, 0
    ids = frame_ids.detach().cpu().numpy().astype(np.int64).reshape(-1)
    mask = np.array([int(f) in invalid for f in ids], dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return csi_features, 0
    out = csi_features.clone()
    out[torch.from_numpy(mask)] = 0
    return out, n
