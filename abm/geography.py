"""State geometry: Census boundary shapes, an Albers projection, centroids and distances.

The boundary file is the Census cartographic boundary shapefile (cb_*_us_state_20m),
downloaded once into data/geo/. Alaska and Hawaii are moved into the conventional
inset positions so the whole country fits one frame.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapefile

ROOT = Path(__file__).resolve().parent.parent
SHAPEFILE = ROOT / "data" / "geo" / "cb_2023_us_state_20m"

# Non-state territories are dropped; DC is kept as a state-like jurisdiction.
EXCLUDED_FIPS = {"72", "60", "66", "69", "78"}

# Albers Equal Area Conic for the contiguous states: central meridian, latitude of
# origin, and the two standard parallels.
_CONUS_PROJ = (-96.0, 37.5, 29.5, 45.5)

# Alaska and Hawaii are far enough off the national central meridian that the
# conic projection shears them badly, so each gets its own conic fitted to its
# own latitudes, then is scaled and dropped into an inset. `width` is the target
# width of the inset in projected metres; `x`/`y` are its lower-left corner.
_INSET = {
    "AK": {
        "proj": (-154.0, 50.5, 55.0, 65.0),
        "width": 1.45e6,
        "x": -2.40e6,
        "y": -2.55e6,
    },
    "HI": {
        "proj": (-157.0, 20.9, 8.0, 18.0),
        "width": 0.62e6,
        "x": -0.78e6,
        "y": -2.25e6,
    },
}

# Alaska's Aleutian chain crosses the antimeridian, which sends those rings to the
# far side of the map. Keep only the mainland and near islands, so the inset is
# scaled to Alaska proper rather than to the tail.
_AK_LON_RANGE = (-170.0, -128.0)


def albers(lon, lat, proj: tuple[float, float, float, float] = _CONUS_PROJ):
    """Project longitude/latitude (degrees) to Albers Equal Area Conic metres."""
    lon = np.radians(np.asarray(lon, dtype=float))
    lat = np.radians(np.asarray(lat, dtype=float))
    lon0, lat0 = math.radians(proj[0]), math.radians(proj[1])
    phi1, phi2 = math.radians(proj[2]), math.radians(proj[3])

    n = 0.5 * (math.sin(phi1) + math.sin(phi2))
    C = math.cos(phi1) ** 2 + 2 * n * math.sin(phi1)
    rho0 = math.sqrt(C - 2 * n * math.sin(lat0)) / n

    theta = n * (lon - lon0)
    rho = np.sqrt(np.maximum(C - 2 * n * np.sin(lat), 0.0)) / n

    R = 6_378_137.0  # WGS84 equatorial radius
    return R * rho * np.sin(theta), R * (rho0 - rho * np.cos(theta))


@lru_cache(maxsize=1)
def load_state_shapes() -> dict[str, list[np.ndarray]]:
    """Return {state abbreviation: list of projected (N, 2) polygon rings}."""
    if not SHAPEFILE.with_suffix(".shp").exists():
        raise FileNotFoundError(
            f"Missing boundary shapefile at {SHAPEFILE}.shp\n"
            "Run: python data/fetch_boundaries.py"
        )

    reader = shapefile.Reader(str(SHAPEFILE))
    shapes: dict[str, list[np.ndarray]] = {}

    for rec, shp in zip(reader.records(), reader.shapes()):
        if rec["STATEFP"] in EXCLUDED_FIPS:
            continue
        abbr = rec["STUSPS"]
        pts = np.asarray(shp.points, dtype=float)
        # Shapefile polygons store all rings end-to-end; `parts` gives ring starts.
        bounds = list(shp.parts) + [len(pts)]
        raw = [pts[s:e] for s, e in zip(bounds[:-1], bounds[1:])]
        raw = [r for r in raw if len(r) >= 3]
        if abbr == "AK":
            lo, hi = _AK_LON_RANGE
            raw = [r for r in raw if lo <= r[:, 0].mean() <= hi]

        proj = _INSET[abbr]["proj"] if abbr in _INSET else _CONUS_PROJ
        rings = []
        for ring in raw:
            x, y = albers(ring[:, 0], ring[:, 1], proj)
            rings.append(np.column_stack([x, y]))
        shapes[abbr] = rings

    for abbr, cfg in _INSET.items():
        if abbr not in shapes:
            continue
        rings = shapes[abbr]
        allpts = np.vstack(rings)
        x0, x1 = allpts[:, 0].min(), allpts[:, 0].max()
        y0 = allpts[:, 1].min()
        s = cfg["width"] / max(x1 - x0, 1.0)
        shapes[abbr] = [
            np.column_stack(
                [(r[:, 0] - x0) * s + cfg["x"], (r[:, 1] - y0) * s + cfg["y"]]
            )
            for r in rings
        ]

    return shapes


@lru_cache(maxsize=1)
def fips_to_abbr() -> dict[str, str]:
    reader = shapefile.Reader(str(SHAPEFILE))
    return {
        rec["STATEFP"]: rec["STUSPS"]
        for rec in reader.records()
        if rec["STATEFP"] not in EXCLUDED_FIPS
    }


@lru_cache(maxsize=1)
def geographic_centroids() -> dict[str, tuple[float, float]]:
    """True (lon, lat) centroid of each state's largest ring - used for distances."""
    reader = shapefile.Reader(str(SHAPEFILE))
    out: dict[str, tuple[float, float]] = {}
    for rec, shp in zip(reader.records(), reader.shapes()):
        if rec["STATEFP"] in EXCLUDED_FIPS:
            continue
        pts = np.asarray(shp.points, dtype=float)
        bounds = list(shp.parts) + [len(pts)]
        largest, best = None, -1.0
        for start, end in zip(bounds[:-1], bounds[1:]):
            ring = pts[start:end]
            if len(ring) < 3:
                continue
            x, y = ring[:, 0], ring[:, 1]
            area = abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            if area > best:
                best, largest = area, ring
        out[rec["STUSPS"]] = (float(largest[:, 0].mean()), float(largest[:, 1].mean()))
    return out


@lru_cache(maxsize=1)
def display_centroids() -> dict[str, tuple[float, float]]:
    """Projected centroid used for drawing flow arrows (respects AK/HI insets)."""
    out = {}
    for abbr, rings in load_state_shapes().items():
        largest = max(rings, key=len)
        out[abbr] = (float(largest[:, 0].mean()), float(largest[:, 1].mean()))
    return out


@lru_cache(maxsize=1)
def state_radii() -> dict[str, float]:
    """Radius in km of a circle with the same land area as the state.

    Uses the ALAND field of the Census boundary file, so it needs no extra data
    and introduces no fitted parameter. It stands for how far, on average, a
    resident must travel to leave their own state.
    """
    reader = shapefile.Reader(str(SHAPEFILE))
    out = {}
    for rec in reader.records():
        if rec["STATEFP"] in EXCLUDED_FIPS:
            continue
        area_km2 = float(rec["ALAND"]) / 1e6
        out[rec["STUSPS"]] = math.sqrt(area_km2 / math.pi)
    return out


def radius_array(order: list[str]) -> np.ndarray:
    radii = state_radii()
    return np.array([radii[a] for a in order], dtype=float)


def distance_matrix(order: list[str]) -> np.ndarray:
    """Great-circle distance in km between state centroids, in the given order."""
    cents = geographic_centroids()
    lon = np.radians([cents[a][0] for a in order])
    lat = np.radians([cents[a][1] for a in order])
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


# Land borders between states, used for the policy-spillover analysis.
ADJACENCY: dict[str, list[str]] = {
    "AL": ["FL", "GA", "MS", "TN"],
    "AK": [],
    "AZ": ["CA", "CO", "NV", "NM", "UT"],
    "AR": ["LA", "MS", "MO", "OK", "TN", "TX"],
    "CA": ["AZ", "NV", "OR"],
    "CO": ["AZ", "KS", "NE", "NM", "OK", "UT", "WY"],
    "CT": ["MA", "NY", "RI"],
    "DE": ["MD", "NJ", "PA"],
    "DC": ["MD", "VA"],
    "FL": ["AL", "GA"],
    "GA": ["AL", "FL", "NC", "SC", "TN"],
    "HI": [],
    "ID": ["MT", "NV", "OR", "UT", "WA", "WY"],
    "IL": ["IN", "IA", "KY", "MO", "WI"],
    "IN": ["IL", "KY", "MI", "OH"],
    "IA": ["IL", "MN", "MO", "NE", "SD", "WI"],
    "KS": ["CO", "MO", "NE", "OK"],
    "KY": ["IL", "IN", "MO", "OH", "TN", "VA", "WV"],
    "LA": ["AR", "MS", "TX"],
    "ME": ["NH"],
    "MD": ["DE", "DC", "PA", "VA", "WV"],
    "MA": ["CT", "NH", "NY", "RI", "VT"],
    "MI": ["IN", "OH", "WI"],
    "MN": ["IA", "ND", "SD", "WI"],
    "MS": ["AL", "AR", "LA", "TN"],
    "MO": ["AR", "IL", "IA", "KS", "KY", "NE", "OK", "TN"],
    "MT": ["ID", "ND", "SD", "WY"],
    "NE": ["CO", "IA", "KS", "MO", "SD", "WY"],
    "NV": ["AZ", "CA", "ID", "OR", "UT"],
    "NH": ["ME", "MA", "VT"],
    "NJ": ["DE", "NY", "PA"],
    "NM": ["AZ", "CO", "OK", "TX", "UT"],
    "NY": ["CT", "MA", "NJ", "PA", "VT"],
    "NC": ["GA", "SC", "TN", "VA"],
    "ND": ["MN", "MT", "SD"],
    "OH": ["IN", "KY", "MI", "PA", "WV"],
    "OK": ["AR", "CO", "KS", "MO", "NM", "TX"],
    "OR": ["CA", "ID", "NV", "WA"],
    "PA": ["DE", "MD", "NJ", "NY", "OH", "WV"],
    "RI": ["CT", "MA"],
    "SC": ["GA", "NC"],
    "SD": ["IA", "MN", "MT", "NE", "ND", "WY"],
    "TN": ["AL", "AR", "GA", "KY", "MS", "MO", "NC", "VA"],
    "TX": ["AR", "LA", "NM", "OK"],
    "UT": ["AZ", "CO", "ID", "NV", "NM", "WY"],
    "VT": ["MA", "NH", "NY"],
    "VA": ["DC", "KY", "MD", "NC", "TN", "WV"],
    "WA": ["ID", "OR"],
    "WV": ["KY", "MD", "OH", "PA", "VA"],
    "WI": ["IL", "IA", "MI", "MN"],
    "WY": ["CO", "ID", "MT", "NE", "SD", "UT"],
}


def adjacency_matrix(order: list[str]) -> np.ndarray:
    idx = {a: i for i, a in enumerate(order)}
    m = np.zeros((len(order), len(order)), dtype=bool)
    for a, neighbours in ADJACENCY.items():
        if a not in idx:
            continue
        for b in neighbours:
            if b in idx:
                m[idx[a], idx[b]] = True
    return m
