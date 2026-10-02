"""
Multi-relational parcel graph construction.

All parcels are a single node type; relations differ by edge type. Every edge
type is made symmetric (both directions stored) and de-duplicated, and the
summary reports both directed edges and unique undirected relations as counted.

Edge types (definitions match the manuscript, Section 3.1):

``spatial_adjacency``
    Lots whose boundaries touch (within ``adjacency_tol_ft`` to absorb
    digitising gaps). Weight = shared boundary length / lot perimeter.
``proximity``
    k nearest lots (centroid distance) within ``proximity_radius_ft``.
    Weight = 1 / (1 + d / 100 ft). This is a distance relation, *not* a
    line-of-sight computation.
``functional_similarity``
    Among each lot's k nearest neighbours, those with the identical MapPLUTO
    land-use code. Weight 1.
``street_frontage``
    Lots whose address is on the same street (house number removed, street
    numbers retained, e.g. "EAST 45 STREET"), linked to their nearest lots on
    that street within ``street_radius_ft``. Weight 1.
``regulatory_coupling``
    k nearest lots within the same zoning district (``ZoneDist1``) within
    ``regulatory_radius_ft``. Weight 1.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

ALL_EDGE_TYPES = [
    "spatial_adjacency",
    "proximity",
    "functional_similarity",
    "street_frontage",
    "regulatory_coupling",
]

RELATION_NAMES = {
    "spatial_adjacency": ("parcel", "adjacent_to", "parcel"),
    "proximity": ("parcel", "near", "parcel"),
    "functional_similarity": ("parcel", "same_use_as", "parcel"),
    "street_frontage": ("parcel", "same_street_as", "parcel"),
    "regulatory_coupling": ("parcel", "same_zone_as", "parcel"),
}

_SUFFIX = {"ST": "STREET", "AVE": "AVENUE", "AV": "AVENUE", "PL": "PLACE", "BLVD": "BOULEVARD",
           "RD": "ROAD", "DR": "DRIVE", "SQ": "SQUARE", "TER": "TERRACE", "E": "EAST", "W": "WEST"}


def parse_street_name(address: str) -> Optional[str]:
    """Strip the house number from an address and normalise the street name.

    >>> parse_street_name("123 WEST 45 STREET")
    'WEST 45 STREET'
    >>> parse_street_name("12-14 E 4TH ST")
    'EAST 4 STREET'
    """
    if not isinstance(address, str) or not address.strip():
        return None
    tokens = address.upper().replace(",", " ").split()
    while tokens and re.fullmatch(r"[\d\-/]+[A-Z]?", tokens[0]):
        tokens = tokens[1:]
    if not tokens:
        return None
    out = []
    for t in tokens:
        t = re.sub(r"^(\d+)(ST|ND|RD|TH)$", r"\1", t)
        out.append(_SUFFIX.get(t, t))
    return " ".join(out)


def _symmetrise(src: np.ndarray, dst: np.ndarray, w: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a symmetric, de-duplicated, self-loop-free edge list (max weight kept)."""
    if len(src) == 0:
        return src, dst, w
    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])
    ww = np.concatenate([w, w])
    keep = s != d
    s, d, ww = s[keep], d[keep], ww[keep]
    key = s.astype(np.int64) * n + d
    order = np.lexsort((-ww, key))
    key, s, d, ww = key[order], s[order], d[order], ww[order]
    first = np.ones(len(key), dtype=bool)
    first[1:] = key[1:] != key[:-1]
    return s[first], d[first], ww[first]


class ParcelGraphBuilder:
    """Build a PyG ``HeteroData`` graph with one parcel node type and several relations."""

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        features: pd.DataFrame,
        edge_types: Optional[List[str]] = None,
        k_neighbors: int = 8,
        adjacency_tol_ft: float = 1.0,
        proximity_radius_ft: float = 300.0,
        street_radius_ft: float = 600.0,
        regulatory_radius_ft: float = 1000.0,
        street_k: int = 2,
        regulatory_k: int = 5,
    ):
        self.gdf = gdf.reset_index(drop=True)
        self.features = features.reset_index(drop=True)
        self.edge_types = list(edge_types or ALL_EDGE_TYPES)
        unknown = set(self.edge_types) - set(ALL_EDGE_TYPES)
        if unknown:
            raise ValueError(f"Unknown edge types: {unknown}")
        self.k = k_neighbors
        self.adjacency_tol_ft = adjacency_tol_ft
        self.proximity_radius_ft = proximity_radius_ft
        self.street_radius_ft = street_radius_ft
        self.regulatory_radius_ft = regulatory_radius_ft
        self.street_k = street_k
        self.regulatory_k = regulatory_k
        self.n = len(self.gdf)
        self.xy = self.gdf[["x", "y"]].values if "x" in self.gdf else np.column_stack(
            [self.gdf.geometry.centroid.x, self.gdf.geometry.centroid.y])
        self.tree = cKDTree(self.xy)
        self.edges: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ builders
    def _knn(self, k: int):
        k_eff = min(k + 1, self.n)
        d, idx = self.tree.query(self.xy, k=k_eff)
        if k_eff == 1:
            d, idx = d[:, None], idx[:, None]
        return d[:, 1:], idx[:, 1:]

    def build_spatial_adjacency(self):
        geoms = self.gdf.geometry.values
        buffered = shapely.buffer(geoms, self.adjacency_tol_ft)
        tree = shapely.STRtree(geoms)
        src, dst = tree.query(buffered, predicate="intersects")
        keep = src != dst
        src, dst = src[keep], dst[keep]
        if len(src) == 0:
            return src, dst, np.zeros(0)
        shared = shapely.length(shapely.intersection(shapely.boundary(geoms[src]), buffered[dst]))
        perim = shapely.length(geoms[src])
        w = np.clip(shared / np.maximum(perim, 1e-9), 0, 1)
        keep = shared > 0
        return src[keep], dst[keep], w[keep]

    def build_proximity(self):
        d, idx = self._knn(self.k)
        rows = np.repeat(np.arange(self.n), idx.shape[1])
        d, idx = d.ravel(), idx.ravel()
        keep = np.isfinite(d) & (d <= self.proximity_radius_ft)
        return rows[keep], idx[keep], 1.0 / (1.0 + d[keep] / 100.0)

    def build_functional_similarity(self):
        _, idx = self._knn(self.k)
        lu = self.gdf["land_use"].astype(str).values
        rows = np.repeat(np.arange(self.n), idx.shape[1])
        cols = idx.ravel()
        keep = lu[rows] == lu[cols]
        return rows[keep], cols[keep], np.ones(keep.sum())

    def build_street_frontage(self):
        streets = self.gdf["address"].map(parse_street_name)
        valid = streets.notna().values
        labels = np.where(valid, streets.fillna("").values, None)
        idx = np.where(valid)[0]
        s, d, w = self._group_knn_subset(idx, labels[valid], self.street_k, self.street_radius_ft)
        return s, d, w

    def _group_knn_subset(self, idx, labels, k, radius):
        if len(idx) == 0:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        saved = self.xy
        src, dst = [], []
        for _, members in pd.Series(idx).groupby(labels):
            m = members.values
            if len(m) < 2:
                continue
            kk = min(k + 1, len(m))
            dd, ii = cKDTree(saved[m]).query(saved[m], k=kk)
            dd, ii = dd[:, 1:], ii[:, 1:]
            r = np.repeat(m, ii.shape[1])
            c = m[ii.ravel()]
            keep = dd.ravel() <= radius
            src.append(r[keep])
            dst.append(c[keep])
        if not src:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        s, d = np.concatenate(src), np.concatenate(dst)
        return s, d, np.ones(len(s))

    def build_regulatory_coupling(self):
        zones = self.gdf["zone_district"].astype(str).values
        valid = zones != "UNKNOWN"
        idx = np.where(valid)[0]
        return self._group_knn_subset(idx, zones[valid], self.regulatory_k, self.regulatory_radius_ft)

    # ------------------------------------------------------------------ assembly
    def build(self) -> HeteroData:
        builders = {
            "spatial_adjacency": self.build_spatial_adjacency,
            "proximity": self.build_proximity,
            "functional_similarity": self.build_functional_similarity,
            "street_frontage": self.build_street_frontage,
            "regulatory_coupling": self.build_regulatory_coupling,
        }
        data = HeteroData()
        data["parcel"].x = torch.tensor(self.features.values, dtype=torch.float32)
        data["parcel"].num_nodes = self.n
        for et in self.edge_types:
            s, d, w = builders[et]()
            s, d, w = _symmetrise(np.asarray(s, dtype=np.int64), np.asarray(d, dtype=np.int64),
                                  np.asarray(w, dtype=np.float64), self.n)
            self.edges[et] = (s, d, w)
            rel = RELATION_NAMES[et]
            data[rel].edge_index = torch.tensor(np.vstack([s, d]), dtype=torch.long)
            data[rel].edge_attr = torch.tensor(w, dtype=torch.float32).unsqueeze(-1)
            logger.info("%s: %d directed edges", et, len(s))
        return data

    # Backwards-compatible name.
    build_heterogeneous_graph = build

    def summary(self) -> Dict:
        """Exact counts of directed edges and unique undirected relations per type."""
        out = {"n_nodes": int(self.n), "edge_types": {}}
        total_dir = 0
        total_und = 0
        for et, (s, d, _) in self.edges.items():
            und = int(np.unique(np.minimum(s, d) * self.n + np.maximum(s, d)).size)
            out["edge_types"][et] = {"directed": int(len(s)), "undirected": und}
            total_dir += len(s)
            total_und += und
        out["total_directed"] = int(total_dir)
        out["total_undirected"] = int(total_und)
        deg = np.zeros(self.n)
        for s, _, _ in self.edges.values():
            np.add.at(deg, s, 1)
        out["isolated_nodes"] = int((deg == 0).sum())
        return out
