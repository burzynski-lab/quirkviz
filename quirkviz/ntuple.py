"""Reader for the HitAnalysis ntuple written by quirkviz.dump_hits.

Si hit positions are global: HitAnalysis applies the local-to-global transform
via GeoSiHit::getGlobalPosition(). With ExtraTruthBranches on, each Si hit also
carries the pdg id of the particle that made it.

Trees are found by branch signature rather than by name, since the Si trees
prefix every branch with the detector name and the TRT tree does not. PyROOT is
used because the ATLAS release ships no uproot.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

QUIRK_PDGID = 10000100


@dataclass
class HitSet:
    """Hits from one subdetector in one event, in mm."""

    system: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    pdg: Optional[np.ndarray] = None
    # The TRT tree carries no pdgId, only barcode, so quirk hits there are
    # identified by matching against the quirk barcodes in the Truth tree.
    barcode: Optional[np.ndarray] = None
    quirk_barcodes: Optional[frozenset] = None

    def __len__(self) -> int:
        return len(self.x)

    @property
    def r(self) -> np.ndarray:
        return np.hypot(self.x, self.y)

    @property
    def phi(self) -> np.ndarray:
        return np.arctan2(self.y, self.x)

    def select(self, mask: np.ndarray) -> "HitSet":
        return HitSet(self.system, self.x[mask], self.y[mask], self.z[mask],
                      None if self.pdg is None else self.pdg[mask],
                      None if self.barcode is None else self.barcode[mask],
                      self.quirk_barcodes)

    def quirk_mask(self) -> np.ndarray:
        """Which hits came from a quirk, by pdg id or failing that by barcode."""
        if self.pdg is not None:
            return np.abs(self.pdg) == QUIRK_PDGID
        if self.barcode is not None and self.quirk_barcodes:
            return np.isin(self.barcode, np.fromiter(self.quirk_barcodes, np.int64))
        return np.zeros(len(self), dtype=bool)

    def quirks(self) -> "HitSet":
        return self.select(self.quirk_mask())

    def others(self) -> "HitSet":
        return self.select(~self.quirk_mask())


@dataclass
class TruthParticle:
    pdg: int
    status: int
    px: float   # MeV
    py: float
    pz: float

    @property
    def p(self) -> float:
        return float(np.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2))


@dataclass
class Event:
    index: int
    hits: Dict[str, HitSet] = field(default_factory=dict)
    truth: List[TruthParticle] = field(default_factory=list)
    # Signal-process vertex, mm. The beamspot is applied at simulation time and
    # its z spread is tens of mm, so this is not the origin.
    vertex: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quirk_barcodes: frozenset = frozenset()

    def all_hits(self) -> HitSet:
        sets = [h for h in self.hits.values() if len(h)]
        if not sets:
            return HitSet("all", *(np.empty(0) for _ in range(3)))
        return HitSet(
            "all",
            np.concatenate([h.x for h in sets]),
            np.concatenate([h.y for h in sets]),
            np.concatenate([h.z for h in sets]),
            np.concatenate([
                h.pdg if h.pdg is not None else np.zeros(len(h)) for h in sets
            ]),
        )

    def quirk_truth(self) -> List[TruthParticle]:
        return [t for t in self.truth
                if abs(t.pdg) == QUIRK_PDGID and t.status == 1]

    def n_quirk_hits(self) -> int:
        return sum(len(h.quirks()) for h in self.hits.values())


_SI_SYSTEMS = {"Pixel": "Pixel", "SCT": "SCT"}


def _read_vertex(tree) -> Tuple[float, float, float]:
    """Quirk production vertex, mm.

    vtx_* is filled per vertex, not per particle, so it cannot be indexed by
    particle. vtx_proc_* is the signal process vertex and is a single entry;
    vtx_prim_* are the generator vertices, which carry the same position.
    """
    for prefix in ("vtx_proc", "vtx_prim"):
        try:
            xs = getattr(tree, f"{prefix}_x")
            ys = getattr(tree, f"{prefix}_y")
            zs = getattr(tree, f"{prefix}_z")
        except AttributeError:
            continue
        if len(xs):
            return (float(xs[0]), float(ys[0]), float(zs[0]))
    return (0.0, 0.0, 0.0)


def _walk(directory, prefix=""):
    """Yield (path, TTree) for every tree in a ROOT directory, recursively."""
    import ROOT
    for key in directory.GetListOfKeys():
        obj = key.ReadObj()
        path = f"{prefix}/{key.GetName()}" if prefix else key.GetName()
        if isinstance(obj, ROOT.TTree):
            yield path, obj
        elif isinstance(obj, ROOT.TDirectory):
            yield from _walk(obj, path)


class HitNtuple:
    """Lazily-read view of one HitAnalysis output file."""

    def __init__(self, path: str):
        import ROOT
        ROOT.gROOT.SetBatch(True)
        self._file = ROOT.TFile.Open(path)
        if not self._file or self._file.IsZombie():
            raise IOError(f"cannot open {path}")
        self._trees = dict(_walk(self._file))
        self._hit_trees = {}
        self._truth_tree = None
        self._classify()

    def _classify(self):
        for path, tree in self._trees.items():
            branches = {b.GetName() for b in tree.GetListOfBranches()}
            if {"truth_px", "truth_py", "truth_pz"} <= branches:
                self._truth_tree = (path, tree)
                continue
            for system, det in _SI_SYSTEMS.items():
                if {f"{det}_x", f"{det}_y", f"{det}_z"} <= branches:
                    self._hit_trees[system] = (tree, det)
                    break
            else:
                if {"x", "y", "z", "r"} <= branches and "TRT" in path:
                    self._hit_trees["TRT"] = (tree, None)

    @property
    def systems(self) -> List[str]:
        return list(self._hit_trees)

    def __len__(self) -> int:
        trees = [t for t, _ in self._hit_trees.values()]
        if self._truth_tree:
            trees.append(self._truth_tree[1])
        return min((t.GetEntries() for t in trees), default=0)

    def event(self, index: int) -> Event:
        evt = Event(index=index)
        quirk_barcodes = frozenset()
        if self._truth_tree and index < self._truth_tree[1].GetEntries():
            tree = self._truth_tree[1]
            tree.GetEntry(index)
            pdg = list(tree.pdg_id)
            status = list(tree.status)
            pxs, pys, pzs = list(tree.truth_px), list(tree.truth_py), list(tree.truth_pz)
            for i in range(min(len(pdg), len(pxs))):
                evt.truth.append(TruthParticle(
                    pdg=int(pdg[i]), status=int(status[i]),
                    px=float(pxs[i]), py=float(pys[i]), pz=float(pzs[i]),
                ))
            if hasattr(tree, "barcode"):
                bcs = list(tree.barcode)
                # Any status: the generator records the quirk several times
                # (status 23 hard process, 44 shower copies, 1 final state),
                # each with its own barcode. Hits carry the status-1 barcode,
                # but matching the whole set is a harmless superset.
                quirk_barcodes = frozenset(
                    int(bcs[i]) for i in range(min(len(pdg), len(bcs)))
                    if abs(int(pdg[i])) == QUIRK_PDGID)
            evt.vertex = _read_vertex(tree)
            evt.quirk_barcodes = quirk_barcodes

        for system, (tree, det) in self._hit_trees.items():
            if index >= tree.GetEntries():
                continue
            tree.GetEntry(index)
            px, py, pz = (f"{det}_x", f"{det}_y", f"{det}_z") if det else ("x", "y", "z")
            pdg_branch = f"{det}_pdgId" if det else None
            bc_branch = f"{det}_barcode" if det else "barcode"
            pdg = None
            if pdg_branch and hasattr(tree, pdg_branch):
                pdg = np.fromiter(getattr(tree, pdg_branch), dtype=np.int64)
            bc = None
            if hasattr(tree, bc_branch):
                bc = np.fromiter(getattr(tree, bc_branch), dtype=np.int64)
            evt.hits[system] = HitSet(
                system,
                np.fromiter(getattr(tree, px), dtype=np.float64),
                np.fromiter(getattr(tree, py), dtype=np.float64),
                np.fromiter(getattr(tree, pz), dtype=np.float64),
                pdg, bc, quirk_barcodes,
            )
        return evt

    def events(self):
        for i in range(len(self)):
            yield self.event(i)

    def close(self):
        self._file.Close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
