"""Nominal ATLAS Run-3 inner detector layout. Design values, drawing only."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class BarrelLayer:
    name: str
    radius: float     # mm
    half_z: float     # mm
    system: str


@dataclass(frozen=True)
class EndcapDisk:
    name: str
    z: float          # mm
    r_inner: float    # mm
    r_outer: float    # mm
    system: str


@dataclass(frozen=True)
class InnerDetector:
    barrel: List[BarrelLayer] = field(default_factory=list)
    endcap: List[EndcapDisk] = field(default_factory=list)
    r_max: float = 1100.0
    z_max: float = 3000.0

    def barrel_by_system(self, system: str) -> List[BarrelLayer]:
        return [b for b in self.barrel if b.system == system]

    def systems(self) -> List[str]:
        seen = []
        for layer in self.barrel:
            if layer.system not in seen:
                seen.append(layer.system)
        return seen

    def subset(self, systems) -> "InnerDetector":
        """A copy holding only the named subdetectors."""
        keep = set(systems)
        barrel = [b for b in self.barrel if b.system in keep]
        endcap = [d for d in self.endcap if d.system in keep]
        r_max = max([b.radius for b in barrel], default=self.r_max)
        z_max = max([abs(d.z) for d in endcap]
                    + [b.half_z for b in barrel], default=self.z_max)
        return InnerDetector(barrel=barrel, endcap=endcap,
                             r_max=r_max, z_max=z_max)

    def radial_span(self, system: str) -> Tuple[float, float]:
        rs = [b.radius for b in self.barrel_by_system(system)]
        return (min(rs), max(rs)) if rs else (0.0, 0.0)


def _run3() -> InnerDetector:
    barrel = [
        BarrelLayer("IBL",     33.25, 331.0, "Pixel"),
        BarrelLayer("B-Layer", 50.5,  400.5, "Pixel"),
        BarrelLayer("Pixel-1", 88.5,  400.5, "Pixel"),
        BarrelLayer("Pixel-2", 122.5, 400.5, "Pixel"),
        BarrelLayer("SCT-0",   299.0, 746.0, "SCT"),
        BarrelLayer("SCT-1",   371.0, 746.0, "SCT"),
        BarrelLayer("SCT-2",   443.0, 746.0, "SCT"),
        BarrelLayer("SCT-3",   514.0, 746.0, "SCT"),
    ]
    # The TRT barrel is a continuous straw volume; these are representative
    # radii across its extent, not physical layers.
    barrel += [
        BarrelLayer(f"TRT-{i}", r, 712.0, "TRT")
        for i, r in enumerate((563.0, 700.0, 840.0, 980.0, 1066.0))
    ]

    endcap = []
    for i, z in enumerate((495.0, 580.0, 650.0)):
        for sign in (+1, -1):
            endcap.append(EndcapDisk(f"Pixel-disk-{i}", sign * z, 88.8, 149.6, "Pixel"))
    for i, z in enumerate((853.8, 934.0, 1091.5, 1299.9, 1399.7, 1771.4, 2115.2, 2505.0)):
        for sign in (+1, -1):
            endcap.append(EndcapDisk(f"SCT-disk-{i}", sign * z, 275.0, 560.0, "SCT"))
    for i, z in enumerate((848.0, 1350.0, 1850.0, 2350.0, 2710.0)):
        for sign in (+1, -1):
            endcap.append(EndcapDisk(f"TRT-disk-{i}", sign * z, 617.0, 1106.0, "TRT"))

    return InnerDetector(barrel=barrel, endcap=endcap, r_max=1100.0, z_max=3000.0)


ID_RUN3 = _run3()

SYSTEM_COLOURS = {
    "Pixel": "#4C72B0",
    "SCT": "#55A868",
    "TRT": "#C44E52",
}
