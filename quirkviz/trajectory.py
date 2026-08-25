"""Quirk trajectories from truth momenta, or from a Geant4 stepping log.

The analytic model assumes a constant string force (V = Lambda^2 r), which is
exactly integrable in the QQbar rest frame; the lab path is that oscillation
boosted along the pair momentum. It includes no magnetic field and no dE/dx,
both of which Geant4 applies.

parse_g4_debug_log reads the stepped positions Geant4 actually used, from an
athena log written with the Quirks DebugSteppingAction enabled.
"""

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HBARC_GEV_MM = 1.9732705e-13   # GeV^-1 -> mm
MEV_TO_GEV = 1.0e-3


def _boost_to_rest(p, P):
    Px, Py, Pz, PE = P
    bx, by, bz = -Px / PE, -Py / PE, -Pz / PE
    b2 = bx * bx + by * by + bz * bz
    if b2 <= 0:
        return p
    g = 1.0 / math.sqrt(1.0 - b2)
    px, py, pz, E = p
    bp = bx * px + by * py + bz * pz
    k = (g - 1.0) / b2
    return (px + k * bp * bx + g * bx * E,
            py + k * bp * by + g * by * E,
            pz + k * bp * bz + g * bz * E,
            g * (E + bp))


def _boost_from_rest(X, P):
    Px, Py, Pz, PE = P
    bx, by, bz = Px / PE, Py / PE, Pz / PE
    b2 = bx * bx + by * by + bz * bz
    if b2 <= 0:
        return X
    g = 1.0 / math.sqrt(1.0 - b2)
    x, y, z, t = X
    bx_ = bx * x + by * y + bz * z
    k = (g - 1.0) / b2
    return (x + k * bx_ * bx + g * bx * t,
            y + k * bx_ * by + g * by * t,
            z + k * bx_ * bz + g * bz * t,
            g * (t + bx_))


def _pair_frame(truth, mass_gev: float):
    """(P, p0, axis) for the pair: total 4-momentum, CM momentum, unit axis."""
    q1, q2 = truth
    p1 = tuple(v * MEV_TO_GEV for v in (q1.px, q1.py, q1.pz))
    p2 = tuple(v * MEV_TO_GEV for v in (q2.px, q2.py, q2.pz))
    e1 = math.sqrt(sum(v * v for v in p1) + mass_gev ** 2)
    e2 = math.sqrt(sum(v * v for v in p2) + mass_gev ** 2)
    P = (p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2], e1 + e2)
    q = _boost_to_rest((*p1, e1), P)
    p0 = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2)
    axis = (q[0] / p0, q[1] / p0, q[2] / p0) if p0 > 0 else (0.0, 0.0, 1.0)
    return P, p0, axis


def quirk_trajectories(truth, lambda_ev: float, mass_gev: Optional[float] = None,
                       n_period: float = 2.0, n_samp: int = 2000,
                       vertex_mm: Sequence[float] = (0.0, 0.0, 0.0)
                       ) -> List[np.ndarray]:
    """Lab-frame paths of both quirks as (N, 3) arrays in mm.

    truth is the pair of final-state TruthParticle objects (momenta in MeV).
    lambda_ev is the infracolour scale in eV, related to the Quirks package
    STRINGFORCE by STRINGFORCE[MeV/mm] = Lambda[eV]^2 * 5.068e-3.
    """
    if len(truth) != 2:
        return []
    if mass_gev is None:
        raise ValueError("mass_gev is required: truth carries momentum, not mass")

    P, p0, n = _pair_frame(truth, mass_gev)
    if p0 <= 0:
        return []

    F = (lambda_ev * 1e-9) ** 2                 # GeV^2
    tau = 4.0 * p0 / F
    E0 = math.sqrt(p0 * p0 + mass_gev * mass_gev)

    paths = [[], []]
    for i in range(n_samp + 1):
        t = n_period * tau * i / n_samp
        ph = (t % tau) * F                      # 0 .. 4 p0
        if ph <= 2 * p0:
            p = p0 - ph
            x = (E0 - math.sqrt(p * p + mass_gev ** 2)) / F
        else:
            p = ph - 3 * p0
            x = -(E0 - math.sqrt(p * p + mass_gev ** 2)) / F
        # Back-to-back in the rest frame: the quirks sit at +x and -x.
        for k, sign in enumerate((+1.0, -1.0)):
            X = (sign * x * n[0], sign * x * n[1], sign * x * n[2], t)
            lx, ly, lz, _ = _boost_from_rest(X, P)
            paths[k].append((lx * HBARC_GEV_MM + vertex_mm[0],
                             ly * HBARC_GEV_MM + vertex_mm[1],
                             lz * HBARC_GEV_MM + vertex_mm[2]))
    return [np.asarray(p) for p in paths]


def trace_through_volume(truth, lambda_ev: float, mass_gev: float,
                         r_max: float = 1100.0, z_max: float = 3000.0,
                         max_periods: float = 4096.0, n_samp_per_period: int = 400,
                         vertex_mm: Sequence[float] = (0.0, 0.0, 0.0)
                         ) -> Tuple[List[np.ndarray], float]:
    """Trace until the pair leaves the volume. Returns (paths, n_periods).

    Lab distance per period scales with the pair boost, so a fixed period count
    spans very different distances event to event.
    """
    n = 2.0
    paths: List[np.ndarray] = []
    while n <= max_periods:
        paths = quirk_trajectories(truth, lambda_ev, mass_gev=mass_gev,
                                   n_period=n,
                                   n_samp=int(min(n, 64) * n_samp_per_period),
                                   vertex_mm=vertex_mm)
        if not paths:
            return [], n
        escaped = any(
            np.any((np.hypot(p[:, 0], p[:, 1]) > r_max) | (np.abs(p[:, 2]) > z_max))
            for p in paths
        )
        if escaped:
            break
        n *= 2.0
    return paths, n


def oscillation_scales(truth, lambda_ev: float, mass_gev: float) -> Dict[str, float]:
    """Half-amplitude A = T*/Lambda^2 and period in mm, beta*, m(QQbar), pT(QQbar).

    A is a rest-frame quantity; the transverse excursion is A*sin(theta*), with
    theta* the angle between the oscillation axis and the boost.
    """
    if len(truth) != 2:
        return {}
    P, p0, _ = _pair_frame(truth, mass_gev)
    F = (lambda_ev * 1e-9) ** 2
    Tstar = math.sqrt(p0 * p0 + mass_gev ** 2) - mass_gev
    M = math.sqrt(max(P[3] ** 2 - P[0] ** 2 - P[1] ** 2 - P[2] ** 2, 1e-12))
    return {
        "amplitude_mm": Tstar / F * HBARC_GEV_MM,
        "period_mm": 4.0 * p0 / F * HBARC_GEV_MM,
        "beta_star": p0 / math.sqrt(p0 * p0 + mass_gev ** 2),
        "m_QQ_GeV": M,
        "pT_pair_GeV": math.hypot(P[0], P[1]),
    }


# CLHEP prints a HepLorentzVector as "(x,y,z;t)"; accept "," as well.
_VEC = re.compile(r"^x([01]):\s*\(([-0-9.eE+]+)\s*,\s*([-0-9.eE+]+)\s*,\s*"
                  r"([-0-9.eE+]+)\s*[;,]\s*([-0-9.eE+]+)\s*\)")


def parse_g4_debug_log(path: str) -> Dict[int, np.ndarray]:
    """Stepped quirk positions from a DebugSteppingAction log, mm.

    Returns {0: (N,3), 1: (M,3)}. Each dump prints both quirks, so every block
    contributes one point to each track; repeated points are dropped.
    """
    tracks = {0: [], 1: []}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _VEC.match(line.strip())
            if not m:
                continue
            idx = int(m.group(1))
            pt = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
            if tracks[idx] and tracks[idx][-1] == pt:
                continue
            tracks[idx].append(pt)
    return {k: np.asarray(v) for k, v in tracks.items() if v}
