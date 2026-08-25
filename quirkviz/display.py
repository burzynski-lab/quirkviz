"""Matplotlib event displays: detector, hits, quirk trajectories."""

from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .geometry import ID_RUN3, InnerDetector, SYSTEM_COLOURS

QUIRK_COLOURS = ("#E24A33", "#348ABD")


def quirk_extent(event, tracks, pad=1.25, floor=60.0):
    """(r, z) extent of the quirk activity in mm, for auto-zooming."""
    rs, zs = [], []
    for hits in event.hits.values():
        q = hits.quirks()
        if len(q):
            rs.append(float(np.max(q.r)))
            zs.append(float(np.max(np.abs(q.z))))
    for pts in (tracks or {}).values():
        if pts is None or len(pts) == 0:
            continue
        rs.append(float(np.max(np.hypot(pts[:, 0], pts[:, 1]))))
        zs.append(float(np.max(np.abs(pts[:, 2]))))
    r = max(rs) * pad if rs else floor
    z = max(zs) * pad if zs else floor
    return max(r, floor), max(z, floor)


def _draw_id_rphi(ax, det: InnerDetector, r_max: float):
    for layer in det.barrel:
        if layer.radius > r_max:
            continue
        ax.add_patch(Circle((0, 0), layer.radius, fill=False, lw=0.8,
                            ec=SYSTEM_COLOURS.get(layer.system, "grey"),
                            alpha=0.55, zorder=1))
    return [plt.Line2D([], [], color=SYSTEM_COLOURS[s], lw=1.2, label=s)
            for s in det.systems() if s in SYSTEM_COLOURS]


def _draw_id_rz(ax, det: InnerDetector, r_max: float, z_max: float):
    for layer in det.barrel:
        if layer.radius > r_max:
            continue
        ax.plot([-layer.half_z, layer.half_z], [layer.radius] * 2,
                color=SYSTEM_COLOURS.get(layer.system, "grey"),
                lw=0.8, alpha=0.55, zorder=1)
    for disk in det.endcap:
        if abs(disk.z) > z_max or disk.r_inner > r_max:
            continue
        ax.plot([disk.z] * 2, [disk.r_inner, min(disk.r_outer, r_max)],
                color=SYSTEM_COLOURS.get(disk.system, "grey"),
                lw=0.8, alpha=0.55, zorder=1)


def _scatter_hits(ax, event, coords, r_max: float, z_max: Optional[float],
                  show_other: bool) -> int:
    n_quirk = 0
    for system, hits in event.hits.items():
        if not len(hits):
            continue
        colour = SYSTEM_COLOURS.get(system, "grey")
        for subset, style in ((hits.others(), "other"), (hits.quirks(), "quirk")):
            if not len(subset) or (style == "other" and not show_other):
                continue
            if coords == "rphi":
                u, v = subset.x, subset.y
                keep = np.hypot(u, v) <= r_max
            else:
                u, v = subset.z, subset.r
                keep = (subset.r <= r_max) & (np.abs(subset.z) <= (z_max or np.inf))
            if style == "quirk":
                n_quirk += int(keep.sum())
                ax.scatter(u[keep], v[keep], s=26, marker="o", facecolors="none",
                           edgecolors=colour, linewidths=1.1, zorder=5)
            else:
                ax.scatter(u[keep], v[keep], s=3, marker=".", color="0.65",
                           alpha=0.6, linewidths=0, zorder=2)
    return n_quirk


def _draw_tracks(ax, tracks: Dict[int, np.ndarray], coords, r_max, z_max,
                 style: str, label_prefix: str):
    ls = "-" if style == "g4" else "--"
    for idx, pts in tracks.items():
        if pts is None or len(pts) == 0:
            continue
        colour = QUIRK_COLOURS[idx % len(QUIRK_COLOURS)]
        r = np.hypot(pts[:, 0], pts[:, 1])
        inside = r <= r_max
        if z_max is not None:
            inside &= np.abs(pts[:, 2]) <= z_max
        u, v = (pts[:, 0], pts[:, 1]) if coords == "rphi" else (pts[:, 2], r)
        # Break the line outside the drawn volume, so re-entrant arcs stay
        # visibly disconnected.
        seg = np.where(~inside, np.nan, 1.0)
        ax.plot(u * seg, v * seg, ls=ls, lw=1.4, color=colour, alpha=0.9,
                zorder=4, label=f"{label_prefix} quirk {idx}")


def plot_rphi(ax, event, tracks_g4=None, tracks_analytic=None,
              det: InnerDetector = ID_RUN3, r_max: float = 1100.0,
              show_other: bool = True):
    handles = _draw_id_rphi(ax, det, r_max)
    n_quirk = _scatter_hits(ax, event, "rphi", r_max, None, show_other)
    if tracks_g4:
        _draw_tracks(ax, tracks_g4, "rphi", r_max, None, "g4", "G4")
    if tracks_analytic:
        _draw_tracks(ax, tracks_analytic, "rphi", r_max, None, "analytic", "analytic")
    ax.set_aspect("equal")
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"r-$\\phi$  ({n_quirk} quirk hits)")
    return handles


def plot_rz(ax, event, tracks_g4=None, tracks_analytic=None,
            det: InnerDetector = ID_RUN3, r_max: float = 1100.0,
            z_max: float = 3000.0, show_other: bool = True):
    _draw_id_rz(ax, det, r_max, z_max)
    _scatter_hits(ax, event, "rz", r_max, z_max, show_other)
    if tracks_g4:
        _draw_tracks(ax, tracks_g4, "rz", r_max, z_max, "g4", "G4")
    if tracks_analytic:
        _draw_tracks(ax, tracks_analytic, "rz", r_max, z_max, "analytic", "analytic")
    ax.set_xlim(-z_max, z_max)
    ax.set_ylim(0, r_max)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title("r-z")


def plot_event(event, tracks_g4=None, tracks_analytic=None,
               det: InnerDetector = ID_RUN3, r_max: float = 1100.0,
               z_max: float = 3000.0, show_other: bool = True,
               subtitle: str = "", figsize=(15, 7)):
    """Side-by-side r-phi and r-z display of one event."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    det_handles = plot_rphi(ax1, event, tracks_g4, tracks_analytic, det,
                            r_max, show_other)
    plot_rz(ax2, event, tracks_g4, tracks_analytic, det, r_max, z_max, show_other)

    track_handles, _ = ax1.get_legend_handles_labels()
    hit_handles = [
        plt.Line2D([], [], ls="none", marker="o", mfc="none", mec="0.2",
                   label="quirk hit"),
        plt.Line2D([], [], ls="none", marker=".", color="0.65", label="other hit"),
    ]
    fig.legend(handles=det_handles + hit_handles + track_handles,
               loc="lower center", ncol=8, frameon=False, fontsize=9)
    fig.suptitle(f"Event {event.index}" + (f"   {subtitle}" if subtitle else ""))
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig
