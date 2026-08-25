#!/usr/bin/env python
"""HitAnalysis ntuple -> event display figures."""

import argparse
import os
import sys

from .display import plot_event, quirk_extent
from .geometry import ID_RUN3
from .ntuple import HitNtuple
from .trajectory import (oscillation_scales, parse_g4_debug_log,
                         quirk_trajectories, trace_through_volume)


def parse_events(spec, n_total):
    if spec in (None, "", "all"):
        return list(range(n_total))
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < n_total]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ntuple")
    ap.add_argument("-o", "--outdir", default="displays")
    ap.add_argument("--events", default="0-9", help="e.g. 0-4,7 or all")
    ap.add_argument("--mass", type=float, default=500.0, help="quirk mass [GeV]")
    ap.add_argument("--lambda-ev", type=float, default=400.0,
                    help="infracolour scale [eV]; must match the simulation")
    ap.add_argument("--periods", default="auto",
                    help="oscillation periods to trace, or auto")
    ap.add_argument("--g4-log", help="athena log with DebugSteppingAction output")
    ap.add_argument("--r-max", type=float, default=None,
                    help="drawn radius in mm (default: fit the drawn subdetectors)")
    ap.add_argument("--z-max", type=float, default=None,
                    help="drawn |z| in mm (default: fit the drawn subdetectors)")
    ap.add_argument("--zoom", action="store_true",
                    help="frame on the quirk activity instead of the whole ID")
    ap.add_argument("--no-analytic", action="store_true")
    ap.add_argument("--hide-other-hits", action="store_true")
    ap.add_argument("--trt", action="store_true",
                    help="draw TRT hits if the ntuple has them")
    ap.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    g4_tracks = parse_g4_debug_log(args.g4_log) if args.g4_log else None
    if args.g4_log and not g4_tracks:
        print(f"warning: no quirk steps found in {args.g4_log}", file=sys.stderr)

    with HitNtuple(args.ntuple) as nt:
        n = len(nt)
        drawn = [s for s in nt.systems if args.trt or s != "TRT"]
        det = ID_RUN3.subset(drawn)
        r_max = args.r_max if args.r_max is not None else det.r_max * 1.08
        z_max = args.z_max if args.z_max is not None else det.z_max * 1.08
        print(f"{args.ntuple}: {n} events, subdetectors {nt.systems}, "
              f"drawing {drawn}")
        indices = parse_events(args.events, n)
        if not indices:
            print("no events selected", file=sys.stderr)
            return 1
        for i in indices:
            evt = nt.event(i)
            if not args.trt:
                evt.hits.pop("TRT", None)
            truth = evt.quirk_truth()
            analytic, subtitle = None, ""
            if truth and not args.no_analytic:
                if len(truth) == 2:
                    if args.periods == "auto":
                        paths, n_per = trace_through_volume(
                            truth, args.lambda_ev, args.mass,
                            r_max=r_max, z_max=z_max,
                            vertex_mm=evt.vertex)
                    else:
                        n_per = float(args.periods)
                        paths = quirk_trajectories(truth, args.lambda_ev,
                                                   mass_gev=args.mass,
                                                   n_period=n_per,
                                                   vertex_mm=evt.vertex)
                    analytic = dict(enumerate(paths))
                    s = oscillation_scales(truth, args.lambda_ev, args.mass)
                    subtitle = (f"m(QQ) = {s['m_QQ_GeV']:.0f} GeV, "
                                f"beta* = {s['beta_star']:.2f}, "
                                f"A = {s['amplitude_mm']:.1f} mm, "
                                f"L = {args.lambda_ev:g} eV, "
                                f"{n_per:g} periods, "
                                f"vtx z = {evt.vertex[2]:.0f} mm")
                else:
                    print(f"event {i}: {len(truth)} final-state quirks, expected 2",
                          file=sys.stderr)
            ev_r, ev_z = r_max, z_max
            if args.zoom:
                ev_r, ev_z = quirk_extent(evt, analytic or g4_tracks)
                ev_r, ev_z = min(ev_r, r_max), min(ev_z, z_max)
            fig = plot_event(evt, tracks_g4=g4_tracks, tracks_analytic=analytic,
                             det=det, r_max=ev_r, z_max=ev_z,
                             show_other=not args.hide_other_hits,
                             subtitle=subtitle)
            path = os.path.join(args.outdir, f"event_{i:04d}.{args.format}")
            fig.savefig(path, dpi=140)
            fig.clf()
            print(f"  event {i}: {evt.n_quirk_hits()} quirk hits, "
                  f"{len(evt.all_hits())} hits total -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
