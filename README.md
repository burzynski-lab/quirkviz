# quirkviz

Event displays for ATLAS quirk simulation: inner detector, sim hits, quirk trajectories.

## Run

```bash
source /home/jburzyns/ATLAS/Quirks/Simulation/setup_sim.sh   # athena + numpy/matplotlib/PyROOT

python -m quirkviz.dump_hits HITS.pool.root -o hits.ntuple.root
python -m quirkviz.display_cli hits.ntuple.root -o displays
```

Run both from the repository root, or put it on `PYTHONPATH`.

## dump_hits

HITS pool.root to flat ntuple. Requires athena (uses the HitAnalysis package).

| option | |
|---|---|
| `hits` | one or more input HITS files |
| `-o, --output` | output ntuple (default `hits.ntuple.root`) |
| `-n, --max-events` | events to read (default all) |
| `--no-trt` | skip the TRT trees |

## display_cli

Ntuple to figures. Needs numpy, matplotlib, PyROOT.

| option | |
|---|---|
| `ntuple` | input ntuple |
| `-o, --outdir` | output directory (default `displays`) |
| `--events` | `0-4,7` or `all` (default `0-9`) |
| `--mass` | quirk mass in GeV (default 500) |
| `--lambda-ev` | infracolour scale in eV, must match the simulation (default 400) |
| `--periods` | oscillation periods to trace, or `auto` (default) |
| `--g4-log` | athena log with DebugSteppingAction output, for the true G4 path |
| `--zoom` | frame on the quirk activity instead of the whole ID |
| `--r-max`, `--z-max` | drawn volume in mm (default 1100, 3000) |
| `--no-analytic` | omit the analytic trajectory |
| `--hide-other-hits` | draw quirk hits only |
| `--format` | `png`, `pdf`, `svg` |

## Plots

Two panels, r-phi and r-z. Open circles are quirk hits, coloured by
subdetector; grey dots are all other hits. Dashed line is the analytic
trajectory from truth momenta; solid is the Geant4 path from `--g4-log`.

The analytic model has no magnetic field and no dE/dx, so it is exact in
amplitude and period but not in azimuthal bending.

## Library

```python
from quirkviz import HitNtuple, trace_through_volume, plot_event

with HitNtuple("hits.ntuple.root") as nt:
    evt = nt.event(0)
    paths, _ = trace_through_volume(evt.quirk_truth(), 400.0, 500.0,
                                    vertex_mm=evt.vertex)
    plot_event(evt, tracks_analytic=dict(enumerate(paths))).savefig("event0.png")
```
