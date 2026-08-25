#!/usr/bin/env python
"""HITS pool.root -> HitAnalysis ntuple. Runs inside athena."""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hits", nargs="+", help="input HITS pool.root file(s)")
    ap.add_argument("-o", "--output", default="hits.ntuple.root")
    ap.add_argument("-n", "--max-events", type=int, default=-1)
    ap.add_argument("--no-trt", action="store_true", help="skip the TRT trees")
    args = ap.parse_args()

    from AthenaConfiguration.AllConfigFlags import initConfigFlags
    from AthenaConfiguration.MainServicesConfig import MainServicesCfg
    from AthenaPoolCnvSvc.PoolReadConfig import PoolReadCfg

    flags = initConfigFlags()
    flags.Input.Files = args.hits
    flags.Output.HISTFileName = args.output
    flags.Exec.MaxEvents = args.max_events
    flags.Common.isOnline = False
    flags.Concurrency.NumThreads = 0
    flags.lock()

    cfg = MainServicesCfg(flags)
    cfg.merge(PoolReadCfg(flags))

    from HitAnalysis.HitAnalysisConfig import (PixelHitAnalysisCfg,
                                               SCTHitAnalysisCfg,
                                               TRTHitAnalysisCfg,
                                               TruthHitAnalysisCfg)
    # ExtraTruthBranches adds the per-hit pdgId used to select quirk hits.
    cfg.merge(PixelHitAnalysisCfg(flags, ExtraTruthBranches=True))
    cfg.merge(SCTHitAnalysisCfg(flags, ExtraTruthBranches=True))
    if not args.no_trt:
        cfg.merge(TRTHitAnalysisCfg(flags))
    cfg.merge(TruthHitAnalysisCfg(flags))

    return cfg.run().isFailure()


if __name__ == "__main__":
    sys.exit(main())
