"""Fit the fill-intensity (A, k) and bounded-vol c(t) calibrations on a season.

    python scripts/02_calibrate.py --sport nba --season 2024-25

Writes data/calib/{sport}_{season}_intensity.json and _vol.json. Calibrate on
the TRAIN season only; the test season must stay out-of-sample.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi_mm.calib.intensity import fit_intensity, save_intensity
from kalshi_mm.calib.vol import fit_vol, save_vol
from kalshi_mm.data.build import iter_games


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sport", default="nba")
    ap.add_argument("--season", default="2024-25")
    ap.add_argument("--base-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/calib")
    ap.add_argument("--pregame-hours", type=float, default=12.0)
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    games = []
    for g in iter_games(args.base_dir, args.sport, args.season,
                        pregame_hours=args.pregame_hours):
        games.append(g)
        if args.max_games and len(games) >= args.max_games:
            break
    logging.info("loaded %d games", len(games))
    if not games:
        sys.exit("no games found - run 01_download.py first")

    out = Path(args.out_dir)
    intensity = fit_intensity(games)
    save_intensity(intensity, out / f"{args.sport}_{args.season}_intensity.json")
    print("\nFill intensity lambda(delta)=A*exp(-k*delta) by hours-to-tip:")
    print(intensity.to_string(index=False))

    vol = fit_vol(games)
    save_vol(vol, out / f"{args.sport}_{args.season}_vol.json")
    print("\nBounded vol c(t) (cents/sqrt-min at mid=50):")
    print(vol.to_string(index=False))


if __name__ == "__main__":
    main()
