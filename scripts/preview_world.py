"""Save a top-down PNG of a generated terrain + track.

Usage:
    python scripts/preview_world.py OUT.png [--seed 0] [--open] [--difficulty 0.5]
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drivesim.terrain import SURFACE_NAMES  # noqa: E402
from drivesim.world import generate_world  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output PNG path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--open", action="store_true", help="point-to-point track instead of a loop")
    ap.add_argument("--difficulty", type=float, default=0.5)
    args = ap.parse_args(argv)

    terrain, track = generate_world(args.seed, closed=not args.open, difficulty=args.difficulty)
    h = terrain.size / 2
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.25, 1]})

    im = ax.imshow(terrain.heights, origin="lower", extent=(-h, h, -h, h), cmap="terrain")
    fig.colorbar(im, ax=ax, label="height (m)", shrink=0.8)
    # Surface overlay: mud/ice only, so the patches stand out.
    ids = terrain.surface_ids
    for sid, color in ((3, "saddlebrown"), (4, "lightblue")):
        mask = np.ma.masked_where(ids != sid, np.ones(ids.shape))
        ax.imshow(mask, origin="lower", extent=(-h, h, -h, h), cmap=matplotlib.colors.ListedColormap([color]), alpha=0.5)
    c = track.centerline
    ax.plot(track.left_edge[:, 0], track.left_edge[:, 1], "k-", lw=0.8)
    ax.plot(track.right_edge[:, 0], track.right_edge[:, 1], "k-", lw=0.8)
    ax.plot(c[:, 0], c[:, 1], "r-", lw=1.2, label="centerline")
    ax.plot(*c[0, :2], "go", ms=8, label="start")
    if not track.closed:
        ax.plot(*track.goal[:2], "bs", ms=8, label="goal")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    kind = "closed" if track.closed else "point-to-point"
    ax.set_title(f"seed {args.seed}, {kind}, length {track.length:.0f} m, width {track.width:.0f} m")
    ax.legend(loc="upper right")

    s = track.s
    ax2.plot(s, c[:, 2], label="elevation (m)")
    ax2.set_xlabel("arc length s (m)")
    ax2.set_ylabel("z (m)")
    ax3 = ax2.twinx()
    ax3.plot(s, track.curvature, color="tab:orange", lw=0.8, label="curvature (1/m)")
    ax3.set_ylabel("curvature (1/m)")
    ax2.set_title("track profile")

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}: length {track.length:.1f} m, "
          f"max |curvature| {np.abs(track.curvature).max():.4f} 1/m, "
          f"surfaces {[SURFACE_NAMES[i] for i in np.unique(ids)]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
