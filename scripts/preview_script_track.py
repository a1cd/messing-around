"""Save top-down PNG previews of scripted tracks (the named examples by default).

Usage:
    python scripts/preview_script_track.py OUT_DIR [--seed 0] [--example NAME ...] [--dsl "S100 L3 S100 CLOSE"]

Each PNG shows the terrain with the road, the segment labels (caution segments
in orange), and an elevation / curvature profile along the route.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drivesim.track_script import EXAMPLES, TrackScript, build_world_from_script  # noqa: E402


def plot(name, script, seed, out_path):
    terrain, track = build_world_from_script(script, seed=seed)
    h = terrain.size / 2
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.3, 1]})
    im = ax.imshow(terrain.heights, origin="lower", extent=(-h, h, -h, h), cmap="terrain")
    fig.colorbar(im, ax=ax, label="height (m)", shrink=0.75)
    c = track.centerline
    ax.plot(track.left_edge[:, 0], track.left_edge[:, 1], "k-", lw=0.6)
    ax.plot(track.right_edge[:, 0], track.right_edge[:, 1], "k-", lw=0.6)
    ax.plot(c[:, 0], c[:, 1], "r-", lw=1.0, label="centreline")
    ax.plot(*c[0, :2], "go", ms=8, label="start")
    if not track.closed:
        ax.plot(*track.goal[:2], "bs", ms=8, label="end")
    for row in track.segments:
        i0 = int(np.searchsorted(track.s, row.s0))
        i1 = int(np.searchsorted(track.s, row.s1))
        if row.kind == "straight" or i1 - i0 < 2:
            continue
        mid = min((i0 + i1) // 2, len(c) - 1)
        color = "darkorange" if row.caution else "navy"
        ax.annotate(row.text, c[mid, :2], fontsize=6, color=color, ha="center", va="bottom")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    kind = "closed" if track.closed else "point-to-point"
    done = " (auto-completed)" if track.closure.get("completed") else ""
    ax.set_title(f"{name}: {kind}{done}, length {track.length:.0f} m")
    ax.legend(loc="upper right", fontsize=8)

    ax2.plot(track.s, c[:, 2], label="elevation (m)")
    ax2.set_xlabel("arc length s (m)")
    ax2.set_ylabel("z (m)")
    ax3 = ax2.twinx()
    ax3.plot(track.s, track.curvature, color="tab:orange", lw=0.7, label="curvature (1/m)")
    ax3.set_ylabel("curvature (1/m)")
    ax2.set_title("track profile")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}: length {track.length:.1f} m, closed={track.closed}, "
          f"segments {len(track.segments)}, max |curvature| {np.abs(track.curvature).max():.4f} 1/m")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="directory for the PNG files")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--example", nargs="*", default=None, help="named examples (default: all)")
    ap.add_argument("--dsl", default=None, help="a DSL script to preview instead")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.dsl:
        plot("custom", TrackScript.from_dsl(args.dsl), args.seed,
             os.path.join(args.out_dir, "custom_script_track.png"))
        return 0
    for name in args.example or list(EXAMPLES):
        plot(name, TrackScript.from_dsl(EXAMPLES[name]), args.seed,
             os.path.join(args.out_dir, f"script_{name}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
