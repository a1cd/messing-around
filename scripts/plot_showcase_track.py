"""Render a top-down layout and elevation profile of the showcase circuit to docs/showcase_track.png.

Usage: python scripts/plot_showcase_track.py [--out docs/showcase_track.png]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from drivesim.track_script import build_showcase_world  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "showcase_track.png"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    terrain, track = build_showcase_world(args.seed)
    P = track.centerline
    xy = P[:, :2]
    z = P[:, 2]
    s = track.s
    # unit normals from the centerline heading
    d = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    hw = track.half_width
    left, right = xy + nrm * hw, xy - nrm * hw

    fig = plt.figure(figsize=(11, 7.2), dpi=110)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0], hspace=0.32)
    ax = fig.add_subplot(gs[0])

    # terrain shading (light)
    n = 160
    lo, hi = terrain.origin, terrain.origin + terrain.size
    gx = np.linspace(xy[:, 0].min() - 40, xy[:, 0].max() + 40, n)
    gy = np.linspace(xy[:, 1].min() - 40, xy[:, 1].max() + 40, n)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.vectorize(lambda a, b: terrain.height_at(a, b))(GX, GY)
    ax.contourf(GX, GY, GZ, levels=24, cmap="Greys", alpha=0.18)
    ax.contour(GX, GY, GZ, levels=12, colors="0.6", linewidths=0.4, alpha=0.6)

    # track surface, coloured by elevation
    zc = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    ring = np.concatenate([left, right[::-1]])  # draw surface as a filled band
    ax.fill(ring[:, 0], ring[:, 1], color="0.82", zorder=2, lw=0)
    segs = np.stack([np.stack([xy[:-1], xy[1:]], axis=1)], axis=0)[0]
    lc = LineCollection(segs, cmap="viridis", linewidths=5, zorder=3)
    lc.set_array(zc[:-1])
    ax.add_collection(lc)
    ax.plot(left[:, 0], left[:, 1], color="0.2", lw=0.8, zorder=4)
    ax.plot(right[:, 0], right[:, 1], color="0.2", lw=0.8, zorder=4)
    ax.plot(xy[:, 0], xy[:, 1], color="k", lw=0.4, ls="--", zorder=4, alpha=0.6)

    # start / finish marker and driving direction arrow
    x0, y0 = xy[0]
    ax.plot([left[0, 0], right[0, 0]], [left[0, 1], right[0, 1]], color="red", lw=3, zorder=5)
    dx, dy = d[0]
    ax.annotate("", xy=(x0 + dx * 120, y0 + dy * 120), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.2), zorder=6)
    ax.text(x0 + dx * 60 - dy * 40, y0 + dy * 60 + dx * 40, "START / FINISH", color="red",
            fontsize=9, weight="bold", zorder=6)

    # direction along the lap, every 500 m
    for k in range(500, int(track.length), 500):
        i = int(k / track.spacing)
        ax.plot(xy[i, 0], xy[i, 1], "o", ms=3, color="k", zorder=6)
        ax.text(xy[i, 0], xy[i, 1], f" {k} m", fontsize=7, zorder=7)

    ax.set_aspect("equal")
    ax.set_title(f"Showcase circuit: {track.length / 1000:.2f} km closed loop, "
                 f"elevation range {z.max() - z.min():.1f} m (colour = elevation)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.autoscale_view()
    cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("relative elevation (0 = lowest point)")

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(s / 1000.0, z, color="C0", lw=1.4)
    ax2.fill_between(s / 1000.0, z.min(), z, color="C0", alpha=0.12)
    ax2.set_xlabel("distance along lap (km)")
    ax2.set_ylabel("elevation (m)")
    ax2.set_xlim(0, track.length / 1000.0)
    ax2.grid(alpha=0.3)
    ax2.set_title("Elevation profile (start/finish at 0 km)", fontsize=10)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} length={track.length:.1f} m n={track.n} "
          f"elev_range={z.max() - z.min():.1f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
