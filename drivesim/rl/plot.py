"""Learning-curve plot from a run directory's ``metrics.csv`` (matplotlib, Agg backend)."""
from __future__ import annotations

import csv
import os


def read_metrics(path: str) -> dict:
    cols: dict = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    cols.setdefault(k, []).append(float("nan"))
    return cols


def plot_run(run_dir: str, out_path: str | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    m = read_metrics(os.path.join(run_dir, "metrics.csv"))
    x = np.array(m["total_steps"]) / 1e6

    def smooth(y, k=5):
        y = np.asarray(y, float)
        if len(y) < k:
            return y
        pad = np.concatenate([np.full(k - 1, y[0]), y])
        return np.convolve(pad, np.ones(k) / k, mode="valid")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(f"PPO learning curves: {os.path.basename(os.path.normpath(run_dir))}")
    a = ax[0, 0]
    a.plot(x, m["success_rate"], alpha=0.3, color="C0")
    a.plot(x, smooth(m["success_rate"]), color="C0", label="success (last 200 eps)")
    a.set_ylim(-0.02, 1.02)
    a.set_title("success rate")
    a.set_xlabel("env steps (M)")
    a.legend(loc="lower right", fontsize=8)
    a = ax[0, 1]
    a.plot(x, m["ep_return"], alpha=0.3, color="C1")
    a.plot(x, smooth(m["ep_return"]), color="C1", label="episode return (raw)")
    a.axhline(0, color="grey", lw=0.5)
    a.set_title("mean episode return (raw reward)")
    a.set_xlabel("env steps (M)")
    a.legend(fontsize=8)
    a = ax[0, 2]
    a.plot(x, m["ep_length"], color="C2")
    a.set_title("mean episode length (steps)")
    a.set_xlabel("env steps (M)")
    a = ax[1, 0]
    a.plot(x, m["pg_loss"], label="policy", alpha=0.7)
    a.plot(x, m["vf_loss"], label="value", alpha=0.7)
    a.set_title("losses")
    a.set_xlabel("env steps (M)")
    a.legend(fontsize=8)
    a = ax[1, 1]
    a.plot(x, m["approx_kl"], label="approx KL", color="C3")
    a.plot(x, m["clipfrac"], label="clip fraction", color="C4")
    a.set_title("policy change per update")
    a.set_xlabel("env steps (M)")
    a.legend(fontsize=8)
    a = ax[1, 2]
    a.plot(x, m["explained_var"], color="C5", label="explained variance")
    a.plot(x, m["entropy"], color="C6", label="entropy")
    a.set_title("value fit and entropy")
    a.set_xlabel("env steps (M)")
    a.legend(fontsize=8)
    if "difficulty" in m:
        for axis in ax.flat:
            axis2 = axis.twinx()
            axis2.step(x, m["difficulty"], where="post", color="grey", alpha=0.25, lw=1)
            axis2.set_ylim(0, 1.05)
            axis2.tick_params(labelsize=7)
    fig.tight_layout()
    out_path = out_path or os.path.join(run_dir, "learning_curves.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
