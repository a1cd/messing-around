"""Simulate a few vehicle scenarios on flat ground and plot speed, rpm, gear and slip.

Usage:
    python scripts/vehicle_demo.py [--out path/to/plot.png]

Scenarios (sports car unless noted):
  1. Full-throttle launch from rest, automatic transmission.
  2. Full braking from 100 km/h (no ABS, so the wheels lock).
  3. Hatchback handbrake slide: 20 m/s, handbrake pulse with steering, throttle.
"""

import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from drivesim.vehicle import Controls, FlatTerrain, Vehicle, VehicleParams  # noqa: E402

DT = 1.0 / 30.0


def record(car, terrain, controls_fn, seconds, stop=None):
    rows = {k: [] for k in ("t", "v", "rpm", "gear", "kr", "kf", "beta", "yaw", "clutch")}
    n = int(seconds / DT)
    for i in range(n):
        car.step(controls_fn(car, i), DT, terrain)
        rows["t"].append(car.time)
        rows["v"].append(car.speed_kmh)
        rows["rpm"].append(car.engine_rpm)
        rows["gear"].append(car.gear)
        k = car.wheel_slip_ratios
        rows["kf"].append(float(np.mean(k[:2])))
        rows["kr"].append(float(np.mean(k[2:])))
        rows["beta"].append(math.degrees(car.body_slip_angle))  # unwrapped below
        rows["yaw"].append(math.degrees(car.yaw_rate))
        rows["clutch"].append(car.clutch_engagement)
        if stop is not None and stop(car):
            break
    out = {k: np.array(v) for k, v in rows.items()}
    out["beta"] = np.degrees(np.unwrap(np.radians(out["beta"])))
    return out


def run_launch():
    t = FlatTerrain()
    car = Vehicle(VehicleParams.sports_car(), "auto")
    car.reset([0.0, 0.0, 0.0], 0.0, t)
    data = record(car, t, lambda c, i: Controls(throttle=1.0), 16.0,
                  stop=lambda c: c.speed_kmh > 160)
    return "Launch 0-160 km/h (sports car, auto)", data


def run_braking():
    t = FlatTerrain()
    car = Vehicle(VehicleParams.sports_car(), "auto")
    car.reset([0.0, 0.0, 0.0], 0.0, t)
    car.vel = [100.0 / 3.6, 0.0, 0.0]
    car.w = [100.0 / 3.6 / car.R] * 4
    car.pt.gear = 6
    car.pt.engage = 0.0
    data = record(car, t, lambda c, i: Controls(brake=1.0), 8.0, stop=lambda c: c.speed < 0.05)
    return "Brake 100 -> 0 km/h (sports car, no ABS)", data


def run_drift():
    t = FlatTerrain()
    car = Vehicle(VehicleParams.hatchback(), "auto")
    car.reset([0.0, 0.0, 0.0], 0.0, t)
    car.vel = [20.0, 0.0, 0.0]
    car.w = [20.0 / car.R] * 4
    car.pt.gear = 2
    car.pt.pending_gear = 2
    car.pt.engage = 1.0
    car.pt.engine_w = car.p.gear_ratios[1] * car.p.final_drive * 20.0 / car.R

    def controls(c, i):
        t_s = i * DT
        hb = 1.0 if 0.5 < t_s < 1.0 else 0.0
        return Controls(steer=0.35 if t_s < 2.5 else 0.0, throttle=0.6, handbrake=hb)

    data = record(car, t, controls, 4.0)
    return "Hatchback handbrake slide (FWD, 20 m/s)", data


def make_figure(scenarios, out_path):
    fig, axes = plt.subplots(4, len(scenarios), figsize=(5.2 * len(scenarios), 11), sharex="col")
    for col, (title, d) in enumerate(scenarios):
        ax = axes[0, col]
        ax.plot(d["t"], d["v"], color="tab:blue")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("speed [km/h]")
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        ax.plot(d["t"], d["rpm"], color="tab:red", label="engine rpm")
        ax.set_ylabel("engine rpm")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.step(d["t"], d["gear"], where="post", color="tab:gray", alpha=0.7, label="gear")
        ax2.set_ylabel("gear")
        ax2.set_yticks([-1, 0, 1, 2, 3, 4, 5, 6])

        ax = axes[2, col]
        ax.plot(d["t"], d["kf"], color="tab:green", label="front slip ratio")
        ax.plot(d["t"], d["kr"], color="tab:purple", label="rear slip ratio")
        ax.set_ylabel("mean wheel slip ratio")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

        ax = axes[3, col]
        ax.plot(d["t"], d["beta"], color="tab:orange", label="body slip beta [deg]")
        ax.plot(d["t"], d["yaw"], color="tab:cyan", label="yaw rate [deg/s]")
        ax.set_ylabel("deg, deg/s")
        ax.set_xlabel("time [s]")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("drivesim vehicle dynamics: scenario traces (dt = 1/30 s, 4 substeps)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(ROOT, "scripts", "vehicle_demo.png"))
    args = parser.parse_args()
    scenarios = [run_launch(), run_braking(), run_drift()]
    for title, d in scenarios:
        print(f"{title}: {d['t'][-1]:.2f} s simulated, max speed {d['v'].max():.1f} km/h, "
              f"gears {sorted(set(d['gear'].tolist()))}")
    make_figure(scenarios, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
