"""Drive the simulated car yourself.

Usage:
    python scripts/drive.py [--seed N] [--difficulty D] [--open | --closed]
                            [--manual] [--car sports_car|hatchback] [--assists]
                            [--fps 30] [--no-viewer]
    python scripts/drive.py --headless-test 300 [--seed N] [--no-viewer]
    python scripts/drive.py --track showcase [--headless-test 300] [--no-viewer]

Keys (see drivesim/README_driving.md):
    A/D or Left/Right  steer        W/Up  gas        S/Down  brake, then reverse
    Space  handbrake                C  camera         R  reset to checkpoint
    T  assist on/off                P  pause          H  help overlay        Esc  quit
    Manual: LShift clutch, E shift up, Q shift down

The simulation runs at a fixed dt = 1/30 s (Vehicle substeps internally). A gamepad is picked
up automatically if one is connected.

``--headless-test N`` runs N steps with a scripted input (no window, no key handling) and
prints a summary. It is meant for CI and smoke tests; add ``--no-viewer`` to skip rendering.
"""

import argparse
import math
import os
import sys
from collections import deque

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from drivesim.human import DeviceReader, HumanInput, RawInput  # noqa: E402
from drivesim.vehicle import Vehicle, hatchback, sports_car  # noqa: E402
from drivesim.track_script import NAMED_SCRIPTS, build_world_from_script  # noqa: E402
from drivesim.world import generate_world  # noqa: E402

DT = 1.0 / 30.0
CHECKPOINT_SPACING = 20.0       # m between recorded safe points
CAMERAS = ("chase", "hood", "top")

HELP_TEXT = (
    "A/D steer   W gas   S brake/reverse   Space handbrake\n"
    "C camera   R reset   T assist   P pause   H help   Esc quit\n"
    "Manual: LShift clutch  E up  Q down"
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--difficulty", type=float, default=0.5)
    track = p.add_mutually_exclusive_group()
    track.add_argument("--closed", action="store_true", help="loop circuit (default)")
    track.add_argument("--open", action="store_true", help="point-to-point track")
    p.add_argument("--manual", action="store_true", help="manual gearbox (clutch + shift keys)")
    p.add_argument("--car", choices=("sports_car", "hatchback"), default="sports_car")
    p.add_argument("--assists", action="store_true", help="start with the stability assist on")
    p.add_argument("--fps", type=float, default=30.0, help="display frame rate cap")
    p.add_argument("--no-viewer", action="store_true", help="simulate only, no rendering or window")
    p.add_argument("--headless-test", type=int, metavar="N", default=0,
                   help="run N scripted steps without a window and print a summary")
    p.add_argument("--track", choices=sorted(NAMED_SCRIPTS), default=None,
                   help="build a named scripted track (e.g. showcase) instead of a generated one")
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=500)
    return p.parse_args(argv)


def scripted_input(i: int, n: int) -> RawInput:
    """Deterministic test driver: accelerate, weave through a turn, brake, then coast."""
    frac = i / max(n, 1)
    held = set()
    if frac < 0.5:
        held.add("w")
    elif frac < 0.7:
        held.add("w")
        held.add("a" if (i // 30) % 2 == 0 else "d")
    elif frac < 0.85:
        held.add("s")
    return RawInput(held=held)


class CheckpointTracker:
    """Remembers the last safe point on the track so R can return the car there."""

    def __init__(self, track):
        self.track = track
        self.s = 0.0
        self.hint = None

    def update(self, car) -> None:
        proj = self.track.project(car.position, hint=self.hint)
        self.hint = proj.index
        if car.is_flipped() or not proj.on_track:
            return
        if int(np.sum(car.wheel_in_contact)) < 4:
            return
        if self.track.closed:
            gap = (proj.s - self.s) % self.track.length
        else:
            gap = proj.s - self.s
        if gap >= CHECKPOINT_SPACING or gap < -CHECKPOINT_SPACING:
            self.s = proj.s

    def reset_pose(self):
        xyz, heading = self.track.point_at(self.s)
        return xyz, float(heading)


def make_car(args, terrain, track):
    params = hatchback() if args.car == "hatchback" else sports_car()
    car = Vehicle(params, transmission="manual" if args.manual else "auto")
    xyz, yaw = track.start_pose()
    car.reset(xyz, yaw, terrain)
    return car


def hud_for(car, track, human, message=None):
    proj = track.project(car.position)
    return {
        "speed_kmh": car.speed_kmh,
        "gear": car.gear,
        "rpm": car.engine_rpm,
        "throttle": human.controls.throttle,
        "brake": human.controls.brake,
        "steer": human.controls.steer,
        "progress": proj.progress,
        "message": message,
    }


def run_headless(args, terrain, track, car) -> int:
    viewer = None
    if not args.no_viewer:
        viewer = _make_viewer(args, terrain, track, headless=True)
    human = HumanInput(manual=args.manual, assists=args.assists)
    n = args.headless_test
    max_speed = 0.0
    for i in range(n):
        raw = scripted_input(i, n)
        ctrl = human.update(raw, DT, car)
        car.step(ctrl, DT, terrain)
        max_speed = max(max_speed, car.speed_kmh)
        if viewer is not None:
            viewer.render(car, hud_for(car, track, human), camera="chase")
    if viewer is not None:
        viewer.close()
    finite = bool(np.all(np.isfinite(car.position)) and math.isfinite(car.speed))
    proj = track.project(car.position)
    print(f"headless: steps={n} sim_time={n * DT:.1f}s max_speed={max_speed:.1f}km/h "
          f"final_speed={car.speed_kmh:.1f}km/h gear={car.gear} progress={proj.progress:.3f} "
          f"recoveries={getattr(car, 'recoveries', 0)} finite={finite}")
    return 0 if finite else 1


def _make_viewer(args, terrain, track, headless):
    from drivesim.viewer import Viewer  # imported lazily so --no-viewer works without it
    return Viewer(terrain, track, width=args.width, height=args.height, headless=headless)


def run_interactive(args, terrain, track, car) -> int:
    import pygame

    # The viewer opens the window, pumps the window's events and flips the display itself.
    viewer = _make_viewer(args, terrain, track, headless=False)
    reader = DeviceReader()
    reader.open()
    human = HumanInput(manual=args.manual, assists=args.assists)
    checkpoints = CheckpointTracker(track)
    clock = pygame.time.Clock()

    camera_idx = 0
    paused = False
    help_on = False
    message = None
    message_until = 0.0
    running = True
    sim_time = 0.0
    try:
        while running and not viewer.closed:
            events = pygame.event.get()
            for ev in events:
                if ev.type == pygame.QUIT:
                    running = False
            raw = reader.read(events)
            human.update(raw, 0.0 if paused else DT, car)   # paused: edges only, no ramps
            for action in human.actions:
                if action == "quit":
                    running = False
                elif action == "pause":
                    paused = not paused
                    message = "PAUSED" if paused else None
                elif action == "help":
                    help_on = not help_on
                elif action == "camera":
                    camera_idx = (camera_idx + 1) % len(CAMERAS)
                elif action == "reset":
                    xyz, yaw = checkpoints.reset_pose()
                    car.reset(xyz, yaw, terrain)
                    human.reset()
                    message, message_until = "reset to checkpoint", sim_time + 2.5
                elif action == "assist":
                    message, message_until = f"assist {'on' if human.assists else 'off'}", sim_time + 2.5
            if not paused:
                car.step(human.controls, DT, terrain)
                sim_time += DT
                checkpoints.update(car)
            if message and not paused and sim_time > message_until:
                message = None
            hud_message = HELP_TEXT if help_on else message
            viewer.render(car, hud_for(car, track, human, hud_message), camera=CAMERAS[camera_idx])
            clock.tick(args.fps)
    finally:
        viewer.close()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.headless_test or args.no_viewer:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if args.track:
        terrain, track = build_world_from_script(NAMED_SCRIPTS[args.track], seed=args.seed)
    else:
        terrain, track = generate_world(args.seed, closed=not args.open, difficulty=args.difficulty)
    car = make_car(args, terrain, track)
    if args.headless_test:
        return run_headless(args, terrain, track, car)
    if args.no_viewer:
        # simulate with idle controls so the path can be smoke-tested without a display
        human = HumanInput(manual=args.manual, assists=args.assists)
        for _ in range(int(args.fps * 2)):
            car.step(human.update(RawInput(), DT, car), DT, terrain)
        print(f"no-viewer: final speed {car.speed_kmh:.1f} km/h")
        return 0
    return run_interactive(args, terrain, track, car)


if __name__ == "__main__":
    sys.exit(main())
