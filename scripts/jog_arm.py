#!/usr/bin/env python3
"""Manual arm jogging tool for setting up waypoints.

Provides an interactive terminal for moving the arm in small increments
and saving named waypoints as YAML snippets.

Usage:
    python scripts/jog_arm.py --mock
    python scripts/jog_arm.py --config-dir config
"""

from __future__ import annotations

import argparse
import re

import yaml

from grader.arm import ArmController
from grader.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive arm jogging and waypoint tool.",
    )
    parser.add_argument("--config-dir", default="config", help="Config directory (default: config).")
    parser.add_argument("--mock", action="store_true", help="Use mock arm (no real hardware).")
    return parser.parse_args()


# Regex for jog commands like "x+10", "z-5.5", etc.
JOG_PATTERN = re.compile(r"^([xyz])([+-]\d+(?:\.\d+)?)$", re.IGNORECASE)


def print_position(arm: ArmController) -> None:
    x, y, z = arm.get_position()
    print(f"  Position: x={x:.2f}  y={y:.2f}  z={z:.2f}")


def print_help() -> None:
    print()
    print("Commands:")
    print("  x+10, x-10, y+10, y-10, z+10, z-10  — jog by amount (any number)")
    print("  home                                  — move to home waypoint")
    print("  save <name>                           — save current position as waypoint")
    print("  goto <name>                           — move to a saved waypoint")
    print("  pos                                   — show current position")
    print("  help                                  — show this help")
    print("  quit                                  — exit")
    print()


def main() -> None:
    args = parse_args()
    cfg = load_config(config_dir=args.config_dir)
    arm = ArmController(cfg["arm"], mock_mode=args.mock)

    saved_waypoints: dict[str, tuple[float, float, float]] = {}

    print("=== Arm Jog Tool ===")
    print(f"Mock mode: {args.mock}")
    print()

    arm.connect()
    try:
        print_position(arm)
        print_help()

        while True:
            try:
                cmd = input("jog> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not cmd:
                continue

            # Quit.
            if cmd.lower() == "quit":
                print("Exiting.")
                break

            # Help.
            if cmd.lower() == "help":
                print_help()
                continue

            # Show position.
            if cmd.lower() == "pos":
                print_position(arm)
                continue

            # Home.
            if cmd.lower() == "home":
                print("  Moving to home...")
                try:
                    arm.home()
                    print_position(arm)
                except Exception as e:
                    print(f"  Error: {e}")
                continue

            # Save waypoint.
            if cmd.lower().startswith("save "):
                name = cmd[5:].strip()
                if not name:
                    print("  Usage: save <name>")
                    continue
                pos = arm.get_position()
                saved_waypoints[name] = pos
                x, y, z = pos
                snippet = {name: {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)}}
                print(f"  Waypoint '{name}' saved.")
                print("  YAML snippet:")
                print("  " + yaml.dump(snippet, default_flow_style=False).replace("\n", "\n  ").rstrip())
                continue

            # Goto waypoint.
            if cmd.lower().startswith("goto "):
                name = cmd[5:].strip()
                if not name:
                    print("  Usage: goto <name>")
                    continue
                if name not in saved_waypoints:
                    print(f"  Unknown waypoint '{name}'. Saved: {list(saved_waypoints.keys())}")
                    continue
                x, y, z = saved_waypoints[name]
                print(f"  Moving to '{name}' ({x:.2f}, {y:.2f}, {z:.2f})...")
                try:
                    arm.move_to(x, y, z)
                    print_position(arm)
                except Exception as e:
                    print(f"  Error: {e}")
                continue

            # Jog command.
            match = JOG_PATTERN.match(cmd)
            if match:
                axis = match.group(1).lower()
                delta = float(match.group(2))
                x, y, z = arm.get_position()
                if axis == "x":
                    x += delta
                elif axis == "y":
                    y += delta
                elif axis == "z":
                    z += delta
                try:
                    arm.move_to(x, y, z)
                    print_position(arm)
                except ValueError as e:
                    print(f"  Out of bounds: {e}")
                except Exception as e:
                    print(f"  Error: {e}")
                continue

            print(f"  Unknown command: '{cmd}'. Type 'help' for commands.")

    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
