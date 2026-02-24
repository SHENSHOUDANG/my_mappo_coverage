#!/usr/bin/env python3
"""Batch generator for patrol route JSON files."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


DEFAULT_META = {"version": 1, "coords": "normalized_0_1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate many patrol routes in one JSON file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/patrol_routes.json"),
        help="Output patrol JSON path.",
    )
    parser.add_argument(
        "--num-routes",
        type=int,
        required=True,
        help="How many routes to generate.",
    )
    parser.add_argument(
        "--points-min",
        type=int,
        default=8,
        help="Minimum waypoints per route.",
    )
    parser.add_argument(
        "--points-max",
        type=int,
        default=20,
        help="Maximum waypoints per route.",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="auto",
        help="Route name prefix.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index used in generated names.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing file; otherwise append routes to it.",
    )
    return parser.parse_args()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def generate_route(rng: random.Random, n_points: int) -> list[list[float]]:
    cx = rng.uniform(0.2, 0.8)
    cy = rng.uniform(0.2, 0.8)
    base_r = rng.uniform(0.12, 0.35)

    angles = sorted(rng.uniform(0.0, 2.0 * math.pi) for _ in range(n_points))
    points: list[list[float]] = []
    for angle in angles:
        r = base_r * rng.uniform(0.6, 1.4)
        x = clamp01(cx + r * math.cos(angle))
        y = clamp01(cy + r * math.sin(angle))
        points.append([x, y])
    return points


def make_unique_name(base: str, existing: set[str]) -> str:
    if base not in existing:
        existing.add(base)
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    name = f"{base}_{i}"
    existing.add(name)
    return name


def load_or_init(path: Path, overwrite: bool) -> dict:
    if overwrite or not path.exists():
        return {"meta": dict(DEFAULT_META), "routes": []}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"meta": dict(DEFAULT_META), "routes": []}

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else dict(DEFAULT_META)
    routes = data.get("routes") if isinstance(data.get("routes"), list) else []
    return {"meta": meta, "routes": routes}


def main() -> None:
    args = parse_args()
    if args.num_routes <= 0:
        raise ValueError("--num-routes must be > 0")
    if args.points_min < 2:
        raise ValueError("--points-min must be >= 2")
    if args.points_max < args.points_min:
        raise ValueError("--points-max must be >= --points-min")

    rng = random.Random(args.seed)
    data = load_or_init(args.output, overwrite=args.overwrite)
    data.setdefault("meta", dict(DEFAULT_META))
    data.setdefault("routes", [])

    existing_names: set[str] = set()
    for route in data["routes"]:
        if isinstance(route, dict):
            name = route.get("name")
            if isinstance(name, str):
                existing_names.add(name)

    generated = 0
    next_id = args.start_index
    while generated < args.num_routes:
        n_points = rng.randint(args.points_min, args.points_max)
        base_name = f"{args.name_prefix}_{next_id:03d}"
        name = make_unique_name(base_name, existing_names)
        waypoints = generate_route(rng, n_points)
        data["routes"].append({"name": name, "waypoints": waypoints})
        generated += 1
        next_id += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"Generated {generated} routes into {args.output} (total routes: {len(data['routes'])})."
    )


if __name__ == "__main__":
    main()
