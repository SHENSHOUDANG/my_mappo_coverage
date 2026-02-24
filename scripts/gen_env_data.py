#!/usr/bin/env python3
"""Batch scenario generator based on one template YAML."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate many scenario YAML files under one train group."
    )
    parser.add_argument(
        "--group-id",
        type=str,
        required=True,
        help="Scenario group id, e.g. 00. Template should be datasets/train/<group-id>/001.yaml",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Optional template YAML path. Default: datasets/train/<group-id>/001.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: template parent directory.",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        required=True,
        help="How many scenario files to generate.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Start scenario index, e.g. 1 -> 001.yaml",
    )
    parser.add_argument(
        "--world-size-min",
        type=float,
        default=200.0,
        help="Minimum world_size for generated scenes.",
    )
    parser.add_argument(
        "--world-size-max",
        type=float,
        default=600.0,
        help="Maximum world_size for generated scenes.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Optional first seed. Default: template seed.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Seed increment per generated scene.",
    )
    parser.add_argument(
        "--target-policy-types",
        type=str,
        default="patrol,random,learn",
        help="Comma-separated policy cycle for Target.policy_type.",
    )
    parser.add_argument(
        "--patrol-path",
        type=str,
        default=None,
        help="Optional override for Target.patrol_path.",
    )
    parser.add_argument(
        "--patrol-source-json",
        type=Path,
        default=None,
        help="Optional patrol JSON path; if set, patrol route names are sampled from this file.",
    )
    parser.add_argument(
        "--patrol-count",
        type=int,
        default=1,
        help="How many patrol route names to assign when policy is patrol.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible world_size/patrol sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing YAML files.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Template must be a YAML mapping: {path}")
    return data


def dump_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_patrol_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    names: list[str] = []
    for route in data.get("routes", []):
        if isinstance(route, dict):
            name = route.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def ensure_required_fields(template: dict) -> None:
    for key in ("world_size", "seed", "Hunter", "Explorer", "Target"):
        if key not in template:
            raise ValueError(f"Template missing required field: {key}")
    for role in ("Hunter", "Explorer", "Target"):
        node = template.get(role)
        if not isinstance(node, dict):
            raise ValueError(f"Template field '{role}' must be a mapping.")
        for key in ("perception_radius", "max_velo"):
            if key not in node:
                raise ValueError(f"Template field '{role}.{key}' is required.")


def main() -> None:
    args = parse_args()
    if args.num_scenes <= 0:
        raise ValueError("--num-scenes must be > 0")
    if args.start_id <= 0:
        raise ValueError("--start-id must be > 0")
    if args.world_size_max < args.world_size_min:
        raise ValueError("--world-size-max must be >= --world-size-min")
    if args.patrol_count <= 0:
        raise ValueError("--patrol-count must be > 0")

    rng = random.Random(args.seed)
    policy_types = [p.strip() for p in args.target_policy_types.split(",") if p.strip()]
    if not policy_types:
        raise ValueError("No valid policy in --target-policy-types")

    template_path = (
        args.template
        if args.template is not None
        else Path("datasets") / "train" / args.group_id / "001.yaml"
    )
    template = load_yaml(template_path)
    ensure_required_fields(template)

    output_dir = args.output_dir if args.output_dir is not None else template_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    patrol_names_pool: list[str] = []
    if args.patrol_source_json is not None:
        patrol_names_pool = load_patrol_names(args.patrol_source_json)
        if not patrol_names_pool:
            raise ValueError(f"No patrol names found in {args.patrol_source_json}")

    seed0 = int(args.seed_start) if args.seed_start is not None else int(template["seed"])
    generated = 0
    skipped = 0

    for offset in range(args.num_scenes):
        scene_id = args.start_id + offset
        out_path = output_dir / f"{scene_id:03d}.yaml"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        scene = yaml.safe_load(yaml.safe_dump(template, sort_keys=False))
        scene["world_size"] = float(rng.uniform(args.world_size_min, args.world_size_max))
        scene["seed"] = int(seed0 + offset * args.seed_step)

        policy_type = policy_types[offset % len(policy_types)]
        target = dict(scene["Target"])
        target["policy_type"] = policy_type
        if args.patrol_path is not None:
            target["patrol_path"] = args.patrol_path

        if policy_type == "patrol":
            if patrol_names_pool:
                if len(patrol_names_pool) >= args.patrol_count:
                    names = rng.sample(patrol_names_pool, args.patrol_count)
                else:
                    names = [rng.choice(patrol_names_pool) for _ in range(args.patrol_count)]
                target["patrol_names"] = names
            else:
                current_names = target.get("patrol_names", [])
                if isinstance(current_names, list) and current_names:
                    if len(current_names) >= args.patrol_count:
                        target["patrol_names"] = current_names[: args.patrol_count]
                    else:
                        target["patrol_names"] = current_names + [current_names[-1]] * (
                            args.patrol_count - len(current_names)
                        )
                else:
                    target["patrol_names"] = ["square"]
        else:
            target["patrol_names"] = []

        scene["Target"] = target
        dump_yaml(out_path, scene)
        generated += 1

    print(
        f"Generated {generated} scene files under {output_dir}. "
        f"Skipped existing: {skipped} (use --overwrite to replace)."
    )
    print(
        "Kept fixed from template within this group: "
        "Hunter/Explorer/Target perception_radius and max_velo."
    )


if __name__ == "__main__":
    main()
