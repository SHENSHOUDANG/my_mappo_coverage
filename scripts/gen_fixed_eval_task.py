#!/usr/bin/env python3
"""
生成固定评估任务JSON文件（支持world_size区间与最大hunter数量约束）。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    """
    功能:
        解析命令行参数。
    输入:
        无。
    输出:
        argparse.Namespace: 解析后的参数对象。
    """
    # Step 1: 构建参数解析器并注册基础参数
    parser = argparse.ArgumentParser(
        description="Generate fixed eval task specs JSON for UAV pursuit."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出任务JSON路径，例如 config/eval_tasks/fixed_eval_generated.json",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        required=True,
        help="生成任务总数。",
    )
    parser.add_argument(
        "--max_hunters",
        type=int,
        required=True,
        help="任务中允许的最大hunter数量（最小固定为1）。",
    )
    parser.add_argument(
        "--world_size",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="可选world_size采样区间[min,max]；未提供则不写入world_size字段。",
    )

    # Step 2: 注册策略与巡逻相关参数
    parser.add_argument(
        "--target_policy_choices",
        type=str,
        default="random,patrol",
        help="Target策略候选，逗号分隔，例如 random,patrol。",
    )
    parser.add_argument(
        "--target_patrol_paths",
        type=str,
        nargs="+",
        default=["datasets/patrol_routes.json"],
        help="patrol路线JSON路径列表（空格分隔）；也兼容单参数逗号分隔写法。",
    )
    parser.add_argument(
        "--target_route_id",
        type=int,
        default=0,
        help="写入任务的target_route_id（通常保持0）。",
    )

    # Step 3: 注册随机种子控制参数
    parser.add_argument(
        "--seed_start",
        type=int,
        default=10000,
        help="任务seed起始值。",
    )
    parser.add_argument(
        "--seed_step",
        type=int,
        default=1,
        help="任务seed递增步长。",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=2026,
        help="生成器随机种子（用于hunter/world_size/策略采样）。",
    )
    return parser.parse_args()


def _load_route_names_from_json(route_path: Path) -> list[str]:
    """
    功能:
        从巡逻路线JSON中读取全部可用路线名。
    输入:
        route_path (Path): 路线JSON文件路径。
    输出:
        list[str]: 路线名列表。
    """
    # Step 1: 文件存在性校验与JSON读取
    if not route_path.exists():
        raise FileNotFoundError(f"Patrol route file not found: {route_path}")
    with route_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Step 2: 解析routes数组中的name字段
    names: list[str] = []
    for route in data.get("routes", []):
        if not isinstance(route, dict):
            continue
        name = route.get("name")
        if isinstance(name, str) and len(name.strip()) > 0:
            names.append(name.strip())
    return names


def _resolve_patrol_route_pool(target_patrol_paths_arg: list[str]) -> list[tuple[str, str]]:
    """
    功能:
        解析多个patrol路线文件，构建(路径, 路线名)候选池。
    输入:
        target_patrol_paths_arg (list[str]): 巡逻路线JSON路径列表（支持元素内逗号拼接）。
    输出:
        list[tuple[str, str]]: 可采样的(路径, 路线名)元组列表。
    """
    # Step 1: 解析路径列表（兼容空格分隔与逗号分隔）
    raw_items = target_patrol_paths_arg if isinstance(target_patrol_paths_arg, list) else [str(target_patrol_paths_arg)]
    path_list: list[str] = []
    for raw in raw_items:
        path_list.extend([x.strip() for x in str(raw).split(",") if x.strip()])
    if len(path_list) == 0:
        raise ValueError("No valid path in --target_patrol_paths")

    # Step 2: 逐路径读取全部route name并构建候选池
    route_pool: list[tuple[str, str]] = []
    for raw_path in path_list:
        names = _load_route_names_from_json(Path(raw_path))
        for name in names:
            route_pool.append((str(raw_path), str(name)))
    return route_pool


def build_tasks(args: argparse.Namespace) -> list[dict]:
    """
    功能:
        根据输入参数生成固定评估任务列表。
    输入:
        args (argparse.Namespace): 命令行参数对象。
    输出:
        list[dict]: 任务规格字典列表。
    """
    # Step 1: 参数有效性校验
    if int(args.num_tasks) <= 0:
        raise ValueError("--num_tasks must be > 0")
    if int(args.max_hunters) < 1:
        raise ValueError("--max_hunters must be >= 1")
    world_size_range = None
    if args.world_size is not None:
        world_min = float(args.world_size[0])
        world_max = float(args.world_size[1])
        if world_max < world_min:
            raise ValueError("--world_size MAX must be >= MIN")
        world_size_range = (world_min, world_max)

    # Step 2: 解析策略候选与巡逻路线候选
    policy_choices = [x.strip().lower() for x in str(args.target_policy_choices).split(",") if x.strip()]
    if len(policy_choices) == 0:
        raise ValueError("No valid policy in --target_policy_choices")
    patrol_route_pool = _resolve_patrol_route_pool(list(args.target_patrol_paths))
    if "patrol" in policy_choices and len(patrol_route_pool) == 0:
        raise ValueError("Patrol policy requested but no valid patrol route names found")

    # Step 3: 采样生成任务列表
    rng = random.Random(int(args.rand_seed))
    tasks: list[dict] = []
    for idx in range(int(args.num_tasks)):
        policy = rng.choice(policy_choices)
        num_hunters = int(rng.randint(1, int(args.max_hunters)))
        route_path, route_name = rng.choice(patrol_route_pool)
        task = {
            "num_hunters": int(num_hunters),
            "seed": int(int(args.seed_start) + idx * int(args.seed_step)),
            "target_policy_source": str(policy),
            "target_patrol_path": str(route_path),
            "target_route_id": int(args.target_route_id),
        }
        if world_size_range is not None:
            task["world_size"] = float(rng.uniform(float(world_size_range[0]), float(world_size_range[1])))
        if policy == "patrol":
            task["target_patrol_names"] = [str(route_name)]
        else:
            task["target_patrol_names"] = [str(route_name)]
        tasks.append(task)
    return tasks


def main() -> None:
    """
    功能:
        程序入口：生成任务并写入JSON文件。
    输入:
        无（从CLI读取）。
    输出:
        无。
    """
    # Step 1: 读取参数并生成任务列表
    args = parse_args()
    tasks = build_tasks(args)

    # Step 2: 写出JSON文件（兼容train.py中带tasks字段的格式）
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() in [".yaml", ".yml"]:
        _dump_grouped_yaml(args.output, tasks)
    else:
        payload = {"tasks": tasks}
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # Step 3: 打印生成摘要
    print(
        "Generated {} fixed eval tasks to {} (max_hunters={}, world_size={}).".format(
            int(len(tasks)),
            str(args.output),
            int(args.max_hunters),
            "disabled" if args.world_size is None else f"[{float(args.world_size[0])}, {float(args.world_size[1])}]",
        )
    )


def _dump_grouped_yaml(output_path: Path, tasks: list[dict]) -> None:
    """
    功能:
        将任务按num_hunters分组并以紧凑对齐风格写入YAML，便于人工检查。
    输入:
        output_path (Path): 输出YAML路径。
        tasks (list[dict]): 任务列表。
    输出:
        无。
    """
    # Step 1: 按num_hunters分组并组内按seed排序
    grouped = defaultdict(list)
    for task in tasks:
        grouped[int(task["num_hunters"])].append(dict(task))
    for hunters in grouped:
        grouped[hunters] = sorted(grouped[hunters], key=lambda x: int(x.get("seed", 0)))

    # Step 2: 统一字段顺序，构建可读性更强的inline映射行
    preferred_order = [
        "num_hunters",
        "world_size",
        "seed",
        "target_policy_source",
        "target_patrol_path",
        "target_patrol_names",
        "target_route_id",
    ]
    lines = ["tasks:"]
    for group_id, hunters in enumerate(sorted(grouped.keys())):
        if group_id > 0:
            lines.append("")
        lines.append(f"  # num_hunters = {int(hunters)}")
        for task in grouped[hunters]:
            kv_parts = []
            for key in preferred_order:
                if key not in task:
                    continue
                value = task[key]
                if isinstance(value, str):
                    value_repr = value
                else:
                    value_repr = json.dumps(value, ensure_ascii=False)
                kv_parts.append(f"{key}: {value_repr}")
            lines.append("  - {" + ", ".join(kv_parts) + "}")

    # Step 3: 写入文件
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
