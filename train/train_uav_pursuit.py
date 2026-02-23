"""
# @Time    : 2024/xx/xx
# @Author  : OpenAI
# @File    : train_uav_pursuit.py
"""

import os
import sys
from pathlib import Path
from typing import Any

import yaml

import matplotlib.pyplot as plt
import numpy as np
import setproctitle
import torch

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from envs.env_wrappers import DummyVecEnv
from envs.uav_pursuit_env import MultiUavPursuitEnv
from runner.shared.uav_pursuit_runner import UavPursuitRunner


ROLE_NAMES = ("hunter", "blocker", "target")


class ConfigDict(dict):
    """Dict config with attribute-style access."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _get_role_params(all_args):
    return {
        "max_speed_hunter": all_args["max_speed_hunter"],
        "max_speed_blocker": all_args["max_speed_blocker"],
        "max_speed_target": all_args["max_speed_target"],
        "perception_hunter": all_args["perception_hunter"],
        "perception_blocker": all_args["perception_blocker"],
        "perception_target": all_args["perception_target"],
        "speed_penalty": all_args["speed_penalty"],
    }


def _render_perception_preview(all_args):
    world_size = all_args["world_size"]
    fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("Perception Preview Before Training")

    positions = {
        "hunter": (-0.55 * world_size, 0.0),
        "blocker": (0.0, 0.0),
        "target": (0.55 * world_size, 0.0),
    }
    colors = {"hunter": "#1f77b4", "blocker": "#2ca02c", "target": "#d62728"}
    markers = {"hunter": "o", "blocker": "s", "target": "*"}

    for role in ROLE_NAMES:
        perception = all_args[f"perception_{role}"]
        x, y = positions[role]
        ax.scatter([x], [y], c=colors[role], s=90, marker=markers[role], label=f"{role} (speed={all_args[f'max_speed_{role}']:.2f})")
        ax.add_patch(plt.Circle((x, y), perception, color=colors[role], alpha=0.12))
        ax.text(x, y, f" {role}\nR={perception:.2f}", fontsize=9, va="bottom")

    ax.legend(loc="upper right")
    plt.show(block=False)
    plt.pause(0.001)


def _interactive_confirm_perception(all_args):
    while True:
        _render_perception_preview(all_args)
        try:
            answer = input(
                "\n当前感知范围: "
                f"hunter={all_args['perception_hunter']}, "
                f"blocker={all_args['perception_blocker']}, "
                f"target={all_args['perception_target']}. "
                "输入 y 确认开始训练；输入 n 重新修改: "
            ).strip().lower()
        except EOFError:
            print("未检测到交互输入，使用当前配置继续训练。")
            plt.close("all")
            return

        if answer == "y":
            plt.close("all")
            return

        if answer != "n":
            print("无效输入，请输入 y 或 n。")
            plt.close("all")
            continue

        for role in ROLE_NAMES:
            key = f"perception_{role}"
            while True:
                raw = input(f"请输入 {role} 的感知范围(当前 {all_args[key]}): ").strip()
                try:
                    val = float(raw)
                except ValueError:
                    print("请输入合法数字。")
                    continue
                if val <= 0.0:
                    print("感知范围必须大于 0。")
                    continue
                all_args[key] = val
                break
        plt.close("all")


def make_train_env(all_args):
    role_params = _get_role_params(all_args)

    def get_env_fn(rank):
        def init_env():
            env = MultiUavPursuitEnv(
                num_hunters=all_args["num_hunters"],
                num_blockers=all_args["num_blockers"],
                world_size=all_args["world_size"],
                dt=all_args["dt"],
                capture_radius=all_args["capture_radius"],
                capture_steps=all_args["capture_steps"],
                collision_radius=all_args.get("collision_radius", 0.02),
                collision_penalty_k=all_args.get("collision_penalty_k", 5.0),
                noisy_target_info_when_unseen=all_args.get("noisy_target_info_when_unseen", False),
                noisy_target_pos_std=all_args.get("noisy_target_pos_std", 0.02),
                noisy_target_vel_std=all_args.get("noisy_target_vel_std", 0.02),
                lost_target_penalty=all_args.get("lost_target_penalty", 0.0),
                lost_target_penalty_age_scale=all_args.get("lost_target_penalty_age_scale", 0.0),
                max_steps=all_args["episode_length"],
                seed=all_args["seed"] + rank * 1000,
                target_policy_source=all_args["target_policy_source"],
                target_patrol_path=all_args.get("target_patrol_path"),
                target_patrol_names=all_args.get("target_patrol_names"),
                **role_params,
            )
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args["n_rollout_threads"])])


def make_eval_env(all_args):
    role_params = _get_role_params(all_args)

    def get_env_fn(rank):
        def init_env():
            env = MultiUavPursuitEnv(
                num_hunters=all_args["num_hunters"],
                num_blockers=all_args["num_blockers"],
                world_size=all_args["world_size"],
                dt=all_args["dt"],
                capture_radius=all_args["capture_radius"],
                capture_steps=all_args["capture_steps"],
                collision_radius=all_args.get("collision_radius", 0.02),
                collision_penalty_k=all_args.get("collision_penalty_k", 5.0),
                noisy_target_info_when_unseen=all_args.get("noisy_target_info_when_unseen", False),
                noisy_target_pos_std=all_args.get("noisy_target_pos_std", 0.02),
                noisy_target_vel_std=all_args.get("noisy_target_vel_std", 0.02),
                lost_target_penalty=all_args.get("lost_target_penalty", 0.0),
                lost_target_penalty_age_scale=all_args.get("lost_target_penalty_age_scale", 0.0),
                max_steps=all_args["episode_length"],
                seed=all_args["seed"] + rank * 1000,
                target_policy_source=all_args["target_policy_source"],
                target_patrol_path=all_args.get("target_patrol_path"),
                target_patrol_names=all_args.get("target_patrol_names"),
                **role_params,
            )
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args["n_eval_rollout_threads"])])


def _resolve_config_path(config_path):
    if not config_path:
        return None
    path = Path(config_path)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parent.parent
    return (repo_root / path).resolve()



def _load_yaml_mapping(yaml_path):
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Scenario file must be a mapping: {yaml_path}")
    return data


def _load_scenario_suite(split_dir):
    resolved = _resolve_config_path(split_dir)
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"Scenario split directory does not exist: {resolved}")
    required_fields = {
        "num_hunters", "num_blockers", "world_size", "dt", "capture_radius",
        "capture_steps", "episode_length", "seed", "initial_positions", "target_patrol_route_id",
    }
    files = sorted([p for p in resolved.iterdir() if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}], key=lambda x: x.stem)
    scenarios = []
    for fp in files:
        cfg = _load_yaml_mapping(fp)
        cfg.setdefault("scenario_id", fp.stem)
        cfg.setdefault("scenario_file", str(fp))
        scenarios.append(cfg)

    normalized = []
    for idx, sc in enumerate(scenarios):
        sc = dict(sc)
        missing = sorted(required_fields - set(sc.keys()))
        if missing:
            raise ValueError(f"scenario entry missing fields: {', '.join(missing)}")
        sc.setdefault("scenario_id", f"scenario_{idx}")
        normalized.append(sc)
    return normalized


def _flatten_config(node: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in node.items():
        if isinstance(value, dict):
            flat.update(_flatten_config(value))
        else:
            if key in flat and flat[key] != value:
                raise ValueError(f"Conflicting values for key '{key}' in YAML config")
            flat[key] = value
    return flat


def _resolve_config_arg(args):
    if not args:
        return "config/pursuit3v1.yaml"
    if len(args) == 1 and not args[0].startswith("-"):
        return args[0]
    for idx, arg in enumerate(args):
        if arg == "--config" and idx + 1 < len(args):
            return args[idx + 1]
    return "config/pursuit3v1.yaml"


def parse_training_config(args):
    config_path = _resolve_config_arg(args)
    resolved_config = _resolve_config_path(config_path)
    yaml_data = _load_yaml_mapping(resolved_config)
    all_args = _flatten_config(yaml_data)

    dataset_root = Path(all_args.get("dataset_root", "datasets"))
    all_args["scenario_suite_val"] = str(dataset_root / "val")
    all_args["scenario_suite_test"] = str(dataset_root / "test")
    all_args.setdefault("eval_dataset_split", "val")
    all_args.setdefault("train_patrol_route_dir", str(dataset_root / "val" / "patrol_routes"))
    all_args.setdefault("interactive_perception_confirm", False)

    all_args["num_agents"] = all_args["num_hunters"] + all_args["num_blockers"] + 1
    all_args["scenario_suite_data"] = _load_scenario_suite(all_args["scenario_suite_val"])
    all_args["test_suite_data"] = _load_scenario_suite(all_args["scenario_suite_test"])
    return ConfigDict(all_args), str(resolved_config)


def main(args):
    all_args, config_path = parse_training_config(args)

    if all_args["algorithm_name"] == "rmappo":
        assert all_args["use_recurrent_policy"] or all_args["use_naive_recurrent_policy"], "check recurrent policy!"
    elif all_args["algorithm_name"] == "mappo":
        assert all_args["use_recurrent_policy"] is False and all_args["use_naive_recurrent_policy"] is False, "check recurrent policy!"
    else:
        raise NotImplementedError

    if all_args["cuda"] and torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            torch.cuda.init()
            _ = torch.randn(1, 1, device="cuda") @ torch.randn(1, 1, device="cuda")
            device = torch.device("cuda:0")
            if all_args["cuda_deterministic"]:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
        except Exception as exc:
            device = torch.device("cpu")
            all_args["cuda"] = False
    else:
        device = torch.device("cpu")

    torch.set_num_threads(all_args["n_training_threads"])

    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args["env_name"]
        / all_args["scenario_name"]
        / all_args["algorithm_name"]
        / all_args["experiment_name"]
    )
    os.makedirs(str(run_dir), exist_ok=True)

    exst_run_nums = [
        int(str(folder.name).split("run")[1])
        for folder in run_dir.iterdir()
        if str(folder.name).startswith("run")
    ]
    curr_run = "run1" if len(exst_run_nums) == 0 else f"run{max(exst_run_nums) + 1}"
    run_dir = run_dir / curr_run
    os.makedirs(str(run_dir), exist_ok=True)
    print(f"Run dir: {str(run_dir)}")
    print(f"TensorBoard: tensorboard --logdir {str(run_dir / 'logs')}")

    if config_path:
        config_file = Path(config_path)
        config_target = run_dir / config_file.name
        config_target.write_text(config_file.read_text(encoding="utf-8"), encoding="utf-8")

    setproctitle.setproctitle(
        str(all_args["algorithm_name"])
        + "-"
        + str(all_args["env_name"])
        + "-"
        + str(all_args["experiment_name"])
        + "@"
        + str(all_args["user_name"])
    )

    torch.manual_seed(all_args["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(all_args["seed"])
    np.random.seed(all_args["seed"])

    if all_args["interactive_perception_confirm"]:
        if sys.stdin.isatty():
            _interactive_confirm_perception(all_args)
        else:
            print("Non-interactive session detected; skipping perception confirmation.")

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args["use_eval"] else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args["num_agents"],
        "device": device,
        "run_dir": run_dir,
    }

    runner = UavPursuitRunner(config)
    runner.run()

    envs.close()
    if all_args["use_eval"] and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
