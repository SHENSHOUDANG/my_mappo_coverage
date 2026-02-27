"""
Inference-only entry for multi-target UAV pursuit scenario.

设计目标:
1) 仅执行推理评估，不进行任何训练更新。
2) 支持加载 hunter/target 已训练 actor 权重。
3) 支持 N hunters + K explorers + M targets 全流程 rollout。
"""

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import imageio.v2 as imageio

try:
    import gym
    from gym import spaces
except ImportError:  # pragma: no cover
    import gymnasium as gym
    from gymnasium import spaces

from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy
from envs.env_uav_multi_infer import MultiUAVInferenceEnv
from runner.uav.role_runner import RoleBasedRunner
from utils.util import load_config


def _build_flat_args_from_cfg(merged_cfg):
    """
    功能:
        将分层配置映射为策略网络初始化所需扁平参数。
    输入:
        merged_cfg (EasyDict): 合并后的配置对象。
    输出:
        argparse.Namespace: 扁平参数对象。
    """
    # 复用已有Runner中的参数映射逻辑，避免字段漂移。
    class _Dummy(object):
        pass

    dummy = _Dummy()
    dummy.cfg = merged_cfg
    return RoleBasedRunner._build_flat_args_for_algorithm(dummy)


def _create_actor_policy(flat_args, obs_dim, action_dim, device):
    """
    功能:
        创建仅用于actor推理的 MAPPO policy 包装。
    输入:
        flat_args (argparse.Namespace): 算法参数。
        obs_dim (int): 观测维度。
        action_dim (int): 动作维度。
        device (torch.device): 推理设备。
    输出:
        RMAPPOPolicy: 可调用 act() 的策略对象。
    """
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(int(obs_dim),), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(int(action_dim),), dtype=np.float32)
    cent_obs_space = obs_space
    return RMAPPOPolicy(flat_args, obs_space, cent_obs_space, act_space, device=device)


def _load_actor_weights(policy, actor_path):
    """
    功能:
        加载 actor 权重并切换到 eval 模式。
    输入:
        policy (RMAPPOPolicy): 策略对象。
        actor_path (str): actor权重路径。
    输出:
        无。
    """
    ckpt = torch.load(str(actor_path), map_location=policy.actor.device)
    policy.actor.load_state_dict(ckpt)
    policy.actor.eval()
    policy.critic.eval()


def _policy_action(policy, obs_vec, rnn_state, deterministic=True):
    """
    功能:
        单样本前向推理动作。
    输入:
        policy (RMAPPOPolicy): 推理策略对象。
        obs_vec (np.ndarray): shape=(obs_dim,)。
        rnn_state (np.ndarray): shape=(1,recurrent_N,hidden_size)。
        deterministic (bool): 是否确定性动作。
    输出:
        tuple:
            - np.ndarray: shape=(2,) 动作。
            - np.ndarray: 更新后的rnn状态。
    """
    obs_batch = np.asarray(obs_vec, dtype=np.float32)[None, :]
    masks = np.ones((1, 1), dtype=np.float32)
    with torch.no_grad():
        action_t, next_rnn = policy.act(
            obs_batch,
            rnn_state,
            masks,
            deterministic=bool(deterministic),
        )
    action = action_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
    action = np.clip(action, -1.0, 1.0)
    return action, next_rnn.detach().cpu().numpy().astype(np.float32)


def main(args):
    """
    功能:
        多目标推理入口，执行完整 rollout 并保存指标汇总。
    输入:
        args (argparse.Namespace): 命令行参数。
    输出:
        无。
    """
    merged_cfg = load_config(args.config_file)
    if "multi_infer" not in merged_cfg:
        raise ValueError("config missing section 'multi_infer'")

    use_cuda = bool(args.cuda) and torch.cuda.is_available()
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    torch.set_num_threads(int(merged_cfg.exp.n_training_threads))

    seed = int(args.seed) if args.seed is not None else int(merged_cfg.exp.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    env = MultiUAVInferenceEnv(merged_cfg)
    env.seed(seed)

    flat_args = _build_flat_args_from_cfg(merged_cfg)

    hunter_policy = None
    if args.hunter_actor is not None:
        hunter_policy = _create_actor_policy(flat_args, obs_dim=env.obs_dim, action_dim=2, device=device)
        _load_actor_weights(hunter_policy, args.hunter_actor)
        print(f"[Infer] loaded hunter actor: {args.hunter_actor}")
    else:
        print("[Infer] hunter actor not provided, fallback to heuristic chase")

    target_policy = None
    if args.target_actor is not None:
        target_policy = _create_actor_policy(flat_args, obs_dim=env.obs_dim, action_dim=2, device=device)
        _load_actor_weights(target_policy, args.target_actor)
        print(f"[Infer] loaded target actor: {args.target_actor}")
    else:
        print("[Infer] target actor not provided, learn-target fallback to zero-action")

    recurrent_N = int(flat_args.recurrent_N)
    hidden_size = int(flat_args.hidden_size)

    episodes = int(args.episodes)
    all_episode_metrics = []

    for ep in range(episodes):
        env.seed(seed + ep * 1000)
        env.reset()
        print(
            "[EpisodeStart] ep={}, targets={}, hunters={}, explorers={}".format(
                int(ep),
                int(env.num_targets),
                int(env.num_hunters),
                int(env.num_explorers),
            )
        )

        hunter_rnn = np.zeros((env.num_hunters, recurrent_N, hidden_size), dtype=np.float32)
        target_rnn = np.zeros((env.num_targets, recurrent_N, hidden_size), dtype=np.float32)
        episode_frames = []

        if bool(args.render):
            env.render(mode="human")
        if bool(args.save_gif):
            frame0 = env.render(mode="rgb_array")
            if isinstance(frame0, np.ndarray):
                episode_frames.append(frame0.copy())

        done = False
        while not done:
            hunter_actions = None
            if hunter_policy is not None:
                hunter_actions = np.zeros((env.num_hunters, 2), dtype=np.float32)
                for hid in range(env.num_hunters):
                    if not env.hunters[hid].alive:
                        continue
                    if int(env.hunter_assignment[hid]) < 0:
                        continue
                    obs_h = env.get_hunter_obs(hid)
                    action_h, next_rnn = _policy_action(
                        hunter_policy,
                        obs_h,
                        hunter_rnn[hid][None, ...],
                        deterministic=bool(args.deterministic),
                    )
                    hunter_actions[hid] = action_h
                    hunter_rnn[hid] = next_rnn[0]

            target_actions = None
            if target_policy is not None:
                target_actions = np.zeros((env.num_targets, 2), dtype=np.float32)
                for tid in range(env.num_targets):
                    if (not bool(env.target_alive[tid])) or (str(env.targets[tid].policy_type).lower() != "learn"):
                        continue
                    obs_t = env.get_target_obs(tid)
                    action_t, next_rnn = _policy_action(
                        target_policy,
                        obs_t,
                        target_rnn[tid][None, ...],
                        deterministic=bool(args.deterministic),
                    )
                    target_actions[tid] = action_t
                    target_rnn[tid] = next_rnn[0]

            done, metrics = env.step(hunter_actions=hunter_actions, target_actions=target_actions)

            if bool(args.render):
                env.render(mode="human")
            if bool(args.save_gif) and int(env.step_count) % max(1, int(args.gif_frame_interval)) == 0:
                frame = env.render(mode="rgb_array")
                if isinstance(frame, np.ndarray):
                    episode_frames.append(frame.copy())

            if int(env.step_count) % max(1, int(args.log_interval)) == 0 or done:
                print(
                    "[Step] ep={}, step={}, captured={}, alive={}, discover_rate={:.3f}, capture_rate={:.3f}".format(
                        int(ep),
                        int(env.step_count),
                        int(metrics["targets_captured"]),
                        int(metrics["targets_alive"]),
                        float(metrics["discover_rate"]),
                        float(metrics["capture_rate"]),
                    )
                )

        ep_metrics = env.get_metrics()
        ep_metrics["episode"] = int(ep)
        all_episode_metrics.append(ep_metrics)

        if bool(args.save_gif):
            gif_name = "{}_ep{:03d}.gif".format(str(args.gif_name_prefix), int(ep))
            gif_path = Path(str(args.output_dir)) / gif_name
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            if len(episode_frames) > 0:
                imageio.mimsave(
                    str(gif_path),
                    episode_frames,
                    duration=float(args.gif_duration),
                    loop=0,
                )
                print(f"[GIF] saved episode gif: {gif_path}")

        print(
            "[EpisodeEnd] ep={}, step={}, capture_rate={:.3f}, discover_rate={:.3f}".format(
                int(ep),
                int(ep_metrics["step"]),
                float(ep_metrics["capture_rate"]),
                float(ep_metrics["discover_rate"]),
            )
        )

    avg_capture = float(np.mean([float(x["capture_rate"]) for x in all_episode_metrics])) if len(all_episode_metrics) > 0 else 0.0
    avg_discover = float(np.mean([float(x["discover_rate"]) for x in all_episode_metrics])) if len(all_episode_metrics) > 0 else 0.0
    avg_steps = float(np.mean([float(x["step"]) for x in all_episode_metrics])) if len(all_episode_metrics) > 0 else 0.0

    summary_out = {
        "config_file": str(args.config_file),
        "hunter_actor": None if args.hunter_actor is None else str(args.hunter_actor),
        "target_actor": None if args.target_actor is None else str(args.target_actor),
        "episodes": int(episodes),
        "avg_capture_rate": float(avg_capture),
        "avg_discover_rate": float(avg_discover),
        "avg_steps": float(avg_steps),
        "episode_metrics": all_episode_metrics,
    }

    out_dir = Path(str(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"multi_infer_summary_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)

    env.close()

    print(
        "[InferDone] episodes={}, avg_capture_rate={:.3f}, avg_discover_rate={:.3f}, avg_steps={:.1f}, summary={}".format(
            int(episodes),
            float(avg_capture),
            float(avg_discover),
            float(avg_steps),
            str(out_path),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference-only multi-target UAV pursuit")
    parser.add_argument("--config_file", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--hunter_actor", type=str, default=None, help="Path to hunter actor_*.pt")
    parser.add_argument("--target_actor", type=str, default=None, help="Path to target actor_*.pt")
    parser.add_argument("--episodes", type=int, default=3, help="Number of rollout episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy action")
    parser.add_argument("--seed", type=int, default=None, help="Override base seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--log_interval", type=int, default=20, help="Step interval to print metrics")
    parser.add_argument("--output_dir", type=str, default="results/multi_infer", help="Summary output directory")
    parser.add_argument("--render", action="store_true", help="Show online visualization window")
    parser.add_argument("--save_gif", action="store_true", help="Save per-episode GIF")
    parser.add_argument("--gif_frame_interval", type=int, default=2, help="Frame sampling interval in env steps")
    parser.add_argument("--gif_duration", type=float, default=1.0, help="GIF frame duration in seconds")
    parser.add_argument("--gif_name_prefix", type=str, default="multi_infer", help="GIF filename prefix")
    args = parser.parse_args()
    main(args)
