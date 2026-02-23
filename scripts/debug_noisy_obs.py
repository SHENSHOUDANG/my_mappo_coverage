#!/usr/bin/env python3
import argparse
import numpy as np

import os
import sys
# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from train.train_uav_pursuit import parse_training_config
from envs.uav_pursuit_env import MultiUavPursuitEnv


def build_env(all_args):
    return MultiUavPursuitEnv(
        num_hunters=all_args["num_hunters"],
        num_blockers=all_args["num_blockers"],
        world_size=all_args["world_size"],
        dt=all_args["dt"],
        capture_radius=all_args["capture_radius"],
        capture_steps=all_args["capture_steps"],
        collision_radius=all_args.get("collision_radius", 0.02),
        collision_penalty_k=all_args.get("collision_penalty_k", 5.0),
        noisy_target_info_when_unseen=all_args.get("noisy_target_info_when_unseen", False),
        noisy_target_pos_std=all_args.get("noisy_target_pos_std", 0.0),
        noisy_target_vel_std=all_args.get("noisy_target_vel_std", 0.0),
        lost_target_penalty=all_args.get("lost_target_penalty", 0.0),
        lost_target_penalty_age_scale=all_args.get("lost_target_penalty_age_scale", 0.0),
        max_steps=all_args["episode_length"],
        seed=all_args["seed"],
        target_policy_source=all_args["target_policy_source"],
        target_patrol_path=all_args.get("target_patrol_path"),
        target_patrol_names=all_args.get("target_patrol_names"),
        max_speed_hunter=all_args["max_speed_hunter"],
        max_speed_blocker=all_args["max_speed_blocker"],
        max_speed_target=all_args["max_speed_target"],
        perception_hunter=all_args["perception_hunter"],
        perception_blocker=all_args["perception_blocker"],
        perception_target=all_args["perception_target"],
        speed_penalty=all_args["speed_penalty"],
    )


def main():
    parser = argparse.ArgumentParser(description="Debug noisy pursuit observations.")
    parser.add_argument("--config", default="config/minimal_test.yaml", help="Config YAML path")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to print")
    parser.add_argument("--agent", type=int, default=0, help="Agent index to print")
    parser.add_argument("--noisy", action="store_true", help="Force noisy_target_info_when_unseen=true")
    parser.add_argument("--pursuit_speed", type=float, default=0.0, help="Max speed for hunter/blocker")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    args = parser.parse_args()

    all_args, _ = parse_training_config(["--config", args.config])
    all_args["max_speed_hunter"] = args.pursuit_speed
    all_args["max_speed_blocker"] = args.pursuit_speed
    if args.noisy:
        all_args["noisy_target_info_when_unseen"] = True
    if args.seed is not None:
        all_args["seed"] = args.seed

    env = build_env(all_args)
    obs = env.reset()
    target_idx = env.agent_num - 1
    agent_pos = env.positions[args.agent]
    agent_vel = env.velocities[args.agent]
    target_pos = env.positions[target_idx]
    target_vel = env.velocities[target_idx]
    rel_pos = target_pos - agent_pos
    rel_vel = target_vel - agent_vel
    print("step", 0, f"agent{args.agent}_pos", agent_pos, "agent_vel", agent_vel)
    print("step", 0, "target_pos", target_pos, "target_vel", target_vel)
    print("step", 0, "rel_pos", rel_pos, "rel_vel", rel_vel)
    print("step", 0, f"obs{args.agent}", obs[args.agent])

    for t in range(1, args.steps + 1):
        actions = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)
        actions[target_idx] = np.random.uniform(-1.0, 1.0, size=env.action_dim)
        obs, rewards, dones, infos = env.step(actions)
        agent_pos = env.positions[args.agent]
        agent_vel = env.velocities[args.agent]
        target_pos = env.positions[target_idx]
        target_vel = env.velocities[target_idx]
        rel_pos = target_pos - agent_pos
        rel_vel = target_vel - agent_vel
        print("step", t, f"agent{args.agent}_pos", agent_pos, "agent_vel", agent_vel)
        print("step", t, "target_pos", target_pos, "target_vel", target_vel)
        print("step", t, "rel_pos", rel_pos, "rel_vel", rel_vel)
        print("step", t, f"rel-tgt obs{args.agent}", obs[args.agent][-10:-5])
        print("step", t, f"mem obs{args.agent}", obs[args.agent][-5:])
        print('-' * 10)

    env.close()


if __name__ == "__main__":
    main()
