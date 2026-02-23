import shutil
import time
from pathlib import Path

import yaml
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.collections import LineCollection

from envs.uav_pursuit_env import MultiUavPursuitEnv
from runner.shared.env_runner import EnvRunner


def _t2n(x):
    return x.detach().cpu().numpy()


class UavPursuitRunner(EnvRunner):
    def __init__(self, config):
        super().__init__(config)
        self.gif_interval = getattr(self.all_args, "gif_interval", 10)
        self.gif_frame_duration = getattr(self.all_args, "gif_frame_duration", 0.1)
        self.gif_dir = Path(self.run_dir) / "gifs"
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        # val/test 场景评估 GIF 输出目录。
        self.eval_gif_dir = Path(self.run_dir) / "eval_gifs"
        self.eval_gif_dir.mkdir(parents=True, exist_ok=True)
        self._patrol_rng = np.random.RandomState(self.all_args.seed + 2027)
        self._best_train_avg_reward = None
        self._best_eval_avg_reward = None
        self._best_capture_success_rate = None
        self._best_avg_capture_steps = None
        self.scenario_suite = getattr(self.all_args, "scenario_suite_data", [])
        self.test_suite = getattr(self.all_args, "test_suite_data", [])
        self.best_model_dir = Path(self.run_dir) / "best_models"
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_report_best_metrics(
        self,
        total_num_steps,
        train_avg_reward=None,
        eval_avg_reward=None,
        capture_success_rate=None,
        avg_capture_steps=None,
    ):
        lines = []

        if train_avg_reward is not None:
            if self._best_train_avg_reward is None or train_avg_reward > self._best_train_avg_reward:
                self._best_train_avg_reward = float(train_avg_reward)
                lines.append(f"best_train_avg_reward: {self._best_train_avg_reward:.4f}")

        if eval_avg_reward is not None:
            if self._best_eval_avg_reward is None or eval_avg_reward > self._best_eval_avg_reward:
                self._best_eval_avg_reward = float(eval_avg_reward)
                lines.append(f"best_eval_avg_reward: {self._best_eval_avg_reward:.4f}")

        if capture_success_rate is not None:
            if self._best_capture_success_rate is None or capture_success_rate > self._best_capture_success_rate:
                self._best_capture_success_rate = float(capture_success_rate)
                lines.append(f"best_capture_success_rate: {self._best_capture_success_rate:.4f}")

        if avg_capture_steps is not None:
            if self._best_avg_capture_steps is None or avg_capture_steps < self._best_avg_capture_steps:
                self._best_avg_capture_steps = float(avg_capture_steps)
                lines.append(f"best_avg_capture_steps: {self._best_avg_capture_steps:.2f}")

        if lines:
            print(f"[step {int(total_num_steps)}] " + " | ".join(lines))

    def _load_patrol_waypoints_from_file(self, route_file):
        # 统一读取 datasets/<split>/patrol_routes 下的 yaml 巡逻路径文件。
        if route_file is None:
            return None
        route_path = Path(route_file)
        payload = yaml.safe_load(route_path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict):
            return payload.get("waypoints", payload)
        return payload

    def _resolve_route_file(self, scenario):
        # 路径文件位于场景同级 patrol_routes 目录，按 route_id 定位。
        route_id = scenario.get("target_patrol_route_id")
        if route_id in (None, "", "null"):
            return None
        if not scenario.get("scenario_file"):
            return None
        base_dir = Path(scenario["scenario_file"]).resolve().parent / "patrol_routes"
        route_stem = str(route_id)
        candidates = [route_stem]
        if route_stem.isdigit():
            candidates.append(route_stem.zfill(3))
        for stem in candidates:
            for ext in (".yaml", ".yml"):
                fp = base_dir / f"{stem}{ext}"
                if fp.exists():
                    return fp
        return None

    def _sample_train_patrol_route(self):
        # 训练阶段也采用与 val/test 一致的 patrol_routes 文件管理。
        route_dir = Path(getattr(self.all_args, "scenario_suite_val", "")) / "patrol_routes"
        if not route_dir.exists():
            return None
        files = sorted([p for p in route_dir.iterdir() if p.suffix.lower() in {".yaml", ".yml"}], key=lambda x: x.stem)
        if not files:
            return None
        return files[self._patrol_rng.randint(0, len(files))]

    def _save_models_to_dir(self, target_dir):
        # 保存当前策略快照，用于后续 test 阶段评估。
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        self.save()
        for model_file in Path(self.save_dir).glob("*.pt"):
            shutil.copy2(model_file, target_dir / model_file.name)

    def _load_models_from_dir(self, model_dir):
        # 加载指定快照策略进行 test 评估。
        self.model_dir = str(model_dir)
        self.restore()

    def run(self):
        # 训练阶段默认以 val 集进行评估，并维护三类最优模型。
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            # 训练 patrol 轨迹使用 datasets/*/patrol_routes 的文件管理方式。
            if self.target_policy_source == "patrol":
                route_file = self._sample_train_patrol_route()
                if route_file is not None:
                    route = self._load_patrol_waypoints_from_file(route_file)
                    self.envs.set_target_patrol_waypoints(route, route_name=route_file.stem)
                    if self.eval_envs is not None:
                        self.eval_envs.set_target_patrol_waypoints(route, route_name=route_file.stem)
            if self.use_linear_lr_decay:
                for group_name in self.group_order:
                    self.trainers[group_name].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                obs, rewards, dones, infos = self.envs.step(actions_env)
                self.insert((obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic))

            self.compute()
            train_infos = self.train()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                avg_rewards = []
                for group_name in self.group_order:
                    group_avg = np.mean(self.buffers[group_name].rewards) * self.episode_length
                    avg_rewards.append(group_avg)
                    train_infos.setdefault(group_name, {})
                    train_infos[group_name]["average_episode_rewards"] = float(group_avg)
                train_infos["system"] = {"average_episode_rewards": float(np.mean(avg_rewards))}
                self.log_train(train_infos, total_num_steps)
                self.record_train_metrics(total_num_steps, train_infos["system"]["average_episode_rewards"])
                self._maybe_report_best_metrics(total_num_steps, train_avg_reward=train_infos["system"]["average_episode_rewards"])

            if episode % self.gif_interval == 0:
                # print(f"episode {episode}, save GIF")
                self._save_training_gif(episode + 1)

            if episode % self.eval_interval == 0 and self.use_eval:
                val_summary = self.eval(total_num_steps, split="val", save_gif=False)
                if val_summary is not None:
                    if val_summary["reward"] >= (self._best_eval_avg_reward if self._best_eval_avg_reward is not None else -np.inf):
                        self._save_models_to_dir(self.best_model_dir / "best_reward")
                    if val_summary["success"] >= (self._best_capture_success_rate if self._best_capture_success_rate is not None else -np.inf):
                        self._save_models_to_dir(self.best_model_dir / "best_success")
                    if val_summary["steps"] is not None and (self._best_avg_capture_steps is None or val_summary["steps"] <= self._best_avg_capture_steps):
                        self._save_models_to_dir(self.best_model_dir / "best_steps")

        # 训练完成后，使用 test 集评估三类最优模型并保存每个场景 GIF。
        if self.use_eval:
            for tag in ["best_reward", "best_success", "best_steps"]:
                model_dir = self.best_model_dir / tag
                if model_dir.exists():
                    self._load_models_from_dir(model_dir)
                    self.eval(episodes * self.episode_length * self.n_rollout_threads, split="test", save_gif=True, model_tag=tag)

    @torch.no_grad()
    def _save_training_gif(self, episode_idx):
        frames = self._collect_episode_frames(episode_idx)
        gif_path = self.gif_dir / f"episode_{episode_idx:04d}.gif"
        imageio.mimsave(str(gif_path), frames, duration=self.gif_frame_duration)

    @torch.no_grad()
    def _collect_episode_frames(self, episode_idx, scenario=None, mode=None):
        env = MultiUavPursuitEnv(
            num_hunters=self.all_args.num_hunters,
            num_blockers=self.all_args.num_blockers,
            world_size=self.all_args.world_size,
            dt=self.all_args.dt,
            capture_radius=self.all_args.capture_radius,
            capture_steps=self.all_args.capture_steps,
            max_steps=self.episode_length,
            seed=self.all_args.seed,
            target_policy_source=self.all_args.target_policy_source,
            target_patrol_path=self.all_args.target_patrol_path,
            target_patrol_names=self.all_args.target_patrol_names,
            max_speed_hunter=self.all_args.max_speed_hunter,
            max_speed_blocker=self.all_args.max_speed_blocker,
            max_speed_target=self.all_args.max_speed_target,
            perception_hunter=self.all_args.perception_hunter,
            perception_blocker=self.all_args.perception_blocker,
            perception_target=self.all_args.perception_target,
        )
        if scenario is not None:
            # 评估 GIF 按场景参数重建环境状态。
            env.apply_scenario_config(scenario)
        if mode is not None:
            env.target_policy_source = mode
        if scenario is not None and mode == "patrol":
            route_file = self._resolve_route_file(scenario)
            route = self._load_patrol_waypoints_from_file(route_file) if route_file else None
            if route is not None:
                env.set_target_patrol_waypoints(route, route_name=f"route_{scenario.get('target_patrol_route_id')}")
        obs = env.reset()

        role_groups = env.role_groups
        hunter_indices = role_groups.get("hunter", [])
        blocker_indices = role_groups.get("blocker", [])
        target_index = role_groups.get("target", [None])[0]
        positions = {idx: [env.positions[idx].copy()] for idx in range(env.agent_num)}
        capture = False

        eval_rnn_states = {
            group_name: np.zeros((1, len(agent_ids), self.recurrent_N, self.hidden_size), dtype=np.float32)
            for group_name, agent_ids in self.policy_groups.items()
        }
        eval_masks = {
            group_name: np.ones((1, len(agent_ids), 1), dtype=np.float32)
            for group_name, agent_ids in self.policy_groups.items()
        }

        target_true = env.positions[target_index].copy() if target_index is not None else None
        target_obs = env.pursuit_obs_target_pos.copy() if env.pursuit_obs_target_pos is not None else None
        target_err = None
        if target_true is not None and target_obs is not None:
            target_err = float(np.linalg.norm(target_true - target_obs))
        frames = [self._draw_frame(positions, env.world_size, env.perception_ranges, hunter_indices, blocker_indices, target_index, capture, episode_idx, step=0, target_true_pos=target_true, target_obs_pos=target_obs, target_obs_err=target_err)]

        for step in range(1, self.episode_length + 1):
            actions_env = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)
            for group_name, agent_ids in self.policy_groups.items():
                trainer = self.trainers[group_name]
                trainer.prep_rollout()
                action, next_rnn = trainer.policy.act(
                    obs[agent_ids],
                    eval_rnn_states[group_name][0],
                    eval_masks[group_name][0],
                    deterministic=True,
                )
                actions_env[agent_ids] = _t2n(action)
                eval_rnn_states[group_name][0] = _t2n(next_rnn)

            obs, rewards, dones, infos = env.step(actions_env)
            capture = capture or any(info.get("capture", False) for info in infos)
            for idx in range(env.agent_num):
                positions[idx].append(env.positions[idx].copy())

            target_true = env.positions[target_index].copy() if target_index is not None else None
            target_obs = env.pursuit_obs_target_pos.copy() if env.pursuit_obs_target_pos is not None else None
            target_err = None
            if target_true is not None and target_obs is not None:
                target_err = float(np.linalg.norm(target_true - target_obs))
            frames.append(self._draw_frame(positions, env.world_size, env.perception_ranges, hunter_indices, blocker_indices, target_index, capture, episode_idx, step=step, target_true_pos=target_true, target_obs_pos=target_obs, target_obs_err=target_err))

            for group_name, agent_ids in self.policy_groups.items():
                group_dones = dones[agent_ids]
                eval_rnn_states[group_name][0][group_dones] = 0.0
                eval_masks[group_name][0] = np.ones((len(agent_ids), 1), dtype=np.float32)
                eval_masks[group_name][0][group_dones] = 0.0

            if np.all(dones):
                break

        env.close()
        return frames

    def _build_target_eval_policy(self, model_path):
        # 懒加载外部 Target policy（仅评估使用），避免重复加载模型文件。
        if not model_path:
            return None
        model_path = str(Path(model_path).resolve())
        if model_path in self._target_eval_policy_cache:
            return self._target_eval_policy_cache[model_path]

        from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        target_idx = self.num_agents - 1
        share_observation_space = self.eval_envs.share_observation_space[target_idx] if self.use_centralized_V else self.eval_envs.observation_space[target_idx]
        target_policy = Policy(
            self.all_args,
            self.eval_envs.observation_space[target_idx],
            share_observation_space,
            self.eval_envs.action_space[target_idx],
            device=self.device,
        )
        actor_state_dict = torch.load(model_path, map_location=self.device)
        target_policy.actor.load_state_dict(actor_state_dict)
        target_policy.actor.eval()
        self._target_eval_policy_cache[model_path] = target_policy
        return target_policy

    def _resolve_target_model_path(self, scenario):
        # 支持绝对路径、相对场景文件路径、以及目录（自动查找 actor_target.pt）。
        raw_path = scenario.get("target_policy_model_path")
        if not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute() and scenario.get("scenario_file"):
            path = Path(scenario["scenario_file"]).resolve().parent / path
        path = path.resolve()
        if path.is_dir():
            candidate = path / "actor_target.pt"
            if candidate.exists():
                return str(candidate)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Target policy model not found: {path}")

    @torch.no_grad()
    def eval(self, total_num_steps, split="val", save_gif=True, model_tag="latest"):
        # 评估按场景逐个执行；target_policy=train 时默认同时评估 patrol 与 train。
        suite = self.scenario_suite if split == "val" else self.test_suite
        if not suite:
            suite = [{
                "scenario_id": f"{split}_default",
                "num_hunters": self.all_args.num_hunters,
                "num_blockers": self.all_args.num_blockers,
                "world_size": self.all_args.world_size,
                "dt": self.all_args.dt,
                "capture_radius": self.all_args.capture_radius,
                "capture_steps": self.all_args.capture_steps,
                "episode_length": self.episode_length,
                "seed": self.all_args.seed,
                "initial_positions": None,
                "target_patrol_route_id": None,
            }]

        summary_rewards, summary_success, summary_steps = [], [], []

        for scenario_idx, scenario in enumerate(suite):
            scenario_id = str(scenario.get("scenario_id", f"{split}_{scenario_idx}"))
            scenario_cfg = dict(scenario)
            self.eval_envs.apply_scenario_config(scenario_cfg)

            # train target 时同时测 patrol/train，否则仅测 patrol。
            if self.target_policy_source == "train":
                eval_modes = scenario.get("eval_target_modes", ["patrol", "train"])
            else:
                eval_modes = ["patrol"]

            for mode in eval_modes:
                eval_episodes = int(self.all_args.eval_episodes)
                scenario_cfg = dict(scenario)
                scenario_cfg["target_policy_source"] = mode
                self.eval_envs.apply_scenario_config(scenario_cfg)
                eval_episode_rewards, capture_flags, capture_steps = [], [], []

                route_file = self._resolve_route_file(scenario)
                route = self._load_patrol_waypoints_from_file(route_file) if route_file else None
                if mode == "patrol" and route is not None:
                    self.eval_envs.set_target_patrol_waypoints(route, route_name=f"route_{scenario.get('target_patrol_route_id')}")

                for _ in range(eval_episodes):
                    eval_obs = self.eval_envs.reset()
                    eval_rnn_states = {
                        group_name: np.zeros((self.n_eval_rollout_threads, len(agent_ids), self.recurrent_N, self.hidden_size), dtype=np.float32)
                        for group_name, agent_ids in self.policy_groups.items()
                    }
                    eval_masks = {
                        group_name: np.ones((self.n_eval_rollout_threads, len(agent_ids), 1), dtype=np.float32)
                        for group_name, agent_ids in self.policy_groups.items()
                    }

                    episode_done = np.zeros(self.n_eval_rollout_threads, dtype=bool)
                    episode_steps = np.zeros(self.n_eval_rollout_threads, dtype=np.int32)
                    episode_capture = np.zeros(self.n_eval_rollout_threads, dtype=bool)
                    episode_capture_step = np.full(self.n_eval_rollout_threads, -1, dtype=np.int32)
                    ep_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32)

                    for _ in range(int(scenario.get("episode_length", self.episode_length))):
                        eval_actions_env = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.eval_envs.action_space[0].shape[0]), dtype=np.float32)
                        for group_name, agent_ids in self.policy_groups.items():
                            trainer = self.trainers[group_name]
                            trainer.prep_rollout()
                            eval_action, group_rnn = trainer.policy.act(
                                np.concatenate(eval_obs[:, agent_ids]),
                                np.concatenate(eval_rnn_states[group_name]),
                                np.concatenate(eval_masks[group_name]),
                                deterministic=True,
                            )
                            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
                            eval_rnn_states[group_name] = np.array(np.split(_t2n(group_rnn), self.n_eval_rollout_threads))
                            eval_actions_env[:, agent_ids, :] = eval_actions

                        # patrol 模式下强制 target 动作为巡逻轨迹；train 模式保持网络控制。
                        if mode == "patrol":
                            pass

                        eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                        ep_rewards += eval_rewards

                        for env_i in range(self.n_eval_rollout_threads):
                            if not episode_done[env_i]:
                                episode_steps[env_i] += 1
                                capture = any(info.get("capture", False) for info in eval_infos[env_i])
                                if capture and not episode_capture[env_i]:
                                    episode_capture[env_i] = True
                                    episode_capture_step[env_i] = episode_steps[env_i]
                                if np.all(eval_dones[env_i]):
                                    episode_done[env_i] = True

                        for group_name, agent_ids in self.policy_groups.items():
                            group_dones = eval_dones[:, agent_ids]
                            eval_rnn_states[group_name][group_dones] = 0.0
                            eval_masks[group_name] = np.ones((self.n_eval_rollout_threads, len(agent_ids), 1), dtype=np.float32)
                            eval_masks[group_name][group_dones] = 0.0

                        if np.all(episode_done):
                            break

                    eval_episode_rewards.append(ep_rewards)
                    capture_flags.extend(episode_capture.tolist())
                    capture_steps.extend([int(cs) for cs in episode_capture_step if cs > 0])

                eval_avg_rewards = float(np.mean([np.mean(ep) for ep in eval_episode_rewards])) if eval_episode_rewards else 0.0
                total_episodes = max(1, eval_episodes * self.n_eval_rollout_threads)
                capture_success_rate = float(np.sum(capture_flags) / total_episodes) if capture_flags else 0.0
                avg_capture_steps = float(np.mean(capture_steps)) if capture_steps else None

                scenario_metric_id = f"{split}/{scenario_id}/{mode}/{model_tag}"
                eval_env_infos = {
                    f"eval/{scenario_metric_id}/average_episode_rewards": eval_avg_rewards,
                    f"eval/{scenario_metric_id}/capture_success_rate": capture_success_rate,
                }
                if avg_capture_steps is not None:
                    eval_env_infos[f"eval/{scenario_metric_id}/avg_capture_steps"] = avg_capture_steps
                self.log_env(eval_env_infos, total_num_steps)
                self.record_eval_metrics(total_num_steps, eval_avg_rewards, capture_success_rate, avg_capture_steps, scenario_id=scenario_metric_id)
                self._maybe_report_best_metrics(total_num_steps, eval_avg_reward=eval_avg_rewards, capture_success_rate=capture_success_rate, avg_capture_steps=avg_capture_steps)

                summary_rewards.append(eval_avg_rewards)
                summary_success.append(capture_success_rate)
                if avg_capture_steps is not None:
                    summary_steps.append(avg_capture_steps)

                if save_gif:
                    frames = self._collect_episode_frames(scenario_idx, scenario=scenario, mode=mode)
                    gif_dir = self.eval_gif_dir / f"{split}_{scenario_idx}"
                    gif_dir.mkdir(parents=True, exist_ok=True)
                    gif_path = gif_dir / f"{scenario_id}_{mode}_{model_tag}.gif"
                    imageio.mimsave(str(gif_path), frames, duration=self.gif_frame_duration)

        if not summary_rewards:
            return None
        return {
            "reward": float(np.mean(summary_rewards)),
            "success": float(np.mean(summary_success)),
            "steps": float(np.mean(summary_steps)) if summary_steps else None,
        }

    def _draw_fade_traj(self, ax, traj, color):
        if len(traj) < 2:
            return
        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        alpha = np.linspace(0.15, 0.9, len(segments))
        rgba = np.tile(plt.matplotlib.colors.to_rgba(color), (len(segments), 1))
        rgba[:, 3] = alpha
        lc = LineCollection(segments, colors=rgba, linewidths=2.0)
        ax.add_collection(lc)

    def _draw_frame(self, positions, world_size, perception_ranges, hunter_indices, blocker_indices, target_index, capture, episode_idx, step, target_true_pos=None, target_obs_pos=None, target_obs_err=None):
        fig, ax = plt.subplots(figsize=(6.8, 6.8), dpi=140)
        ax.set_xlim(-world_size, world_size)
        ax.set_ylim(-world_size, world_size)
        ax.set_aspect("equal")
        ax.set_title(f"Episode {episode_idx} - Step {step}")
        ax.grid(True, linestyle="--", alpha=0.25)

        palette = {"hunter": "#1f77b4", "blocker": "#2ca02c", "target": "#d62728"}

        for i, idx in enumerate(hunter_indices):
            traj = np.array(positions[idx])
            self._draw_fade_traj(ax, traj, palette["hunter"])
            ax.scatter(traj[-1, 0], traj[-1, 1], color=palette["hunter"], s=45)
            pr = plt.Circle((traj[-1, 0], traj[-1, 1]), perception_ranges["hunter"], color=palette["hunter"], alpha=0.08)
            ax.add_patch(pr)
            ax.text(traj[-1, 0], traj[-1, 1], f"H{i+1}", fontsize=8)

        for i, idx in enumerate(blocker_indices):
            traj = np.array(positions[idx])
            self._draw_fade_traj(ax, traj, palette["blocker"])
            ax.scatter(traj[-1, 0], traj[-1, 1], color=palette["blocker"], marker="s", s=42)
            pr = plt.Circle((traj[-1, 0], traj[-1, 1]), perception_ranges["blocker"], color=palette["blocker"], alpha=0.07)
            ax.add_patch(pr)
            ax.text(traj[-1, 0], traj[-1, 1], f"B{i+1}", fontsize=8)

        if target_index is not None:
            traj = np.array(positions[target_index])
            self._draw_fade_traj(ax, traj, palette["target"])
            ax.scatter(traj[-1, 0], traj[-1, 1], color=palette["target"], marker="*", s=95)
            pr = plt.Circle((traj[-1, 0], traj[-1, 1]), perception_ranges["target"], color=palette["target"], alpha=0.08)
            ax.add_patch(pr)
            ax.text(traj[-1, 0], traj[-1, 1], "Target", fontsize=8)
            if target_obs_pos is not None:
                ax.scatter(target_obs_pos[0], target_obs_pos[1], color="#ff7f0e", marker="x", s=55)
                ax.text(target_obs_pos[0], target_obs_pos[1], "Obs", fontsize=8)
                if target_obs_err is not None and target_obs_err > 0.0:
                    err_circle = plt.Circle((target_obs_pos[0], target_obs_pos[1]), target_obs_err, color="#ff7f0e", alpha=0.12, linestyle="--", fill=True)
                    ax.add_patch(err_circle)

        status = "Captured" if capture else "Not captured"
        ax.text(0.02, 0.98, f"Status: {status}", transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.canvas.get_width_height()
        try:
            buf = canvas.tostring_rgb()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        except AttributeError:
            buf = canvas.tostring_argb()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
            image = image[:, :, [1, 2, 3]]
        plt.close(fig)
        return image
