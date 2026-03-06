"""
Role-based Runner for Multi-UAV Pursuit.

核心特性:
1) 读取分层配置（merged_cfg），不依赖全局扁平化参数。
2) 仅在初始化算法组件（Policy/Trainer/Buffer）时构建flat args。
3) 同角色共享策略：当前hunter共享一个policy；target可选是否训练。
4) 训练中周期性执行evaluation，并基于环境render接口保存train/eval GIF。
"""

import os
import time
from pathlib import Path
from itertools import chain
import argparse
import csv
import shutil
import copy
import json

import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from utils.separated_buffer import SeparatedReplayBuffer


def _t2n(x):
    """
    功能:
        将Torch张量转为Numpy数组。
    输入:
        x (torch.Tensor): 任意形状张量。
    输出:
        np.ndarray: 转到CPU并detach后的numpy数组。
    """
    return x.detach().cpu().numpy()


class RoleBasedRunner(object):
    """
    UAV Pursuit任务专用训练执行器。
    """

    def __init__(self, runner_cfg, merged_cfg):
        """
        功能:
            初始化UAV角色化Runner，构建策略、训练器、缓存与日志路径。
        输入:
            runner_cfg (dict): 外部运行配置，包含envs/eval_envs/device/run_dir/num_agents。
            merged_cfg (EasyDict): 分层配置对象。
        输出:
            无。
        """
        # Step 1: 保存外部上下文
        self.cfg = merged_cfg
        self.envs = runner_cfg["envs"]
        self.eval_envs = runner_cfg["eval_envs"]
        self.eval_envs_target_learn = runner_cfg["eval_envs_target_learn"] if "eval_envs_target_learn" in runner_cfg else None
        self.device = runner_cfg["device"]
        self.run_dir = runner_cfg["run_dir"]
        self.num_agents = int(runner_cfg["num_agents"])

        # Step 2: 读取训练主参数
        self.train_max_hunters_num = int(
            runner_cfg.get("train_max_hunters_num", int(self.cfg.env.max_hunters_num))
        )
        self.eval_max_hunters_num = int(
            runner_cfg.get("eval_max_hunters_num", self.train_max_hunters_num)
        )
        self.target_index = self.num_agents - 1
        self.target_trainable = str(self.cfg.env.target_policy_source).lower() == "learn"

        self.episode_length = int(self.cfg.env.episode_length)
        self.eval_episode_length = int(self.cfg.eval.eval_episode_length)
        
        self.n_rollout_threads = int(self.cfg.exp.n_rollout_threads)
        self.n_eval_rollout_threads = int(self.cfg.exp.n_eval_rollout_threads)
        self.num_env_steps = int(self.cfg.exp.num_env_steps)
        self.use_eval = bool(self.cfg.eval.use_eval)
        self.eval_interval = int(self.cfg.eval.val_interval)
        self.save_interval = int(self.cfg.logging.save_interval)
        self.log_interval = int(self.cfg.logging.log_interval)
        self.use_linear_lr_decay = bool(self.cfg.schedule.use_linear_lr_decay)
        self.hidden_size = int(self.cfg.model.hidden_size)
        self.recurrent_N = int(self.cfg.model.recurrent_N)
        self.use_centralized_V = bool(self.cfg.model.use_centralized_V)
        self.algorithm_name = str(self.cfg.exp.algorithm_name)
        self.experiment_name = str(self.cfg.exp.experiment_name)
        self.env_name = str(self.cfg.env.env_name)
        self.role_buffer_mode = str(getattr(self.cfg.exp, "role_buffer_mode", "separate")).lower()
        if self.role_buffer_mode not in ("separate", "role_shared"):
            raise ValueError(
                "Unsupported exp.role_buffer_mode: {} (choices: separate, role_shared)".format(
                    str(self.role_buffer_mode)
                )
            )

        # Step 3: 创建日志/模型目录
        self.log_dir = str(self.run_dir / "logs")
        self.save_dir = str(self.run_dir / "models")
        self.gif_dir = str(self.run_dir / "gifs")
        self.best_dir = str(self.save_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.gif_dir, exist_ok=True)
        os.makedirs(self.best_dir, exist_ok=True)
        self.writter = SummaryWriter(self.log_dir)

        self.log_csv_path = str(self.run_dir / "log.csv")
        self.eval_csv_path = str(self.run_dir / "eval.csv")
        self.init_csv_files = bool(runner_cfg.get("init_csv", True))
        if self.init_csv_files:
            self._init_csv_files()
        self.gif_frame_interval = max(1, int(self.cfg.logging.gif_frame_interval))
        self.gif_frame_duration = float(self.cfg.logging.gif_frame_duration)
        self.log_gif = bool(getattr(self.cfg.logging, "log_gif", True))
        self.time_stat = bool(runner_cfg.get("time_stat", False))
        self.gif_start_step = int(self.cfg.render.gif_start_step)
        # 训练阶段默认仅绘制第0个rollout环境，降低开销；评估仍绘制全部eval环境。
        self.train_gif_env_ids = [0]

        self.pending_train_gif_once = False
        self.best_eval_metrics = {
            "reward": -np.inf,
            "capture_rate": -np.inf,
            "capture_steps": np.inf,
            "max_escape_gap_angle": np.inf,
        }
        self.best_eval_metrics_by_bucket = {
            "target_learn": {
                "eval_reward": -np.inf,
                "capture_rate": -np.inf,
                "capture_steps": np.inf,
                "max_escape_gap_angle": np.inf,
            }
        }
        self.eval_hunter_bucket_metrics_cache = {}
        print(
            "[DomainRand] train.enable={}, interval={}, prob={}, hunter_choices={}, seed_range={}, target_policies={}, patrol_pool={}".format(
                bool(self.cfg.domain_randomization.train_split.enable),
                int(self.cfg.domain_randomization.train_split.regen_interval_episode),
                float(self.cfg.domain_randomization.train_split.regen_prob),
                list(self.cfg.domain_randomization.train_split.hunter_count_choices),
                list(self.cfg.domain_randomization.train_split.seed_range),
                list(self.cfg.domain_randomization.train_split.target_policy_choices),
                list(self.cfg.domain_randomization.train_split.patrol_name_choices),
            )
        )
        print(
            "[EvalConfig] fixed_task_source={}, inline_fixed_tasks={}".format(
                "inline" if self.cfg.eval.fixed_tasks_file is None else str(self.cfg.eval.fixed_tasks_file),
                int(len(self.cfg.eval.fixed_tasks)),
            )
        )
        print("[GIFConfig] log_gif(train_stage)={}".format(bool(self.log_gif)))
        print("[BufferMode] role_buffer_mode={}".format(str(self.role_buffer_mode)))

        # Step 4: 仅在算法组件初始化时构建flat args
        self.flat_args = self._build_flat_args_for_algorithm()

        # Step 5: 导入MAPPO算法组件
        from algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
        from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        # Step 6: 角色定义与受控智能体集合
        self.agent_role = {aid: "hunter" for aid in range(self.train_max_hunters_num)}
        self.agent_role[self.target_index] = "target"
        self.controlled_agents = list(range(self.train_max_hunters_num))
        if self.target_trainable:
            self.controlled_agents.append(self.target_index)
        self.role_to_agents = {"hunter": list(range(self.train_max_hunters_num))}
        if self.target_trainable:
            self.role_to_agents["target"] = [self.target_index]

        # Step 7: 构建角色级别policy与trainer
        self.role_policies = {}
        self.role_trainers = {}
        role_represent_agent = {"hunter": 0}
        if self.target_trainable:
            print("Training Target Policy ...")
            role_represent_agent["target"] = self.target_index

        for role_name, rep_agent_id in role_represent_agent.items():
            if self.use_centralized_V:
                share_obs_space = self.envs.share_observation_space[rep_agent_id]
            else:
                share_obs_space = self.envs.observation_space[rep_agent_id]

            policy = Policy(
                self.flat_args,
                self.envs.observation_space[rep_agent_id],
                share_obs_space,
                self.envs.action_space[rep_agent_id],
                device=self.device,
            )
            trainer = TrainAlgo(self.flat_args, policy, device=self.device)
            self.role_policies[role_name] = policy
            self.role_trainers[role_name] = trainer

        # Step 8: 构建智能体级别buffer
        self.buffers = {}
        for agent_id in self.controlled_agents:
            if self.use_centralized_V:
                share_obs_space = self.envs.share_observation_space[agent_id]
            else:
                share_obs_space = self.envs.observation_space[agent_id]
            self.buffers[agent_id] = SeparatedReplayBuffer(
                self.flat_args,
                self.envs.observation_space[agent_id],
                share_obs_space,
                self.envs.action_space[agent_id],
            )

        # Step 9: 可选角色共享训练buffer（仅用于train阶段聚合更新）
        self.role_shared_train_buffers = {}
        if self.role_buffer_mode == "role_shared":
            for role_name, agent_ids in self.role_to_agents.items():
                if len(agent_ids) == 0:
                    continue
                rep_agent_id = int(agent_ids[0])
                flat_args_role = copy.deepcopy(self.flat_args)
                flat_args_role.n_rollout_threads = int(self.n_rollout_threads) * int(len(agent_ids))
                if self.use_centralized_V:
                    share_obs_space = self.envs.share_observation_space[rep_agent_id]
                else:
                    share_obs_space = self.envs.observation_space[rep_agent_id]
                self.role_shared_train_buffers[role_name] = SeparatedReplayBuffer(
                    flat_args_role,
                    self.envs.observation_space[rep_agent_id],
                    share_obs_space,
                    self.envs.action_space[rep_agent_id],
                )

    def _build_flat_args_for_algorithm(self):
        """
        功能:
            将分层配置映射为算法组件需要的扁平参数命名空间。
        输入:
            无（读取self.cfg）。
        输出:
            argparse.Namespace: 算法与buffer初始化参数。
        """
        # Step 1: 创建空命名空间
        args = argparse.Namespace()
        train_max_hunters_num = int(
            getattr(self, "train_max_hunters_num", int(self.cfg.env.max_hunters_num))
        )
        eval_episode_length = int(
            getattr(
                self,
                "eval_episode_length",
                max(
                    1,
                    int(
                        getattr(
                            self.cfg.eval,
                            "eval_episode_length",
                            self.cfg.env.episode_length,
                        )
                    ),
                ),
            )
        )

        # Step 2: 写入实验与环境参数
        args.algorithm_name = self.cfg.exp.algorithm_name
        args.experiment_name = self.cfg.exp.experiment_name
        args.seed = self.cfg.exp.seed
        args.cuda = self.cfg.exp.cuda
        args.cuda_deterministic = self.cfg.exp.cuda_deterministic
        args.n_training_threads = self.cfg.exp.n_training_threads
        args.n_rollout_threads = self.cfg.exp.n_rollout_threads
        args.n_eval_rollout_threads = self.cfg.exp.n_eval_rollout_threads
        args.n_render_rollout_threads = self.cfg.exp.n_render_rollout_threads
        args.num_env_steps = self.cfg.exp.num_env_steps

        args.env_name = self.cfg.env.env_name
        args.episode_length = self.cfg.env.episode_length
        args.num_agents = int(train_max_hunters_num) + int(self.cfg.env.num_explorers) + 1
        args.use_obs_instead_of_state = False

        # Step 3: 写入模型参数
        args.share_policy = self.cfg.model.policy_share
        args.use_centralized_V = self.cfg.model.use_centralized_V
        args.stacked_frames = self.cfg.model.stacked_frames
        args.use_stacked_frames = self.cfg.model.use_stacked_frames
        args.hidden_size = self.cfg.model.hidden_size
        args.layer_N = self.cfg.model.layer_N
        args.use_ReLU = self.cfg.model.use_ReLU
        args.use_popart = self.cfg.model.use_popart
        args.use_valuenorm = self.cfg.model.use_valuenorm
        args.use_feature_normalization = self.cfg.model.use_feature_normalization
        args.use_orthogonal = self.cfg.model.use_orthogonal
        args.gain = self.cfg.model.gain
        args.use_naive_recurrent_policy = self.cfg.model.use_naive_recurrent_policy
        args.use_recurrent_policy = self.cfg.model.use_recurrent_policy
        args.recurrent_N = self.cfg.model.recurrent_N
        args.data_chunk_length = self.cfg.model.data_chunk_length

        # Step 4: 写入优化器与PPO参数
        args.lr = self.cfg.optim.lr
        args.critic_lr = self.cfg.optim.critic_lr
        args.opti_eps = self.cfg.optim.opti_eps
        args.weight_decay = self.cfg.optim.weight_decay

        args.ppo_epoch = self.cfg.ppo.ppo_epoch
        args.use_clipped_value_loss = self.cfg.ppo.use_clipped_value_loss
        args.clip_param = self.cfg.ppo.clip_param
        args.num_mini_batch = self.cfg.ppo.num_mini_batch
        args.entropy_coef = self.cfg.ppo.entropy_coef
        args.value_loss_coef = self.cfg.ppo.value_loss_coef
        args.use_max_grad_norm = self.cfg.ppo.use_max_grad_norm
        args.max_grad_norm = self.cfg.ppo.max_grad_norm
        args.use_gae = self.cfg.ppo.use_gae
        args.gamma = self.cfg.ppo.gamma
        args.gae_lambda = self.cfg.ppo.gae_lambda
        args.use_proper_time_limits = self.cfg.ppo.use_proper_time_limits
        args.use_huber_loss = self.cfg.ppo.use_huber_loss
        args.use_value_active_masks = self.cfg.ppo.use_value_active_masks
        args.use_policy_active_masks = self.cfg.ppo.use_policy_active_masks
        args.huber_delta = self.cfg.ppo.huber_delta

        # Step 5: 写入调度/日志/评估/渲染参数
        args.use_linear_lr_decay = self.cfg.schedule.use_linear_lr_decay
        args.save_interval = self.cfg.logging.save_interval
        args.log_interval = self.cfg.logging.log_interval
        args.use_eval = self.cfg.eval.use_eval
        args.eval_interval = self.cfg.eval.val_interval
        args.eval_episode_length = eval_episode_length
        args.save_gifs = self.cfg.render.save_gifs
        args.use_render = self.cfg.render.use_render
        args.render_episodes = self.cfg.render.render_episodes
        args.ifi = self.cfg.render.ifi
        args.model_dir = self.cfg.pretrained.model_dir
        return args

    def run(self):
        """
        功能:
            执行主训练循环，按间隔保存模型、记录日志并执行evaluation。
        输入:
            无。
        输出:
            无。
        """
        # Step 1: 预填充buffer初始观测
        self.warmup()

        # Step 2: 计算总训练episode并开始计时
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        print(
            "[TrainStart] env={}, algo={}, exp={}, episodes={}, episode_len={}, rollout_threads={}, total_steps={}".format(
                self.env_name,
                self.algorithm_name,
                self.experiment_name,
                int(episodes),
                int(self.episode_length),
                int(self.n_rollout_threads),
                int(self.num_env_steps),
            )
        )
        print(
            "[TrainStart] eval={}, eval_interval={}, eval_episode_len={}, save_interval={}, log_interval={}, log_csv={}, eval_csv={}".format(
                bool(self.use_eval),
                int(self.eval_interval),
                int(self.eval_episode_length),
                int(self.save_interval),
                int(self.log_interval),
                self.log_csv_path,
                self.eval_csv_path,
            )
        )

        # Step 3: episode主循环
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for role_name in self.role_trainers.keys():
                    self.role_trainers[role_name].policy.lr_decay(episode, episodes)

            # 是否需要进行eval，需要的话根据log_gif参数进行GIF录制
            do_eval_this_episode = self.use_eval and episode % max(1, self.eval_interval) == 0

            do_train_gif_this_episode = bool(do_eval_this_episode and self.log_gif)
            if do_train_gif_this_episode:
                self._print_timed(
                    "[TrainGIF] episode {} save once (env ids: {})".format(
                        int(episode),
                        ",".join([str(int(x)) for x in self.train_gif_env_ids]),
                    )
                )

            if do_eval_this_episode:
                self._print_timed("[Train] episode {} trigger eval".format(int(episode)))

            if hasattr(self.envs, "capture_terminal_frame"):
                self.envs.capture_terminal_frame = bool(do_train_gif_this_episode)

            train_frames = [[] for _ in range(self.n_rollout_threads)]
            train_gif_finished = np.zeros(self.n_rollout_threads, dtype=bool)
            train_capture_step = np.full(self.n_rollout_threads, -1, dtype=np.int32)
            train_alive_rate = np.full(self.n_rollout_threads, 0.0, dtype=np.float32)
            episode_hunter_reward = 0.0
            episode_target_reward = 0.0
            episode_active_hunter_slots = None
            last_infos = None

            for step in range(self.episode_length):
                # 训练GIF采样：仅在被触发的单个episode中记录指定env（默认env0）。
                if do_train_gif_this_episode and step % self.gif_frame_interval == 0:
                    for env_i in self.train_gif_env_ids:
                        if env_i < 0 or env_i >= self.n_rollout_threads:
                            continue
                        if train_gif_finished[env_i]:
                            continue
                        frame = self.envs.render(
                            mode="rgb_array",
                            env_id=int(env_i),
                            title=f"Train Episode {int(episode)}",
                        )
                        if isinstance(frame, np.ndarray):
                            train_frames[env_i].append(frame.copy())

                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Observe reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_hunter_reward += float(np.sum(rewards[:, : self.train_max_hunters_num, 0]))
                episode_target_reward += float(np.sum(rewards[:, self.target_index, 0]))

                # 本次episode阶段active的hunters数量
                if episode_active_hunter_slots is None:
                    active_cnt = 0
                    for env_infos in infos:
                        for hid in range(self.train_max_hunters_num):
                            if hid < len(env_infos) and bool(env_infos[hid].get("active_agent", True)):
                                active_cnt += 1
                    episode_active_hunter_slots = int(active_cnt)
                last_infos = infos

                if do_train_gif_this_episode:
                    for env_i in self.train_gif_env_ids:
                        if env_i < 0 or env_i >= self.n_rollout_threads:
                            continue
                        if train_gif_finished[env_i]:
                            continue
                        if bool(np.all(dones[env_i])):
                            env_infos = infos[env_i]
                            if any(bool(agent_info.get("captured", False)) for agent_info in env_infos):
                                train_capture_step[env_i] = int(step + 1)
                            hunter_alive_flags = [
                                float(agent_info.get("alive", False))
                                for agent_info in env_infos[: self.train_max_hunters_num]
                            ]
                            train_alive_rate[env_i] = (
                                float(np.mean(hunter_alive_flags)) if len(hunter_alive_flags) > 0 else 0.0
                            )
                            terminal_frame = self._extract_terminal_frame(env_infos)
                            if terminal_frame is not None:
                                train_frames[env_i].append(terminal_frame.copy())
                            train_gif_finished[env_i] = True

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            hunter_alive_mean, target_alive_mean = self._extract_alive_stats(last_infos)
            hunter_reward_denominator = (
                episode_active_hunter_slots
                if episode_active_hunter_slots is not None
                    else self.n_rollout_threads * self.train_max_hunters_num
            )
            self._append_log_csv(
                episode=episode,
                total_num_steps=total_num_steps,
                hunter_alive_mean=hunter_alive_mean,
                target_alive_mean=target_alive_mean,
                hunter_reward_mean=episode_hunter_reward
                / max(1, hunter_reward_denominator),
                target_reward_mean=episode_target_reward / max(1, self.n_rollout_threads),
            )

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()
                print("[Checkpoint] episode {} model saved to {}".format(int(episode), self.save_dir))

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                elapsed = float(end - start)
                fps = float(total_num_steps / (elapsed + 1e-6))  # step / sec
                progress = float((episode + 1) / max(1, episodes))
                eta_sec = max(0.0, (self.num_env_steps - total_num_steps) / max(fps, 1e-6))
                print(
                    "\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {} step/sec.\n".format(
                        self.env_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(fps),
                    )
                )
                self._print_timed(
                    "[Progress] episode={}/{}".format(int(episode), int(episodes))
                )
                self._print_metric_table(
                    "ProgressMetrics",
                    {
                        "progress_pct": "{:.1f}".format(100.0 * progress),
                        "elapsed": self._format_duration(elapsed),
                        "eta": self._format_duration(eta_sec),
                        "hunter_reward_mean": "{:.4f}".format(
                            episode_hunter_reward / max(1, hunter_reward_denominator)
                        ),
                        "target_reward_mean": "{:.4f}".format(
                            episode_target_reward / max(1, self.n_rollout_threads)
                        ),
                        "hunter_alive": "{:.4f}".format(hunter_alive_mean),
                        "target_alive": "{:.4f}".format(target_alive_mean),
                    },
                )
                self.log_train(train_infos, total_num_steps)

            # save one-shot train gif (triggered by last eval improvement)
            if do_train_gif_this_episode:
                for env_i in self.train_gif_env_ids:
                    if env_i >= self.n_rollout_threads:
                        continue
                    train_gif_path = Path(self.gif_dir) / f"train_{int(episode)}_env_{int(env_i)}.gif"
                    self._save_gif(
                        train_frames[env_i],
                        train_gif_path,
                        episode_id=int(episode),
                        capture_step=None if train_capture_step[env_i] <= 0 else int(train_capture_step[env_i]),
                        alive_rate=float(train_alive_rate[env_i]),
                    )
                self._print_timed(
                    "[GIF] saved train gifs for episode {} to {} ({} envs)".format(
                        int(episode), self.gif_dir, int(len(self.train_gif_env_ids))
                    )
                )

            # eval（单阶段：训练期间直接评估；若log_gif=True则仅保存前3个env的GIF）
            if do_eval_this_episode:
                fixed_metrics = self.eval(
                    total_num_steps,
                    episode,
                    eval_envs=self.eval_envs,
                    bucket="fixed",
                    save_gifs=bool(self.log_gif),
                    record_logs=True,
                    gif_env_limit=3,
                )
                self._maybe_save_best_models(episode, fixed_metrics)

                if self.eval_envs_target_learn is not None:
                    target_learn_metrics = self.eval(
                        total_num_steps,
                        episode,
                        eval_envs=self.eval_envs_target_learn,
                        bucket="target_learn",
                        save_gifs=bool(self.log_gif),
                        record_logs=True,
                        gif_env_limit=3,
                    )
                    self._update_bucket_best_metrics("target_learn", target_learn_metrics)
            if hasattr(self.envs, "capture_terminal_frame"):
                self.envs.capture_terminal_frame = False

        # Step 4: 训练完成后，重载所有最优模型并强制评估+保存GIF
        self._final_eval_saved_best_models(total_num_steps=int(self.num_env_steps), episode=int(episodes - 1))

    def run_time_stat(self):
        """
        功能:
            以低侵入方式统计训练耗时瓶颈，并复用run()原始流程执行训练。
        输入:
            无。
        输出:
            无（打印并写出每个episode的耗时统计到time_stat.csv）。
        """
        # Step 1: 初始化统计上下文与CSV
        self._init_time_stat_context()
        print("[TimeStat] enabled, running instrumented training via run()")

        # Step 2: 保存原始方法引用
        orig_collect = self.collect
        orig_env_step = self.envs.step
        orig_insert = self.insert
        orig_compute = self.compute
        orig_train = self.train
        orig_eval = self.eval
        orig_save_gif = self._save_gif
        orig_final_eval = self._final_eval_saved_best_models

        def _wrapped_collect(step):
            if int(step) == 0:
                self._start_time_stat_episode()
            if self._time_stat_ctx["current"] is not None:
                self._time_stat_ctx["current"]["step_clock_start"] = time.perf_counter()
            t0 = time.perf_counter()
            out = orig_collect(step)
            dt = time.perf_counter() - t0
            self._accumulate_time_stat("collect", dt)
            return out

        def _wrapped_env_step(actions_env):
            t0 = time.perf_counter()
            out = orig_env_step(actions_env)
            dt = time.perf_counter() - t0
            self._accumulate_time_stat("env_step", dt)
            return out

        def _wrapped_insert(data):
            t0 = time.perf_counter()
            out = orig_insert(data)
            dt = time.perf_counter() - t0
            self._accumulate_time_stat("insert", dt)
            current = self._time_stat_ctx["current"]
            if current is not None and current["step_clock_start"] is not None:
                step_dt = time.perf_counter() - float(current["step_clock_start"])
                current["step_total_sec"] += float(step_dt)
                current["step_calls"] += 1
                current["step_clock_start"] = None
            return out

        def _wrapped_compute():
            t0 = time.perf_counter()
            out = orig_compute()
            dt = time.perf_counter() - t0
            self._accumulate_time_stat("compute", dt, with_calls=False)
            return out

        def _wrapped_train():
            t0 = time.perf_counter()
            out = orig_train()
            dt = time.perf_counter() - t0
            self._accumulate_time_stat("train", dt, with_calls=False)
            return out

        def _wrapped_eval(*args, **kwargs):
            t0 = time.perf_counter()
            out = orig_eval(*args, **kwargs)
            dt = time.perf_counter() - t0
            self._accumulate_time_stat("eval", dt)
            return out

        def _wrapped_save_gif(*args, **kwargs):
            t0 = time.perf_counter()
            out = orig_save_gif(*args, **kwargs)
            dt = time.perf_counter() - t0
            self._accumulate_time_stat("save_gif", dt)
            return out

        def _wrapped_final_eval(*args, **kwargs):
            self._finalize_time_stat_episode()
            return orig_final_eval(*args, **kwargs)

        # Step 3: 注入包装方法并执行原run
        try:
            self.collect = _wrapped_collect
            self.envs.step = _wrapped_env_step
            self.insert = _wrapped_insert
            self.compute = _wrapped_compute
            self.train = _wrapped_train
            self.eval = _wrapped_eval
            self._save_gif = _wrapped_save_gif
            self._final_eval_saved_best_models = _wrapped_final_eval
            self.run()
        finally:
            # Step 4: 恢复原始方法，避免后续调用污染
            self.collect = orig_collect
            self.envs.step = orig_env_step
            self.insert = orig_insert
            self.compute = orig_compute
            self.train = orig_train
            self.eval = orig_eval
            self._save_gif = orig_save_gif
            self._final_eval_saved_best_models = orig_final_eval
            self._finalize_time_stat_episode()
            self._time_stat_ctx = None

    def _init_time_stat_context(self):
        """
        功能:
            初始化time_stat上下文与CSV文件表头。
        输入:
            无。
        输出:
            无。
        """
        csv_path = Path(self.run_dir) / "time_stat.csv"
        self._time_stat_ctx = {
            "csv_path": csv_path,
            "episode_idx": -1,
            "current": None,
        }
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "episode_total_sec",
                    "step_calls",
                    "step_avg_sec",
                    "collect_calls",
                    "collect_total_sec",
                    "collect_avg_sec",
                    "env_step_calls",
                    "env_step_total_sec",
                    "env_step_avg_sec",
                    "insert_calls",
                    "insert_total_sec",
                    "insert_avg_sec",
                    "compute_total_sec",
                    "train_total_sec",
                    "eval_calls",
                    "eval_total_sec",
                    "eval_avg_sec",
                    "save_gif_calls",
                    "save_gif_total_sec",
                    "save_gif_avg_sec",
                ]
            )

    def _start_time_stat_episode(self):
        """
        功能:
            在新episode开始时切换统计窗口（并先结算上一episode）。
        输入:
            无。
        输出:
            无。
        """
        self._finalize_time_stat_episode()
        self._time_stat_ctx["episode_idx"] += 1
        self._time_stat_ctx["current"] = {
            "episode": int(self._time_stat_ctx["episode_idx"]),
            "episode_clock_start": time.perf_counter(),
            "step_clock_start": None,
            "step_total_sec": 0.0,
            "step_calls": 0,
            "collect_total_sec": 0.0,
            "collect_calls": 0,
            "env_step_total_sec": 0.0,
            "env_step_calls": 0,
            "insert_total_sec": 0.0,
            "insert_calls": 0,
            "compute_total_sec": 0.0,
            "train_total_sec": 0.0,
            "eval_total_sec": 0.0,
            "eval_calls": 0,
            "save_gif_total_sec": 0.0,
            "save_gif_calls": 0,
        }

    def _accumulate_time_stat(self, key, dt, with_calls=True):
        """
        功能:
            将单次计时累加到当前episode统计项。
        输入:
            key (str): 统计项前缀（collect/env_step/insert/eval/save_gif/compute/train）。
            dt (float): 本次耗时（秒）。
            with_calls (bool): 是否累计调用次数计数器。
        输出:
            无。
        """
        if self._time_stat_ctx is None:
            return
        current = self._time_stat_ctx.get("current", None)
        if current is None:
            return
        total_key = f"{key}_total_sec"
        call_key = f"{key}_calls"
        if total_key in current:
            current[total_key] += float(dt)
        if with_calls and call_key in current:
            current[call_key] += 1

    def _finalize_time_stat_episode(self):
        """
        功能:
            结束并输出当前episode的耗时统计（打印+CSV）。
        输入:
            无。
        输出:
            无。
        """
        if self._time_stat_ctx is None:
            return
        current = self._time_stat_ctx.get("current", None)
        if current is None:
            return

        episode_total_sec = time.perf_counter() - float(current["episode_clock_start"])
        step_avg = float(current["step_total_sec"]) / max(1, int(current["step_calls"]))
        collect_avg = float(current["collect_total_sec"]) / max(1, int(current["collect_calls"]))
        env_step_avg = float(current["env_step_total_sec"]) / max(1, int(current["env_step_calls"]))
        insert_avg = float(current["insert_total_sec"]) / max(1, int(current["insert_calls"]))
        eval_avg = float(current["eval_total_sec"]) / max(1, int(current["eval_calls"]))
        save_gif_avg = float(current["save_gif_total_sec"]) / max(1, int(current["save_gif_calls"]))

        print(
            "[TimeStat][Ep {}] ep={:.3f}s | step(avg)={:.4f}s | collect(avg)={:.4f}s | env.step(avg)={:.4f}s | insert(avg)={:.4f}s | compute={:.3f}s | train={:.3f}s | eval(avg)={:.3f}s x{} | save_gif(avg)={:.3f}s x{}".format(
                int(current["episode"]),
                float(episode_total_sec),
                float(step_avg),
                float(collect_avg),
                float(env_step_avg),
                float(insert_avg),
                float(current["compute_total_sec"]),
                float(current["train_total_sec"]),
                float(eval_avg),
                int(current["eval_calls"]),
                float(save_gif_avg),
                int(current["save_gif_calls"]),
            )
        )

        with open(self._time_stat_ctx["csv_path"], "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(current["episode"]),
                    float(episode_total_sec),
                    int(current["step_calls"]),
                    float(step_avg),
                    int(current["collect_calls"]),
                    float(current["collect_total_sec"]),
                    float(collect_avg),
                    int(current["env_step_calls"]),
                    float(current["env_step_total_sec"]),
                    float(env_step_avg),
                    int(current["insert_calls"]),
                    float(current["insert_total_sec"]),
                    float(insert_avg),
                    float(current["compute_total_sec"]),
                    float(current["train_total_sec"]),
                    int(current["eval_calls"]),
                    float(current["eval_total_sec"]),
                    float(eval_avg),
                    int(current["save_gif_calls"]),
                    float(current["save_gif_total_sec"]),
                    float(save_gif_avg),
                ]
            )
        self._time_stat_ctx["current"] = None

    def warmup(self):
        """
        功能:
            重置训练环境并把第0步观测写入各受控agent buffer。
        输入:
            无。
        输出:
            无。
        """
        # Step 1: reset env
        obs = self.envs.reset(mode="initial")
        self._write_reset_obs_to_buffers(obs)

    def _write_reset_obs_to_buffers(self, obs):
        """
        功能:
            将环境reset后的初始观测写入buffer起始位置，并清空RNN状态。
        输入:
            obs (np.ndarray): shape=(n_rollout_threads,num_agents,obs_dim)。
        输出:
            无。
        """
        # Step 1: 计算集中式critic使用的share_obs
        share_obs_all = np.array([list(chain(*o)) for o in obs], dtype=np.float32)

        # Step 2: 写入每个受控agent的buffer[0]
        for agent_id in self.controlled_agents:
            if self.use_centralized_V:
                share_obs = share_obs_all
            else:
                share_obs = np.array(list(obs[:, agent_id]), dtype=np.float32)
            self.buffers[agent_id].share_obs[0] = share_obs.copy()
            self.buffers[agent_id].obs[0] = np.array(list(obs[:, agent_id]), dtype=np.float32).copy()
            self.buffers[agent_id].rnn_states[0] = 0.0
            self.buffers[agent_id].rnn_states_critic[0] = 0.0
            self.buffers[agent_id].masks[0] = 1.0
            self.buffers[agent_id].active_masks[0] = 1.0

    @torch.no_grad()
    def collect(self, step):
        """
        功能:
            在训练环境采样动作并汇总为环境动作张量。
        输入:
            step (int): 当前episode时间步。
        输出:
            tuple: values/actions/action_log_probs/rnn_states/rnn_states_critic/actions_env。
        """
        # Step 1: 准备动作容器（未受控agent默认零动作）
        act_dim = int(self.envs.action_space[0].shape[0])
        actions_env = np.zeros((self.n_rollout_threads, self.num_agents, act_dim), dtype=np.float32)

        # Step 2: 逐受控agent采样动作
        values = {}
        actions = {}
        action_log_probs = {}
        rnn_states = {}
        rnn_states_critic = {}

        for agent_id in self.controlled_agents:
            role = self.agent_role[agent_id]
            trainer = self.role_trainers[role]
            trainer.prep_rollout()
            buffer = self.buffers[agent_id]

            value, action, action_log_prob, rnn_state, rnn_state_critic = trainer.policy.get_actions(
                buffer.share_obs[step],
                buffer.obs[step],
                buffer.rnn_states[step],
                buffer.rnn_states_critic[step],
                buffer.masks[step],
            )

            values[agent_id] = _t2n(value)
            actions[agent_id] = _t2n(action)
            action_log_probs[agent_id] = _t2n(action_log_prob)
            rnn_states[agent_id] = _t2n(rnn_state)
            rnn_states_critic[agent_id] = _t2n(rnn_state_critic)
            actions_env[:, agent_id, :] = actions[agent_id]

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        """
        功能:
            将一步交互数据写入各受控agent的replay buffer。
        输入:
            data (tuple): obs/rewards/dones/infos/values/actions/action_log_probs/rnn_states/rnn_states_critic。
        输出:
            无。
        """
        # Step 1: 解包输入数据
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        # Step 2: 计算集中式share_obs
        share_obs_all = np.array([list(chain(*o)) for o in obs], dtype=np.float32)

        # Step 3: 写入每个受控agent buffer
        for agent_id in self.controlled_agents:
            done_mask = dones[:, agent_id].astype(bool)
            rnn_state = rnn_states[agent_id].copy()
            rnn_state_critic = rnn_states_critic[agent_id].copy()

            rnn_state[done_mask] = np.zeros(
                (done_mask.sum(), self.recurrent_N, self.hidden_size), dtype=np.float32
            )
            rnn_state_critic[done_mask] = np.zeros(
                (done_mask.sum(), self.recurrent_N, self.hidden_size), dtype=np.float32
            )

            masks = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
            masks[done_mask] = 0.0

            if self.use_centralized_V:
                share_obs = share_obs_all
            else:
                share_obs = np.array(list(obs[:, agent_id]), dtype=np.float32)

            active_masks = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
            for env_i in range(self.n_rollout_threads):
                agent_info = infos[env_i][agent_id] if agent_id < len(infos[env_i]) else {}
                active_flag = bool(agent_info.get("active_agent", True)) and bool(agent_info.get("alive", True))
                if not active_flag:
                    active_masks[env_i, 0] = 0.0
            active_masks[done_mask] = 0.0

            self.buffers[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id]), dtype=np.float32),
                rnn_state,
                rnn_state_critic,
                actions[agent_id],
                action_log_probs[agent_id],
                values[agent_id],
                rewards[:, agent_id],
                masks,
                active_masks=active_masks,
            )

    @torch.no_grad()
    def compute(self):
        """
        功能:
            为每个受控agent计算下一状态价值并写回returns。
        输入:
            无。
        输出:
            无。
        """
        # Step 1: 对每个受控agent计算GAE returns
        for agent_id in self.controlled_agents:
            role = self.agent_role[agent_id]
            trainer = self.role_trainers[role]
            trainer.prep_rollout()
            next_value = trainer.policy.get_values(
                self.buffers[agent_id].share_obs[-1],
                self.buffers[agent_id].rnn_states_critic[-1],
                self.buffers[agent_id].masks[-1],
            )
            next_value = _t2n(next_value)
            self.buffers[agent_id].compute_returns(next_value, trainer.value_normalizer)

    def train(self):
        """
        功能:
            执行PPO参数更新，并返回各受控agent训练日志。
        输入:
            无。
        输出:
            dict: 训练日志字典。
        """
        if self.role_buffer_mode == "role_shared":
            return self._train_with_role_shared_buffers()
        return self._train_with_separate_buffers()

    def _train_with_separate_buffers(self):
        """
        功能:
            使用“每个agent一个buffer”的方式执行训练更新（现有默认行为）。
        输入:
            无。
        输出:
            dict: 按agent记录的训练日志字典。
        """
        # Step 1: 按受控agent进行训练
        train_infos = {}
        for agent_id in self.controlled_agents:
            role = self.agent_role[agent_id]
            trainer = self.role_trainers[role]
            trainer.prep_training()
            info = trainer.train(self.buffers[agent_id])
            self.buffers[agent_id].after_update()
            info["average_episode_rewards"] = (
                float(np.mean(self.buffers[agent_id].rewards)) * self.episode_length
            )
            train_infos[f"{role}_{agent_id}"] = info
        return train_infos

    def _sync_role_shared_train_buffer(self, role_name, agent_ids):
        """
        功能:
            将同角色的多个agent buffer沿线程维拼接到角色共享训练buffer。
        输入:
            role_name (str): 角色名称（hunter/target）。
            agent_ids (list[int]): 该角色包含的agent id列表。
        输出:
            SeparatedReplayBuffer: 角色共享训练buffer（已填充本episode数据）。
        """
        role_buffer = self.role_shared_train_buffers[role_name]

        def _concat_field(field_name):
            field_vals = [getattr(self.buffers[int(aid)], field_name) for aid in agent_ids]
            return np.concatenate(field_vals, axis=1)

        # Step 1: 逐字段拼接
        role_buffer.share_obs[...] = _concat_field("share_obs")
        role_buffer.obs[...] = _concat_field("obs")
        role_buffer.rnn_states[...] = _concat_field("rnn_states")
        role_buffer.rnn_states_critic[...] = _concat_field("rnn_states_critic")
        role_buffer.value_preds[...] = _concat_field("value_preds")
        role_buffer.returns[...] = _concat_field("returns")
        role_buffer.actions[...] = _concat_field("actions")
        role_buffer.action_log_probs[...] = _concat_field("action_log_probs")
        role_buffer.rewards[...] = _concat_field("rewards")
        role_buffer.masks[...] = _concat_field("masks")
        role_buffer.bad_masks[...] = _concat_field("bad_masks")
        role_buffer.active_masks[...] = _concat_field("active_masks")
        if role_buffer.available_actions is not None:
            role_buffer.available_actions[...] = _concat_field("available_actions")

        # Step 2: 保持step状态与完整episode对齐
        role_buffer.step = int(self.buffers[int(agent_ids[0])].step)
        return role_buffer

    def _train_with_role_shared_buffers(self):
        """
        功能:
            使用“同角色共享训练buffer”执行训练更新（每个角色每轮只更新一次）。
        输入:
            无。
        输出:
            dict: 按角色记录的训练日志字典。
        """
        train_infos = {}
        for role_name, agent_ids in self.role_to_agents.items():
            if len(agent_ids) == 0:
                continue
            trainer = self.role_trainers[role_name]
            trainer.prep_training()
            role_buffer = self._sync_role_shared_train_buffer(role_name, agent_ids)
            info = trainer.train(role_buffer)

            # Step 1: 角色奖励按该角色全部agent的reward均值统计。
            role_rewards = [self.buffers[int(aid)].rewards for aid in agent_ids]
            role_reward_mean = float(np.mean(np.concatenate(role_rewards, axis=1))) * self.episode_length
            info["average_episode_rewards"] = role_reward_mean
            info["role_num_agents"] = int(len(agent_ids))
            train_infos[f"{role_name}_shared"] = info

            # Step 2: 训练后仍需推进各agent buffer到下一episode起点。
            for aid in agent_ids:
                self.buffers[int(aid)].after_update()
        return train_infos

    def save(self):
        """
        功能:
            将当前角色策略参数保存到常规模型目录。
        输入:
            无。
        输出:
            无。
        """
        # Step 1: 按角色保存actor/critic
        for role, trainer in self.role_trainers.items():
            torch.save(trainer.policy.actor.state_dict(), os.path.join(self.save_dir, f"actor_{role}.pt"))
            torch.save(trainer.policy.critic.state_dict(), os.path.join(self.save_dir, f"critic_{role}.pt"))

    @torch.no_grad()
    def eval(
        self,
        total_num_steps,
        episode,
        eval_envs=None,
        bucket="fixed",
        save_gifs=False,
        record_logs=True,
        gif_output_dir=None,
        gif_env_limit=None,
    ):
        """
        功能:
            执行周期性evaluation（每个eval_env运行1个episode），
            记录reward/捕获率/捕获步数；仅在save_gifs=True时保存GIF。
        输入:
            total_num_steps (int): 当前累计环境步数。
            episode (int): 当前训练episode编号。
            eval_envs (VecEnv | None): 评估环境；None时使用self.eval_envs。
            bucket (str): 评估桶名称（fixed/target_learn等）。
            save_gifs (bool): 是否保存GIF。
            record_logs (bool): 是否写TB与CSV。
            gif_output_dir (str | Path | None): GIF输出目录，None时使用self.gif_dir。
            gif_env_limit (int | None): 保存GIF的环境数量上限，None表示保存全部。
        输出:
            dict: evaluation指标字典。
        """
        # Step 1: 准备评估状态
        if eval_envs is None:
            eval_envs = self.eval_envs

        if eval_envs is None:
            return {
                "eval_reward": 0.0,
                "capture_rate": 0.0,
                "capture_steps": float(self.eval_episode_length),
                "alive_rate": 0.0,
                "max_escape_gap_angle": float("nan"),
                "captured_episodes": 0,
                "total_eval_episodes": 0,
            }

        eval_obs = eval_envs.reset(mode="recover")
        n_env = int(eval_obs.shape[0])
        eval_num_agents = int(eval_obs.shape[1])
        eval_num_hunters = int(max(1, eval_num_agents - 1))
        eval_target_index = int(eval_num_agents - 1)
        act_dim = int(eval_envs.action_space[0].shape[0])
        eval_controlled_agents = list(range(eval_num_hunters))
        if self.target_trainable:
            eval_controlled_agents.append(eval_target_index)
        self._print_timed(
            "[EvalStart] bucket={}, episode={}, total_steps={}, eval_envs={}".format(
                str(bucket), int(episode), int(total_num_steps), n_env
            )
        )
        if hasattr(eval_envs, "capture_terminal_frame"):
            eval_envs.capture_terminal_frame = bool(save_gifs)

        eval_rnn_states = {
            aid: np.zeros((n_env, self.recurrent_N, self.hidden_size), dtype=np.float32)
            for aid in eval_controlled_agents
        }
        eval_masks = {
            aid: np.ones((n_env, 1), dtype=np.float32)
            for aid in eval_controlled_agents
        }

        env_episode_rewards = np.zeros(n_env, dtype=np.float32)
        env_captured = np.zeros(n_env, dtype=bool)
        env_capture_step = np.full(n_env, -1, dtype=np.int32)
        env_alive_rate = np.full(n_env, 0.0, dtype=np.float32)
        env_active_hunter_count = np.full(n_env, int(eval_num_hunters), dtype=np.int32)
        # 每个环境仅记录“首次成功捕捉时刻”的escape_gap_angle。
        env_capture_escape_gap_angle = np.full(n_env, np.nan, dtype=np.float32)
        env_finished = np.zeros(n_env, dtype=bool)
        eval_frames = [[] for _ in range(n_env)]
        gif_env_ids = list(range(n_env))
        if gif_env_limit is not None:
            gif_env_ids = list(range(max(0, min(int(gif_env_limit), int(n_env)))))
        gif_env_id_set = set(gif_env_ids)

        # Step 2: 记录初始帧
        if save_gifs:
            for env_i in gif_env_ids:
                frame = eval_envs.render(
                    mode="rgb_array",
                    env_id=int(env_i),
                    title=f"Eval({str(bucket)}) Episode {int(episode)}",
                )
                if isinstance(frame, np.ndarray):
                    eval_frames[env_i].append(frame.copy())

        # Step 3: rollout一个评估episode长度
        eval_infos = [None for _ in range(n_env)]
        for eval_step in range(self.eval_episode_length):
            eval_actions_env = np.zeros((n_env, eval_num_agents, act_dim), dtype=np.float32)

            for agent_id in eval_controlled_agents:
                role = "target" if (agent_id == eval_target_index and self.target_trainable) else "hunter"
                trainer = self.role_trainers[role]
                trainer.prep_rollout()
                eval_action, next_eval_rnn_state = trainer.policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[agent_id],
                    eval_masks[agent_id],
                    deterministic=True,
                )
                eval_actions_env[:, agent_id, :] = _t2n(eval_action)
                eval_rnn_states[agent_id] = _t2n(next_eval_rnn_state)

            eval_obs, eval_rewards, eval_dones, eval_infos = eval_envs.step(eval_actions_env)

            for env_i in range(n_env):
                metric_valid = False
                metric_angle = float("nan")
                if eval_infos[env_i] is not None and len(eval_infos[env_i]) > 0:
                    env_active_hunter_count[env_i] = int(
                        self._extract_active_hunter_count(
                            eval_infos[env_i],
                            max_hunters_num=eval_num_hunters,
                        )
                    )
                    metric_info = (
                        eval_infos[env_i][eval_target_index]
                        if eval_target_index < len(eval_infos[env_i])
                        else eval_infos[env_i][0]
                    )
                    metric_valid = bool(metric_info.get("max_escape_gap_metric_valid", False))
                    metric_angle = float(metric_info.get("max_escape_gap_angle", float("nan")))
                    
                if env_finished[env_i]:
                    continue

                env_episode_rewards[env_i] += float(
                    np.sum(eval_rewards[env_i, : eval_num_hunters, 0])
                    / max(1, eval_num_hunters)
                )
                if (not env_captured[env_i]) and any(
                    bool(agent_info.get("captured", False)) for agent_info in eval_infos[env_i]
                ):
                    env_captured[env_i] = True
                    env_capture_step[env_i] = int(eval_step + 1)
                    # 仅记录“成功捕捉当下”的escape_gap_angle。
                    if metric_valid and np.isfinite(metric_angle):
                        env_capture_escape_gap_angle[env_i] = float(metric_angle)
                        
                if bool(np.all(eval_dones[env_i])):
                    env_alive_rate[env_i] = float(
                        self._compute_hunter_alive_rate(
                            eval_infos[env_i],
                            max_hunters_num=eval_num_hunters,
                        )
                    )
                    if save_gifs and env_i in gif_env_id_set:
                        terminal_frame = self._extract_terminal_frame(eval_infos[env_i])
                        if terminal_frame is not None:
                            eval_frames[env_i].append(terminal_frame.copy())
                    env_finished[env_i] = True

            if save_gifs and (eval_step + 1) % self.gif_frame_interval == 0:
                for env_i in gif_env_ids:
                    if env_finished[env_i]:
                        continue
                    frame = eval_envs.render(
                        mode="rgb_array",
                        env_id=int(env_i),
                        title=f"Eval({str(bucket)}) Episode {int(episode)}",
                    )
                    if isinstance(frame, np.ndarray):
                        eval_frames[env_i].append(frame.copy())

            for agent_id in eval_controlled_agents:
                done_mask = eval_dones[:, agent_id].astype(bool)
                eval_masks[agent_id] = np.ones((n_env, 1), dtype=np.float32)
                eval_masks[agent_id][done_mask] = 0.0
                eval_rnn_states[agent_id][done_mask] = 0.0

            if bool(np.all(env_finished)):
                break

        # Step 3.1: 对未结束环境做alive_rate回填，避免日志或GIF头信息缺失。
        for env_i in range(n_env):
            if bool(env_finished[env_i]):
                continue
            if eval_infos[env_i] is None or len(eval_infos[env_i]) == 0:
                continue
            env_alive_rate[env_i] = float(
                self._compute_hunter_alive_rate(
                    eval_infos[env_i],
                    max_hunters_num=eval_num_hunters,
                )
            )

        # Step 4: 汇总评估指标
        total_eval_episodes = int(n_env)
        captured_episodes = int(np.sum(env_captured))
        capture_rate = float(captured_episodes / max(1, total_eval_episodes))
        eval_reward = float(np.mean(env_episode_rewards)) if total_eval_episodes > 0 else 0.0

        captured_steps = [int(env_capture_step[i]) for i in range(n_env) if env_captured[i] and env_capture_step[i] > 0]
        capture_steps = (
            float(np.mean(captured_steps)) if len(captured_steps) > 0 else float(self.eval_episode_length)
        )
        captured_valid_mask = np.logical_and(env_captured, np.isfinite(env_capture_escape_gap_angle))
        if bool(np.any(captured_valid_mask)):
            max_escape_gap_angle = float(np.mean(env_capture_escape_gap_angle[captured_valid_mask]))
        else:
            max_escape_gap_angle = float("nan")

        eval_metrics = {
            "eval_reward": eval_reward,
            "capture_rate": capture_rate,
            "capture_steps": capture_steps,
            "alive_rate": float(np.mean(env_alive_rate)) if total_eval_episodes > 0 else 0.0,
            "max_escape_gap_angle": float(max_escape_gap_angle),
            "captured_episodes": int(captured_episodes),
            "total_eval_episodes": int(total_eval_episodes),
        }
        eval_metrics_by_hunters = self._build_eval_metrics_by_hunter_count(
            env_episode_rewards=env_episode_rewards,
            env_captured=env_captured,
            env_capture_step=env_capture_step,
            env_alive_rate=env_alive_rate,
            env_active_hunter_count=env_active_hunter_count,
            env_escape_gap_angle_mean=env_capture_escape_gap_angle,
        )

        self._print_metric_table(
            "EvalMetrics[{}]".format(str(bucket)),
            {
                "episode": str(int(episode)),
                "reward": "{:.4f}".format(eval_reward),
                "capture_rate": "{:.4f}".format(capture_rate),
                "capture_steps": "{:.2f}".format(capture_steps),
                "alive_rate": "{:.4f}".format(float(eval_metrics["alive_rate"])),
                "max_escape_gap": (
                    "{:.4f}".format(float(max_escape_gap_angle))
                    if np.isfinite(max_escape_gap_angle)
                    else "NA"
                ),
                "captured_ep": str(int(captured_episodes)),
                "total_eval_ep": str(int(total_eval_episodes)),
            },
        )

        # Step 5: 记录TB与CSV（可选）
        if record_logs:
            self.writter.add_scalars(
                f"eval/{str(bucket)}/eval_reward",
                {f"eval/{str(bucket)}/eval_reward": eval_reward},
                total_num_steps,
            )
            self.writter.add_scalars(
                f"eval/{str(bucket)}/capture_rate",
                {f"eval/{str(bucket)}/capture_rate": capture_rate},
                total_num_steps,
            )
            self.writter.add_scalars(
                f"eval/{str(bucket)}/capture_steps",
                {f"eval/{str(bucket)}/capture_steps": capture_steps},
                total_num_steps,
            )
            self.writter.add_scalars(
                f"eval/{str(bucket)}/alive_rate",
                {f"eval/{str(bucket)}/alive_rate": float(eval_metrics["alive_rate"])},
                total_num_steps,
            )
            if np.isfinite(max_escape_gap_angle):
                self.writter.add_scalars(
                    f"eval/{str(bucket)}/max_escape_gap_angle",
                    {f"eval/{str(bucket)}/max_escape_gap_angle": float(max_escape_gap_angle)},
                    total_num_steps,
                )

            self._append_eval_csv(
                episode=episode,
                total_num_steps=total_num_steps,
                bucket=bucket,
                eval_reward=eval_reward,
                capture_rate=capture_rate,
                capture_steps=capture_steps,
                alive_rate=float(eval_metrics["alive_rate"]),
                max_escape_gap_angle=float(max_escape_gap_angle),
                captured_episodes=captured_episodes,
                total_eval_episodes=total_eval_episodes,
            )
            for hunter_count, grouped_metrics in sorted(eval_metrics_by_hunters.items(), key=lambda x: int(x[0])):
                self._append_eval_csv(
                    episode=episode,
                    total_num_steps=total_num_steps,
                    bucket=f"{str(bucket)}_num_hunters_{int(hunter_count)}",
                    eval_reward=float(grouped_metrics["eval_reward"]),
                    capture_rate=float(grouped_metrics["capture_rate"]),
                    capture_steps=float(grouped_metrics["capture_steps"]),
                    alive_rate=float(grouped_metrics["alive_rate"]),
                    max_escape_gap_angle=float(grouped_metrics["max_escape_gap_angle"]),
                    captured_episodes=int(grouped_metrics["captured_episodes"]),
                    total_eval_episodes=int(grouped_metrics["total_eval_episodes"]),
                )

        # Step 6: 保存每个eval_env GIF（可选）
        if save_gifs:
            out_dir = Path(self.gif_dir) if gif_output_dir is None else Path(gif_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for env_i in gif_env_ids:
                gif_path = out_dir / f"e-{str(bucket)}-{int(episode)}-env-{int(env_i)}.gif"
                self._save_gif(
                    eval_frames[env_i],
                    gif_path,
                    episode_id=int(episode),
                    capture_step=None if env_capture_step[env_i] <= 0 else int(env_capture_step[env_i]),
                    alive_rate=float(env_alive_rate[env_i]),
                )
            self._print_timed(
                "[GIF] saved eval({}) gifs for episode {} to {} ({} envs)".format(
                    str(bucket), int(episode), str(out_dir), int(len(gif_env_ids))
                )
            )
        # Step 7: 始终保存“按hunter数量分桶”曲线图，不受log_gif控制。
        out_dir = Path(self.gif_dir) if gif_output_dir is None else Path(gif_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._update_eval_hunter_bucket_plot(
            episode=episode,
            total_num_steps=total_num_steps,
            bucket=bucket,
            eval_metrics_by_hunters=eval_metrics_by_hunters,
            output_dir=out_dir,
        )
        if hasattr(eval_envs, "capture_terminal_frame"):
            eval_envs.capture_terminal_frame = False

        return eval_metrics

    def _save_gif(self, frames, gif_path, episode_id=None, capture_step=None, alive_rate=None):
        """
        功能:
            将帧序列写为GIF，并在顶部叠加固定episode级摘要标题。
        输入:
            frames (list[np.ndarray]): RGB帧列表。
            gif_path (Path): GIF输出路径。
            episode_id (int | None): 当前episode编号。
            capture_step (int | None): 捕获发生步数，None表示未捕获。
            alive_rate (float | None): 终止时hunter存活率。
        输出:
            无。
        """
        # Step 1: 空帧保护
        if frames is None or len(frames) == 0:
            return

        # Step 2: 保留全部采样帧，不做抽帧限帧；每帧时长由配置控制。
        sampled_frames = [f for f in frames]

        # Step 3: 生成固定标题并叠加到所有帧顶部。
        header_text = self._build_gif_header_text(episode_id, capture_step, alive_rate)
        sampled_frames = [self._overlay_gif_header(frame, header_text) for frame in sampled_frames]

        # Step 4: 终止帧额外停留3.0秒便于观察结束状态（依赖gif_frame_duration）。
        hold_frames = max(1, int(round(3.0 / max(self.gif_frame_duration, 1e-6))))
        sampled_frames.extend([sampled_frames[-1].copy() for _ in range(hold_frames)])

        # Step 5: 写出GIF
        imageio.mimsave(str(gif_path), sampled_frames, duration=float(self.gif_frame_duration), loop=0)

    def _build_gif_header_text(self, episode_id, capture_step, alive_rate):
        """
        功能:
            构建GIF固定顶部标题文本。
        输入:
            episode_id (int | None): 当前episode编号。
            capture_step (int | None): 捕获发生步数，None表示未捕获。
            alive_rate (float | None): 终止时hunter存活率。
        输出:
            str: 用于叠加到GIF顶部的固定标题字符串。
        """
        # Step 1: 兼容空值并格式化字段
        episode_str = "NA" if episode_id is None else str(int(episode_id))
        capture_str = "NA" if capture_step is None else str(int(capture_step))
        alive_str = "NA" if alive_rate is None else f"{100.0 * float(alive_rate):.1f}%"

        # Step 2: 返回固定标题模板
        return f"Episode {episode_str} | Capture Step {capture_str} | Alive Rate {alive_str}"

    def _overlay_gif_header(self, frame, header_text):
        """
        功能:
            在单帧图像顶部叠加固定标题横幅，返回新的RGB帧。
        输入:
            frame (np.ndarray): 原始RGB帧。
            header_text (str): 顶部标题文本。
        输出:
            np.ndarray: 叠加标题后的RGB帧。
        """
        # Step 1: 将输入帧归一到uint8 RGB格式
        frame_arr = np.asarray(frame)
        if frame_arr.dtype != np.uint8:
            frame_arr = np.clip(frame_arr, 0, 255).astype(np.uint8)
        if frame_arr.ndim == 2:
            frame_arr = np.repeat(frame_arr[..., None], repeats=3, axis=2)
        if frame_arr.shape[2] > 3:
            frame_arr = frame_arr[:, :, :3]

        # Step 2: 使用matplotlib生成“标题横幅 + 原图”拼接画布
        h, w = frame_arr.shape[0], frame_arr.shape[1]
        header_h = max(28, int(round(h * 0.08)))
        dpi = 100
        fig = plt.figure(figsize=(w / dpi, (h + header_h) / dpi), dpi=dpi)
        gs = fig.add_gridspec(2, 1, height_ratios=[header_h, h], hspace=0.0)
        ax_head = fig.add_subplot(gs[0])
        ax_img = fig.add_subplot(gs[1])

        ax_head.set_facecolor("#101010")
        ax_head.text(
            0.5,
            0.5,
            str(header_text),
            color="white",
            fontsize=11,
            ha="center",
            va="center",
        )
        ax_head.set_xticks([])
        ax_head.set_yticks([])
        for spine in ax_head.spines.values():
            spine.set_visible(False)

        ax_img.imshow(frame_arr)
        ax_img.axis("off")
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, hspace=0.0)

        # Step 3: 回读画布像素并关闭figure，避免内存泄漏
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        out_frame = np.asarray(buffer[:, :, :3], dtype=np.uint8).copy()
        plt.close(fig)
        return out_frame

    def _extract_terminal_frame(self, env_infos):
        """
        功能:
            从单环境infos中提取终止时刻帧（若有）。
        输入:
            env_infos (list[dict] | np.ndarray): 单个环境的多agent info集合。
        输出:
            np.ndarray | None: 终止帧RGB数组，不存在则返回None。
        """
        if env_infos is None:
            return None
        for agent_info in env_infos:
            if isinstance(agent_info, dict) and "terminal_frame" in agent_info:
                return agent_info["terminal_frame"]
        return None

    def log_train(self, train_infos, total_num_steps):
        """
        功能:
            将训练指标写入TensorBoard。
        输入:
            train_infos (dict): train()返回的训练指标字典。
            total_num_steps (int): 当前累计环境步数。
        输出:
            无。
        """
        # Step 1: 展开tag/metric并写入TB
        for tag, info in train_infos.items():
            for metric_name, metric_value in info.items():
                tb_key = f"{tag}/{metric_name}"
                self.writter.add_scalars(tb_key, {tb_key: metric_value}, total_num_steps)

    def _init_csv_files(self):
        """
        功能:
            创建训练与评估CSV文件并写入表头。
        输入:
            无。
        输出:
            无。
        """
        # Step 1: 创建训练日志CSV
        with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "total_num_steps",
                    "num_hunters",
                    "num_targets",
                    "hunter_alive_mean",
                    "target_alive_mean",
                    "hunter_reward_mean",
                    "target_reward_mean",
                ]
            )

        # Step 2: 创建评估日志CSV
        with open(self.eval_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "total_num_steps",
                    "bucket",
                    "eval_reward",
                    "capture_rate",
                    "capture_steps",
                    "alive_rate",
                    "max_escape_gap_angle",
                    "captured_episodes",
                    "total_eval_episodes",
                ]
            )

    def _append_log_csv(
        self,
        episode,
        total_num_steps,
        hunter_alive_mean,
        target_alive_mean,
        hunter_reward_mean,
        target_reward_mean,
    ):
        """
        功能:
            向log.csv追加一行训练统计。
        输入:
            episode (int): 当前训练episode编号。
            total_num_steps (int): 当前累计环境步数。
            hunter_alive_mean (float): Hunter平均存活比例。
            target_alive_mean (float): Target平均存活比例。
            hunter_reward_mean (float): Hunter平均reward。
            target_reward_mean (float): Target平均reward。
        输出:
            无。
        """
        # Step 1: 追加训练统计行
        with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(episode),
                    int(total_num_steps),
                    int(self.train_max_hunters_num),
                    1,
                    float(hunter_alive_mean),
                    float(target_alive_mean),
                    float(hunter_reward_mean),
                    float(target_reward_mean),
                ]
            )

    def _append_eval_csv(
        self,
        episode,
        total_num_steps,
        bucket,
        eval_reward,
        capture_rate,
        capture_steps,
        alive_rate,
        max_escape_gap_angle,
        captured_episodes,
        total_eval_episodes,
    ):
        """
        功能:
            向eval.csv追加一行评估统计。
        输入:
            episode (int): 当前训练episode编号。
            total_num_steps (int): 当前累计环境步数。
            eval_reward (float): 评估平均reward。
            capture_rate (float): 捕获成功率。
            capture_steps (float): 平均捕获步数（仅成功episode）。
            alive_rate (float): 终止时hunter平均存活率（按激活hunter归一化）。
            max_escape_gap_angle (float): 最大潜在可逃脱空间夹角均值（弧度）。
            captured_episodes (int): 捕获成功episode数。
            total_eval_episodes (int): 评估episode总数。
        输出:
            无。
        """
        # Step 1: 追加评估统计行
        with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(episode),
                    int(total_num_steps),
                    str(bucket),
                    float(eval_reward),
                    float(capture_rate),
                    float(capture_steps),
                    float(alive_rate),
                    float(max_escape_gap_angle),
                    int(captured_episodes),
                    int(total_eval_episodes),
                ]
            )

    def _extract_alive_stats(self, infos):
        """
        功能:
            从环境infos中提取Hunter与Target存活统计。
        输入:
            infos (np.ndarray | list): shape=[n_env, n_agent] 的info集合。
        输出:
            tuple[float, float]: (hunter_alive_mean, target_alive_mean)。
        """
        # Step 1: 空infos保护
        if infos is None:
            return 0.0, 0.0

        # Step 2: 分环境统计存活率
        hunter_alive = []
        target_alive = []
        for env_infos in infos:
            if env_infos is None or len(env_infos) == 0:
                continue
            hunter_alive.append(
                float(
                    self._compute_hunter_alive_rate(
                        env_infos,
                        max_hunters_num=self.train_max_hunters_num,
                    )
                )
            )
            target_alive.append(float(env_infos[self.target_index].get("alive", False)))

        if len(hunter_alive) == 0:
            return 0.0, 0.0
        return float(np.mean(hunter_alive)), float(np.mean(target_alive))

    def _extract_active_hunter_count(self, env_infos, max_hunters_num):
        """
        功能:
            统计单环境当前激活的hunter数量。
        输入:
            env_infos (list[dict]): 单环境多智能体info列表。
        输出:
            int: 激活hunter数量，最小返回1。
        """
        # Step 1: 空输入回退到配置最大hunter数量。
        if env_infos is None or len(env_infos) == 0:
            return int(max(1, int(max_hunters_num)))

        # Step 2: 优先按active_agent字段统计激活hunter数量。
        active_count = 0
        for hid in range(int(max_hunters_num)):
            if hid >= len(env_infos):
                break
            agent_info = env_infos[hid]
            if not isinstance(agent_info, dict):
                continue
            if bool(agent_info.get("active_agent", True)):
                active_count += 1
        return int(max(1, active_count))

    def _compute_hunter_alive_rate(self, env_infos, max_hunters_num):
        """
        功能:
            计算单环境hunter存活率（分母为激活hunter数量，而非最大hunter槽位）。
        输入:
            env_infos (list[dict]): 单环境多智能体info列表。
        输出:
            float: hunter存活率，范围[0,1]。
        """
        # Step 1: 空输入直接返回0。
        if env_infos is None or len(env_infos) == 0:
            return 0.0

        # Step 2: 仅统计active_agent=True的hunter，修正分母偏大问题。
        alive_count = 0.0
        active_count = 0.0
        for hid in range(int(max_hunters_num)):
            if hid >= len(env_infos):
                break
            agent_info = env_infos[hid]
            if not isinstance(agent_info, dict):
                continue
            if not bool(agent_info.get("active_agent", True)):
                continue
            active_count += 1.0
            if bool(agent_info.get("alive", False)):
                alive_count += 1.0
        if active_count <= 0.0:
            return 0.0
        return float(alive_count / active_count)

    def _build_eval_metrics_by_hunter_count(
        self,
        env_episode_rewards,
        env_captured,
        env_capture_step,
        env_alive_rate,
        env_active_hunter_count,
        env_escape_gap_angle_mean,
    ):
        """
        功能:
            将评估环境按hunter数量分桶，聚合各性能指标。
        输入:
            env_episode_rewards (np.ndarray): shape=(n_env,) 每环境reward。
            env_captured (np.ndarray): shape=(n_env,) 每环境是否捕获成功。
            env_capture_step (np.ndarray): shape=(n_env,) 每环境捕获步数，未捕获为-1。
            env_alive_rate (np.ndarray): shape=(n_env,) 每环境终止alive_rate。
            env_active_hunter_count (np.ndarray): shape=(n_env,) 每环境激活hunter数量。
            env_escape_gap_angle_mean (np.ndarray): shape=(n_env,) 每环境最大潜在可逃脱空间角度均值（NaN表示无效）。
        输出:
            dict[int, dict]: 按hunter数量索引的指标字典。
        """
        # Step 1: 规范化输入，准备分桶容器。
        rewards = np.asarray(env_episode_rewards, dtype=np.float32).reshape(-1)
        captured_flags = np.asarray(env_captured, dtype=bool).reshape(-1)
        capture_steps = np.asarray(env_capture_step, dtype=np.int32).reshape(-1)
        alive_rates = np.asarray(env_alive_rate, dtype=np.float32).reshape(-1)
        hunter_counts = np.asarray(env_active_hunter_count, dtype=np.int32).reshape(-1)
        escape_gap_means = np.asarray(env_escape_gap_angle_mean, dtype=np.float32).reshape(-1)
        out = {}

        # Step 2: 逐hunter数量聚合reward/capture_rate/capture_steps。
        for hunter_count in sorted([int(x) for x in np.unique(hunter_counts)]):
            mask = hunter_counts == int(hunter_count)
            if not bool(np.any(mask)):
                continue
            total_eval_episodes = int(np.sum(mask))
            captured_episodes = int(np.sum(captured_flags[mask]))
            grouped_capture_steps = [
                int(x)
                for x in capture_steps[mask]
                if int(x) > 0
            ]
            mean_capture_steps = (
                float(np.mean(grouped_capture_steps))
                if len(grouped_capture_steps) > 0
                else float(self.eval_episode_length)
            )
            grouped_escape_angles = escape_gap_means[
                np.logical_and(mask, np.logical_and(captured_flags, np.isfinite(escape_gap_means)))
            ]
            grouped_escape_angle_mean = (
                float(np.mean(grouped_escape_angles))
                if grouped_escape_angles.size > 0
                else float("nan")
            )
            out[int(hunter_count)] = {
                "eval_reward": float(np.mean(rewards[mask])) if total_eval_episodes > 0 else 0.0,
                "capture_rate": float(captured_episodes / max(1, total_eval_episodes)),
                "capture_steps": float(mean_capture_steps),
                "alive_rate": float(np.mean(alive_rates[mask])) if total_eval_episodes > 0 else 0.0,
                "max_escape_gap_angle": float(grouped_escape_angle_mean),
                "captured_episodes": int(captured_episodes),
                "total_eval_episodes": int(total_eval_episodes),
            }
        return out

    def _canonical_eval_bucket_name(self, bucket):
        """
        功能:
            统一评估bucket命名，便于跨桶绘图对齐。
        输入:
            bucket (str): 原始bucket名称。
        输出:
            str: 标准化bucket名称（fixed或learn等）。
        """
        # Step 1: learn桶兼容target_learn命名。
        bucket_name = str(bucket).lower()
        if bucket_name in ("target_learn", "learn"):
            return "learn"
        return bucket_name

    def _update_eval_hunter_bucket_plot(
        self,
        episode,
        total_num_steps,
        bucket,
        eval_metrics_by_hunters,
        output_dir,
    ):
        """
        功能:
            更新评估分桶缓存并输出“指标-随hunter数量变化”曲线图。
        输入:
            episode (int): 当前episode编号。
            total_num_steps (int): 当前累计环境步数。
            bucket (str): 当前评估桶名称。
            eval_metrics_by_hunters (dict[int, dict]): 当前bucket的分桶指标。
            output_dir (Path): 图像输出目录。
        输出:
            无。
        """
        # Step 1: 构建缓存键并写入当前bucket数据。
        out_dir = Path(output_dir)
        cache_key = (str(out_dir.resolve()), int(episode), int(total_num_steps))
        if cache_key not in self.eval_hunter_bucket_metrics_cache:
            self.eval_hunter_bucket_metrics_cache[cache_key] = {}
        canonical_bucket = self._canonical_eval_bucket_name(bucket)
        self.eval_hunter_bucket_metrics_cache[cache_key][canonical_bucket] = dict(eval_metrics_by_hunters)

        # Step 2: 基于当前可用桶（fixed/learn）绘制并覆盖保存同名图。
        bucket_to_metrics = self.eval_hunter_bucket_metrics_cache[cache_key]
        metrics_order = [
            "eval_reward",
            "capture_rate",
            "capture_steps",
            "alive_rate",
            "max_escape_gap_angle",
        ]
        metric_titles = {
            "eval_reward": "Eval Reward",
            "capture_rate": "Capture Rate",
            "capture_steps": "Capture Steps",
            "alive_rate": "Alive Rate",
            "max_escape_gap_angle": "Max Potential Escape Gap Angle",
        }
        bucket_style = {
            "fixed": {"label": "fixed", "color": "#1f77b4", "marker": "o"},
            "learn": {"label": "learn", "color": "#ff7f0e", "marker": "s"},
        }

        fig, axes = plt.subplots(len(metrics_order), 1, figsize=(7.5, 17.0), dpi=120)
        plot_export = {
            "episode": int(episode),
            "total_num_steps": int(total_num_steps),
            "buckets": {},
        }
        for idx, metric_name in enumerate(metrics_order):
            ax = axes[idx]
            x_ticks = set()
            for bucket_name in ("fixed", "learn"):
                if bucket_name not in bucket_to_metrics:
                    continue
                grouped = bucket_to_metrics[bucket_name]
                if grouped is None or len(grouped) == 0:
                    continue
                x_vals = sorted([int(k) for k in grouped.keys()])
                y_vals = []
                for hc in x_vals:
                    metric_val = float(grouped[int(hc)].get(metric_name, float("nan")))
                    if metric_name == "max_escape_gap_angle" and (not np.isfinite(metric_val)):
                        metric_val = 360.0
                    y_vals.append(float(metric_val))
                x_ticks.update(x_vals)
                style = bucket_style[bucket_name]
                ax.plot(
                    x_vals,
                    y_vals,
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.8,
                    markersize=4.5,
                )
                if len(x_vals) > 0:
                    if metric_name in ("eval_reward", "capture_rate", "alive_rate"):
                        best_pos = int(np.argmax(np.asarray(y_vals, dtype=np.float32)))
                    else:
                        best_pos = int(np.argmin(np.asarray(y_vals, dtype=np.float32)))
                    best_x = int(x_vals[best_pos])
                    best_y = float(y_vals[best_pos])
                    ax.scatter(
                        [best_x],
                        [best_y],
                        marker="*",
                        s=70,
                        color=style["color"],
                        zorder=4,
                    )
                    ax.annotate(
                        f"{best_y:.3f}",
                        xy=(best_x, best_y),
                        xytext=(6, 6),
                        textcoords="offset points",
                        color=style["color"],
                        fontsize=8,
                    )
                if bucket_name not in plot_export["buckets"]:
                    plot_export["buckets"][bucket_name] = {}
                plot_export["buckets"][bucket_name][metric_name] = {
                    "x": [int(x) for x in x_vals],
                    "y": [float(y) for y in y_vals],
                }
            ax.set_title(metric_titles[metric_name])
            ax.set_xlabel("Number of Hunters")
            if len(x_ticks) > 0:
                ax.set_xticks(sorted([int(x) for x in x_ticks]))
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
            if metric_name in ("capture_rate", "alive_rate"):
                ax.set_ylim(0.0, 1.0)
            if metric_name == "max_escape_gap_angle":
                ax.set_ylim(0.0, 360.0)
            ax.set_ylabel("Value")
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(loc="best")

        fig.suptitle(
            "Eval Metrics vs Hunters | episode={} | steps={}".format(int(episode), int(total_num_steps))
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
        out_path = out_dir / f"eval_hunter_bucket_metrics_ep_{int(episode)}.png"
        fig.savefig(str(out_path))
        plt.close(fig)

        # Step 3: 导出可复绘数据，便于后续跨模型分桶对比脚本直接读取。
        json_path = out_dir / f"eval_hunter_bucket_metrics_ep_{int(episode)}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(plot_export, f, ensure_ascii=False, indent=2)

    def _format_duration(self, seconds):
        """
        功能:
            将秒数格式化为易读的h/m/s字符串。
        输入:
            seconds (float): 时长（秒）。
        输出:
            str: 形如\"01h23m45s\"的字符串。
        """
        total_sec = int(max(0.0, float(seconds)))
        h = total_sec // 3600
        m = (total_sec % 3600) // 60
        s = total_sec % 60
        if h > 0:
            return f"{h:02d}h{m:02d}m{s:02d}s"
        if m > 0:
            return f"{m:02d}m{s:02d}s"
        return f"{s:02d}s"

    def _time_prefix(self):
        """
        功能:
            生成当前本地系统时间前缀字符串。
        输入:
            无。
        输出:
            str: 形如\"[2026-02-27 20:15:30]\"的时间前缀。
        """
        return f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]"

    def _print_timed(self, message):
        """
        功能:
            以统一时间前缀打印日志，便于不同阶段耗时对齐比对。
        输入:
            message (str): 原始日志内容。
        输出:
            无。
        """
        print(f"{self._time_prefix()} {str(message)}\n")

    def _print_metric_table(self, title, metrics):
        """
        功能:
            以两行对齐表格打印指标（第一行名称，第二行数值）。
        输入:
            title (str): 表格标题。
            metrics (dict): 指标名到指标值映射。
        输出:
            无。
        """
        if metrics is None or len(metrics) == 0:
            self._print_timed(f"[{str(title)}] (empty)")
            return

        keys = [str(k) for k in metrics.keys()]
        values = [str(v) for v in metrics.values()]
        widths = [max(len(k), len(v)) for k, v in zip(keys, values)]
        header = " | ".join(k.ljust(w) for k, w in zip(keys, widths))
        row = " | ".join(v.ljust(w) for v, w in zip(values, widths))

        self._print_timed(f"[{str(title)}]")
        print(header)
        print(row)
        print("")

    def _peek_fixed_metric_updates(self, eval_metrics):
        """
        功能:
            仅比较fixed桶当前指标与历史最优，不修改任何状态。
        输入:
            eval_metrics (dict): 当前fixed评估指标。
        输出:
            list[str]: 被刷新（更优）的指标名称列表。
        """
        updated_metrics = []
        if float(eval_metrics["eval_reward"]) > float(self.best_eval_metrics["reward"]):
            updated_metrics.append("reward")
        if float(eval_metrics["capture_rate"]) > float(self.best_eval_metrics["capture_rate"]):
            updated_metrics.append("capture_rate")
        if float(eval_metrics["capture_steps"]) < float(self.best_eval_metrics["capture_steps"]):
            updated_metrics.append("capture_steps")
        cur_gap = float(eval_metrics.get("max_escape_gap_angle", float("nan")))
        if np.isfinite(cur_gap) and cur_gap < float(self.best_eval_metrics["max_escape_gap_angle"]):
            updated_metrics.append("max_escape_gap_angle")
        return updated_metrics

    def _peek_bucket_metric_updates(self, bucket, eval_metrics):
        """
        功能:
            比较指定bucket当前指标与该bucket历史最优，不修改任何状态。
        输入:
            bucket (str): 指标桶名称。
            eval_metrics (dict): 当前评估指标。
        输出:
            list[str]: 被刷新（更优）的指标名称列表。
        """
        if bucket not in self.best_eval_metrics_by_bucket:
            self.best_eval_metrics_by_bucket[bucket] = {
                "eval_reward": -np.inf,
                "capture_rate": -np.inf,
                "capture_steps": np.inf,
                "max_escape_gap_angle": np.inf,
            }
        best = self.best_eval_metrics_by_bucket[bucket]
        updated_metrics = []
        if float(eval_metrics["eval_reward"]) > float(best["eval_reward"]):
            updated_metrics.append("eval_reward")
        if float(eval_metrics["capture_rate"]) > float(best["capture_rate"]):
            updated_metrics.append("capture_rate")
        if float(eval_metrics["capture_steps"]) < float(best["capture_steps"]):
            updated_metrics.append("capture_steps")
        cur_gap = float(eval_metrics.get("max_escape_gap_angle", float("nan")))
        if np.isfinite(cur_gap) and cur_gap < float(best["max_escape_gap_angle"]):
            updated_metrics.append("max_escape_gap_angle")
        return updated_metrics

    def _update_bucket_best_metrics(self, bucket, eval_metrics):
        """
        功能:
            更新指定bucket的历史最优指标，并打印更新摘要。
        输入:
            bucket (str): 指标桶名称。
            eval_metrics (dict): 当前评估指标。
        输出:
            list[str]: 实际被更新的指标名称列表。
        """
        if bucket not in self.best_eval_metrics_by_bucket:
            self.best_eval_metrics_by_bucket[bucket] = {
                "eval_reward": -np.inf,
                "capture_rate": -np.inf,
                "capture_steps": np.inf,
                "max_escape_gap_angle": np.inf,
            }
        best = self.best_eval_metrics_by_bucket[bucket]
        updated_metrics = self._peek_bucket_metric_updates(bucket, eval_metrics)
        if "eval_reward" in updated_metrics:
            best["eval_reward"] = float(eval_metrics["eval_reward"])
        if "capture_rate" in updated_metrics:
            best["capture_rate"] = float(eval_metrics["capture_rate"])
        if "capture_steps" in updated_metrics:
            best["capture_steps"] = float(eval_metrics["capture_steps"])
        if "max_escape_gap_angle" in updated_metrics:
            best["max_escape_gap_angle"] = float(eval_metrics["max_escape_gap_angle"])
        self._print_metric_table(
            "EvalSummary[{}]".format(str(bucket)),
            {
                "cur_reward": "{:.4f}".format(float(eval_metrics["eval_reward"])),
                "cur_capture_rate": "{:.4f}".format(float(eval_metrics["capture_rate"])),
                "cur_capture_steps": "{:.2f}".format(float(eval_metrics["capture_steps"])),
                "cur_max_escape_gap": (
                    "{:.4f}".format(float(eval_metrics["max_escape_gap_angle"]))
                    if np.isfinite(float(eval_metrics.get("max_escape_gap_angle", float("nan"))))
                    else "NA"
                ),
                "best_reward": "{:.4f}".format(float(best["eval_reward"])),
                "best_capture_rate": "{:.4f}".format(float(best["capture_rate"])),
                "best_capture_steps": "{:.2f}".format(float(best["capture_steps"])),
                "best_max_escape_gap": (
                    "{:.4f}".format(float(best["max_escape_gap_angle"]))
                    if np.isfinite(float(best.get("max_escape_gap_angle", float("nan"))))
                    else "NA"
                ),
                "updated": ",".join(updated_metrics) if len(updated_metrics) > 0 else "none",
            },
        )
        return updated_metrics

    def _maybe_save_best_models(self, episode, eval_metrics):
        """
        功能:
            按reward/capture_rate/capture_steps/max_escape_gap_angle四个维度保存最佳模型。
        输入:
            episode (int): 当前训练episode编号。
            eval_metrics (dict): 当前评估指标字典。
        输出:
            list[str]: 被更新的指标名称列表。
        """
        updated_metrics = []
        # Step 1: reward越大越好
        if float(eval_metrics["eval_reward"]) > float(self.best_eval_metrics["reward"]):
            self.best_eval_metrics["reward"] = float(eval_metrics["eval_reward"])
            self._save_best_snapshot("reward", episode, eval_metrics)
            updated_metrics.append("reward")

        # Step 2: capture_rate越大越好
        if float(eval_metrics["capture_rate"]) > float(self.best_eval_metrics["capture_rate"]):
            self.best_eval_metrics["capture_rate"] = float(eval_metrics["capture_rate"])
            self._save_best_snapshot("capture_rate", episode, eval_metrics)
            updated_metrics.append("capture_rate")

        # Step 3: capture_steps越小越好
        if float(eval_metrics["capture_steps"]) < float(self.best_eval_metrics["capture_steps"]):
            self.best_eval_metrics["capture_steps"] = float(eval_metrics["capture_steps"])
            self._save_best_snapshot("capture_steps", episode, eval_metrics)
            updated_metrics.append("capture_steps")

        # Step 4: max_escape_gap_angle越小越好（仅在指标有效时参与刷新）
        cur_gap = float(eval_metrics.get("max_escape_gap_angle", float("nan")))
        if np.isfinite(cur_gap) and cur_gap < float(self.best_eval_metrics["max_escape_gap_angle"]):
            self.best_eval_metrics["max_escape_gap_angle"] = float(cur_gap)
            self._save_best_snapshot("max_escape_gap_angle", episode, eval_metrics)
            updated_metrics.append("max_escape_gap_angle")

        self._print_metric_table(
            "EvalSummary[fixed]",
            {
                "episode": str(int(episode)),
                "cur_reward": "{:.4f}".format(float(eval_metrics["eval_reward"])),
                "cur_capture_rate": "{:.4f}".format(float(eval_metrics["capture_rate"])),
                "cur_capture_steps": "{:.2f}".format(float(eval_metrics["capture_steps"])),
                "cur_max_escape_gap": (
                    "{:.4f}".format(float(eval_metrics["max_escape_gap_angle"]))
                    if np.isfinite(float(eval_metrics.get("max_escape_gap_angle", float("nan"))))
                    else "NA"
                ),
                "best_reward": "{:.4f}".format(float(self.best_eval_metrics["reward"])),
                "best_capture_rate": "{:.4f}".format(float(self.best_eval_metrics["capture_rate"])),
                "best_capture_steps": "{:.2f}".format(float(self.best_eval_metrics["capture_steps"])),
                "best_max_escape_gap": (
                    "{:.4f}".format(float(self.best_eval_metrics["max_escape_gap_angle"]))
                    if np.isfinite(float(self.best_eval_metrics.get("max_escape_gap_angle", float("nan"))))
                    else "NA"
                ),
                "updated": ",".join(updated_metrics) if len(updated_metrics) > 0 else "none",
            },
        )
        return updated_metrics

    def _save_best_snapshot(self, metric_name, episode, eval_metrics):
        """
        功能:
            保存某个最优指标对应的角色模型与指标文本。
        输入:
            metric_name (str): 指标名称（reward/capture_rate/capture_steps/max_escape_gap_angle）。
            episode (int): 当前训练episode编号。
            eval_metrics (dict): 当前评估指标。
        输出:
            无。
        """
        # Step 1: 创建指标目录（run_dir/models/best_eval_{metric}）
        metric_dir = Path(self.best_dir) / f"best_eval_{str(metric_name)}"
        metric_dir.mkdir(parents=True, exist_ok=True)

        # Step 2: 保存角色模型
        for role, trainer in self.role_trainers.items():
            torch.save(trainer.policy.actor.state_dict(), str(metric_dir / f"actor_{role}.pt"))
            torch.save(trainer.policy.critic.state_dict(), str(metric_dir / f"critic_{role}.pt"))

        # Step 3: 复制本次eval的GIF到最优目录
        for old_gif in metric_dir.glob("e-*-env-*.gif"):
            old_gif.unlink(missing_ok=True)
        eval_gif_paths = sorted(Path(self.gif_dir).glob(f"e-fixed-{int(episode)}-env-*.gif"))
        if len(eval_gif_paths) == 0:
            eval_gif_paths = sorted(Path(self.gif_dir).glob(f"e-{int(episode)}-env-*.gif"))
        for gif_path in eval_gif_paths:
            shutil.copy2(str(gif_path), str(metric_dir / gif_path.name))

        # Step 3.1: 复制评估分桶图与对应可复绘数据到最优目录。
        self._copy_eval_hunter_bucket_artifacts(episode=episode, dst_dir=metric_dir)

        # Step 4: 写入最优指标文本
        with open(metric_dir / "best_info.txt", "w", encoding="utf-8") as f:
            f.write(f"metric={metric_name}\n")
            f.write(f"episode={int(episode)}\n")
            f.write(f"eval_reward={float(eval_metrics['eval_reward']):.6f}\n")
            f.write(f"capture_rate={float(eval_metrics['capture_rate']):.6f}\n")
            f.write(f"capture_steps={float(eval_metrics['capture_steps']):.6f}\n")
            f.write(f"max_escape_gap_angle={float(eval_metrics.get('max_escape_gap_angle', float('nan'))):.6f}\n")
            f.write(f"captured_episodes={int(eval_metrics.get('captured_episodes', 0))}\n")
            f.write(f"total_eval_episodes={int(eval_metrics.get('total_eval_episodes', 0))}\n")
            f.write(f"best_reward={float(self.best_eval_metrics['reward']):.6f}\n")
            f.write(f"best_capture_rate={float(self.best_eval_metrics['capture_rate']):.6f}\n")
            f.write(f"best_capture_steps={float(self.best_eval_metrics['capture_steps']):.6f}\n")
            f.write(f"best_max_escape_gap_angle={float(self.best_eval_metrics.get('max_escape_gap_angle', float('nan'))):.6f}\n")
        print(
            "[Best] metric={} updated at episode {} | reward={:.4f} | capture_rate={:.4f} | capture_steps={:.2f} | max_escape_gap={:.4f} | path={}".format(
                str(metric_name),
                int(episode),
                float(eval_metrics["eval_reward"]),
                float(eval_metrics["capture_rate"]),
                float(eval_metrics["capture_steps"]),
                float(eval_metrics.get("max_escape_gap_angle", float("nan"))),
                str(metric_dir),
            )
        )

    def _copy_eval_hunter_bucket_artifacts(self, episode, dst_dir):
        """
        功能:
            将指定episode的hunter分桶指标图及其可复绘JSON数据复制到目标目录。
        输入:
            episode (int): 评估episode编号。
            dst_dir (str | Path): 目标目录。
        输出:
            无。
        """
        # Step 1: 准备路径与通配模式。
        dst = Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        png_name = f"eval_hunter_bucket_metrics_ep_{int(episode)}.png"
        json_name = f"eval_hunter_bucket_metrics_ep_{int(episode)}.json"
        png_src = Path(self.gif_dir) / png_name
        json_src = Path(self.gif_dir) / json_name

        # Step 2: 优先复制同episode文件；不存在时回退到最新同前缀文件。
        if png_src.exists():
            shutil.copy2(str(png_src), str(dst / png_name))
        else:
            png_candidates = sorted(Path(self.gif_dir).glob("eval_hunter_bucket_metrics_ep_*.png"))
            if len(png_candidates) > 0:
                shutil.copy2(str(png_candidates[-1]), str(dst / png_candidates[-1].name))

        if json_src.exists():
            shutil.copy2(str(json_src), str(dst / json_name))
        else:
            json_candidates = sorted(Path(self.gif_dir).glob("eval_hunter_bucket_metrics_ep_*.json"))
            if len(json_candidates) > 0:
                shutil.copy2(str(json_candidates[-1]), str(dst / json_candidates[-1].name))

    def _load_models_from_dir(self, model_dir):
        """
        功能:
            从指定目录加载各角色actor/critic参数到当前runner。
        输入:
            model_dir (str | Path): 模型目录，包含actor_{role}.pt与critic_{role}.pt。
        输出:
            bool: True表示全部角色模型加载成功；False表示存在缺失文件。
        """
        # Step 1: 统一目录对象并逐角色检查文件存在性
        load_dir = Path(model_dir)
        for role in self.role_trainers.keys():
            actor_path = load_dir / f"actor_{role}.pt"
            critic_path = load_dir / f"critic_{role}.pt"
            if (not actor_path.exists()) or (not critic_path.exists()):
                print(
                    "[FinalEval] skip {} due to missing model files for role={} (actor_exists={}, critic_exists={})".format(
                        str(load_dir),
                        str(role),
                        bool(actor_path.exists()),
                        bool(critic_path.exists()),
                    )
                )
                return False

        # Step 2: 逐角色加载权重并切换到eval模式
        for role, trainer in self.role_trainers.items():
            actor_state = torch.load(str(load_dir / f"actor_{role}.pt"), map_location=self.device)
            critic_state = torch.load(str(load_dir / f"critic_{role}.pt"), map_location=self.device)
            trainer.policy.actor.load_state_dict(actor_state)
            trainer.policy.critic.load_state_dict(critic_state)
            trainer.prep_rollout()
        return True

    @torch.no_grad()
    def _final_eval_saved_best_models(self, total_num_steps, episode, model_glob=None):
        """
        功能:
            训练完成后重载models下可用模型目录，分别在fixed/target_learn上评估并强制保存GIF到各自res目录。
        输入:
            total_num_steps (int): 评估记录使用的总步数。
            episode (int): 训练完成时的episode编号。
            model_glob (str | None): 仅评估匹配该glob的模型目录（相对models目录）。
        输出:
            无。
        """
        # Step 1: 收集待评估模型目录（优先包含models根目录，再包含best_eval_*目录）
        candidate_dirs = []
        root_model_dir = Path(self.save_dir)
        if root_model_dir.exists() and (model_glob is None):
            candidate_dirs.append(root_model_dir)
        glob_pattern = "best_eval_*" if model_glob is None else str(model_glob)
        candidate_dirs.extend(sorted(Path(self.best_dir).glob(glob_pattern)))

        model_dirs = []
        for model_dir in candidate_dirs:
            if self._load_models_from_dir(model_dir):
                model_dirs.append(model_dir)

        if len(model_dirs) == 0:
            print("[FinalEval] no valid model directories found under {}".format(self.best_dir))
            return

        print("[FinalEval] start evaluating {} model directories".format(len(model_dirs)))

        # Step 2: 逐目录加载模型并在fixed/target_learn上评估，GIF输出到目录/res
        for model_dir in model_dirs:
            self._load_models_from_dir(model_dir)
            res_dir = Path(model_dir) / "res"
            res_dir.mkdir(parents=True, exist_ok=True)
            for old_gif in res_dir.glob("e-*-env-*.gif"):
                old_gif.unlink(missing_ok=True)

            self.eval(
                total_num_steps=total_num_steps,
                episode=episode,
                eval_envs=self.eval_envs,
                bucket="fixed",
                save_gifs=True,
                record_logs=False,
                gif_output_dir=res_dir,
            )
            if self.eval_envs_target_learn is not None:
                self.eval(
                    total_num_steps=total_num_steps,
                    episode=episode,
                    eval_envs=self.eval_envs_target_learn,
                    bucket="target_learn",
                    save_gifs=True,
                    record_logs=False,
                    gif_output_dir=res_dir,
                )
            print("[FinalEval] finished {} -> {}".format(str(model_dir), str(res_dir)))
