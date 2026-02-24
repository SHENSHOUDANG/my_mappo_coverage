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

import numpy as np
import torch
import imageio.v2 as imageio
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
        self.device = runner_cfg["device"]
        self.run_dir = runner_cfg["run_dir"]
        self.num_agents = int(runner_cfg["num_agents"])

        # Step 2: 读取训练主参数
        self.num_hunters = int(self.cfg.env.num_hunters)
        self.target_index = self.num_agents - 1
        self.target_trainable = str(self.cfg.env.target_policy_source).lower() == "train"

        self.episode_length = int(self.cfg.env.episode_length)
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

        # Step 3: 创建日志/模型目录
        self.log_dir = str(self.run_dir / "logs")
        self.save_dir = str(self.run_dir / "models")
        self.gif_dir = str(self.run_dir / "eval_gifs")
        self.best_dir = str(self.run_dir / "best_models")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.gif_dir, exist_ok=True)
        os.makedirs(self.best_dir, exist_ok=True)
        self.writter = SummaryWriter(self.log_dir)

        self.log_csv_path = str(self.run_dir / "log.csv")
        self.eval_csv_path = str(self.run_dir / "eval.csv")
        self._init_csv_files()
        self.best_metrics = {
            "reward": -np.inf,
            "capture_rate": -np.inf,
            "capture_steps": np.inf,
        }

        # Step 4: 仅在算法组件初始化时构建flat args
        self.flat_args = self._build_flat_args_for_algorithm()

        # Step 5: 导入MAPPO算法组件
        from algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
        from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        # Step 6: 角色定义与受控智能体集合
        self.agent_role = {aid: "hunter" for aid in range(self.num_hunters)}
        self.agent_role[self.target_index] = "target"
        self.controlled_agents = list(range(self.num_hunters))
        if self.target_trainable:
            self.controlled_agents.append(self.target_index)

        # Step 7: 构建角色级别policy与trainer
        self.role_policies = {}
        self.role_trainers = {}
        role_represent_agent = {"hunter": 0}
        if self.target_trainable:
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
        args.num_agents = int(self.cfg.env.num_hunters) + int(self.cfg.env.num_explorers) + 1
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

        # Step 3: episode主循环
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for role_name in self.role_trainers.keys():
                    self.role_trainers[role_name].policy.lr_decay(episode, episodes)

            do_eval_this_episode = self.use_eval and episode % max(1, self.eval_interval) == 0
            train_frames = []
            episode_hunter_reward = 0.0
            episode_target_reward = 0.0
            last_infos = None

            for step in range(self.episode_length):
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
                episode_hunter_reward += float(np.sum(rewards[:, : self.num_hunters, 0]))
                episode_target_reward += float(np.sum(rewards[:, self.target_index, 0]))
                last_infos = infos

                if do_eval_this_episode:
                    frame_batch = self.envs.render(mode="rgb_array", title=f"Train Episode {int(episode)}")
                    if isinstance(frame_batch, np.ndarray) and frame_batch.shape[0] > 0:
                        train_frames.append(frame_batch[0].copy())

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
            self._append_log_csv(
                episode=episode,
                total_num_steps=total_num_steps,
                hunter_alive_mean=hunter_alive_mean,
                target_alive_mean=target_alive_mean,
                hunter_reward_mean=episode_hunter_reward
                / max(1, self.n_rollout_threads * self.num_hunters),
                target_reward_mean=episode_target_reward / max(1, self.n_rollout_threads),
            )

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.env_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start + 1e-6)),
                    )
                )
                self.log_train(train_infos, total_num_steps)

            # eval + train gif
            if do_eval_this_episode:
                train_gif_path = Path(self.gif_dir) / f"train_{int(episode)}.gif"
                print(
                    f"\t Save GIF to {train_gif_path}\n"
                )
                self._save_gif(train_frames, train_gif_path)
                self.eval(total_num_steps, episode)

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
        obs = self.envs.reset()

        # Step 2: 计算集中式critic使用的share_obs
        share_obs_all = np.array([list(chain(*o)) for o in obs], dtype=np.float32)

        # Step 3: 写入每个受控agent的buffer[0]
        for agent_id in self.controlled_agents:
            if self.use_centralized_V:
                share_obs = share_obs_all
            else:
                share_obs = np.array(list(obs[:, agent_id]), dtype=np.float32)
            self.buffers[agent_id].share_obs[0] = share_obs.copy()
            self.buffers[agent_id].obs[0] = np.array(list(obs[:, agent_id]), dtype=np.float32).copy()

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
    def eval(self, total_num_steps, episode):
        """
        功能:
            执行周期性evaluation（每个eval_env运行1个episode），
            记录reward/捕获率/捕获步数，并保存每个eval_env的GIF。
        输入:
            total_num_steps (int): 当前累计环境步数。
            episode (int): 当前训练episode编号。
        输出:
            dict: evaluation指标字典。
        """
        # Step 1: 准备评估状态
        if self.eval_envs is None:
            return {
                "eval_reward": 0.0,
                "capture_rate": 0.0,
                "capture_steps": float(self.episode_length),
                "captured_episodes": 0,
                "total_eval_episodes": 0,
            }

        eval_obs = self.eval_envs.reset()
        n_env = int(eval_obs.shape[0])
        act_dim = int(self.eval_envs.action_space[0].shape[0])

        eval_rnn_states = {
            aid: np.zeros((n_env, self.recurrent_N, self.hidden_size), dtype=np.float32)
            for aid in self.controlled_agents
        }
        eval_masks = {
            aid: np.ones((n_env, 1), dtype=np.float32)
            for aid in self.controlled_agents
        }

        env_episode_rewards = np.zeros(n_env, dtype=np.float32)
        env_captured = np.zeros(n_env, dtype=bool)
        env_capture_step = np.full(n_env, -1, dtype=np.int32)
        env_finished = np.zeros(n_env, dtype=bool)
        eval_frames = [[] for _ in range(n_env)]

        # Step 2: 记录初始帧
        frame_batch = self.eval_envs.render(mode="rgb_array", title=f"Eval Episode {int(episode)}")
        if isinstance(frame_batch, np.ndarray):
            for env_i in range(min(n_env, frame_batch.shape[0])):
                eval_frames[env_i].append(frame_batch[env_i].copy())

        # Step 3: rollout一个评估episode长度
        for eval_step in range(self.episode_length):
            eval_actions_env = np.zeros((n_env, self.num_agents, act_dim), dtype=np.float32)

            for agent_id in self.controlled_agents:
                role = self.agent_role[agent_id]
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

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

            for env_i in range(n_env):
                if env_finished[env_i]:
                    continue
                env_episode_rewards[env_i] += float(
                    np.sum(eval_rewards[env_i, : self.num_hunters, 0]) / max(1, self.num_hunters)
                )
                if (not env_captured[env_i]) and any(
                    bool(agent_info.get("captured", False)) for agent_info in eval_infos[env_i]
                ):
                    env_captured[env_i] = True
                    env_capture_step[env_i] = int(eval_step + 1)
                if bool(np.all(eval_dones[env_i])):
                    env_finished[env_i] = True

            frame_batch = self.eval_envs.render(mode="rgb_array", title=f"Eval Episode {int(episode)}")
            if isinstance(frame_batch, np.ndarray):
                for env_i in range(min(n_env, frame_batch.shape[0])):
                    if not env_finished[env_i]:
                        eval_frames[env_i].append(frame_batch[env_i].copy())

            for agent_id in self.controlled_agents:
                done_mask = eval_dones[:, agent_id].astype(bool)
                eval_masks[agent_id] = np.ones((n_env, 1), dtype=np.float32)
                eval_masks[agent_id][done_mask] = 0.0
                eval_rnn_states[agent_id][done_mask] = 0.0

            if bool(np.all(env_finished)):
                break

        # Step 4: 汇总评估指标
        total_eval_episodes = int(n_env)
        captured_episodes = int(np.sum(env_captured))
        capture_rate = float(captured_episodes / max(1, total_eval_episodes))
        eval_reward = float(np.mean(env_episode_rewards)) if total_eval_episodes > 0 else 0.0

        captured_steps = [int(env_capture_step[i]) for i in range(n_env) if env_captured[i] and env_capture_step[i] > 0]
        capture_steps = (
            float(np.mean(captured_steps)) if len(captured_steps) > 0 else float(self.episode_length)
        )

        eval_metrics = {
            "eval_reward": eval_reward,
            "capture_rate": capture_rate,
            "capture_steps": capture_steps,
            "captured_episodes": int(captured_episodes),
            "total_eval_episodes": int(total_eval_episodes),
        }

        print(
            "[Eval] episode={}, reward={:.4f}, capture_rate={:.4f}, capture_steps={:.2f}".format(
                int(episode),
                eval_reward,
                capture_rate,
                capture_steps,
            )
        )

        # Step 5: 记录TB与CSV
        self.writter.add_scalars("eval/eval_reward", {"eval/eval_reward": eval_reward}, total_num_steps)
        self.writter.add_scalars("eval/capture_rate", {"eval/capture_rate": capture_rate}, total_num_steps)
        self.writter.add_scalars("eval/capture_steps", {"eval/capture_steps": capture_steps}, total_num_steps)

        self._append_eval_csv(
            episode=episode,
            total_num_steps=total_num_steps,
            eval_reward=eval_reward,
            capture_rate=capture_rate,
            capture_steps=capture_steps,
            captured_episodes=captured_episodes,
            total_eval_episodes=total_eval_episodes,
        )

        # Step 6: 保存每个eval_env GIF
        for env_i in range(n_env):
            gif_path = Path(self.gif_dir) / f"e-{int(episode)}-env-{int(env_i)}.gif"
            self._save_gif(eval_frames[env_i], gif_path)

        # Step 7: 按指标保存最佳模型
        self._maybe_save_best_models(episode, eval_metrics)
        return eval_metrics

    def _save_gif(self, frames, gif_path):
        """
        功能:
            将帧序列采样为3秒GIF并写入磁盘。
        输入:
            frames (list[np.ndarray]): RGB帧列表。
            gif_path (Path): GIF输出路径。
        输出:
            无。
        """
        # Step 1: 空帧保护
        if frames is None or len(frames) == 0:
            return

        # Step 2: 统一为3秒（10fps，共30帧）
        fps = 10
        duration_sec = 3
        frame_count = max(1, fps * duration_sec)
        idx = np.linspace(0, len(frames) - 1, frame_count).astype(np.int32)
        sampled_frames = [frames[i] for i in idx]

        # Step 3: 写出GIF
        imageio.mimsave(str(gif_path), sampled_frames, duration=1.0 / float(fps), loop=0)

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
                    "eval_reward",
                    "capture_rate",
                    "capture_steps",
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
                    int(self.num_hunters),
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
        eval_reward,
        capture_rate,
        capture_steps,
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
                    float(eval_reward),
                    float(capture_rate),
                    float(capture_steps),
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
            hunter_flags = [
                float(agent_info.get("alive", False))
                for agent_info in env_infos[: self.num_hunters]
            ]
            hunter_alive.append(float(np.mean(hunter_flags)) if hunter_flags else 0.0)
            target_alive.append(float(env_infos[self.target_index].get("alive", False)))

        if len(hunter_alive) == 0:
            return 0.0, 0.0
        return float(np.mean(hunter_alive)), float(np.mean(target_alive))

    def _maybe_save_best_models(self, episode, eval_metrics):
        """
        功能:
            按reward/capture_rate/capture_steps三个维度保存最佳模型。
        输入:
            episode (int): 当前训练episode编号。
            eval_metrics (dict): 当前评估指标字典。
        输出:
            无。
        """
        # Step 1: reward越大越好
        if float(eval_metrics["eval_reward"]) > float(self.best_metrics["reward"]):
            self.best_metrics["reward"] = float(eval_metrics["eval_reward"])
            self._save_best_snapshot("reward", episode, eval_metrics)

        # Step 2: capture_rate越大越好
        if float(eval_metrics["capture_rate"]) > float(self.best_metrics["capture_rate"]):
            self.best_metrics["capture_rate"] = float(eval_metrics["capture_rate"])
            self._save_best_snapshot("capture_rate", episode, eval_metrics)

        # Step 3: capture_steps越小越好
        if float(eval_metrics["capture_steps"]) < float(self.best_metrics["capture_steps"]):
            self.best_metrics["capture_steps"] = float(eval_metrics["capture_steps"])
            self._save_best_snapshot("capture_steps", episode, eval_metrics)

    def _save_best_snapshot(self, metric_name, episode, eval_metrics):
        """
        功能:
            保存某个最优指标对应的角色模型与指标文本。
        输入:
            metric_name (str): 指标名称（reward/capture_rate/capture_steps）。
            episode (int): 当前训练episode编号。
            eval_metrics (dict): 当前评估指标。
        输出:
            无。
        """
        # Step 1: 创建指标目录
        metric_dir = Path(self.best_dir) / str(metric_name)
        metric_dir.mkdir(parents=True, exist_ok=True)

        # Step 2: 保存角色模型
        for role, trainer in self.role_trainers.items():
            torch.save(trainer.policy.actor.state_dict(), str(metric_dir / f"actor_{role}.pt"))
            torch.save(trainer.policy.critic.state_dict(), str(metric_dir / f"critic_{role}.pt"))

        # Step 3: 写入最优指标文本
        with open(metric_dir / "best_info.txt", "w", encoding="utf-8") as f:
            f.write(f"metric={metric_name}\n")
            f.write(f"episode={int(episode)}\n")
            f.write(f"eval_reward={float(eval_metrics['eval_reward']):.6f}\n")
            f.write(f"capture_rate={float(eval_metrics['capture_rate']):.6f}\n")
            f.write(f"capture_steps={float(eval_metrics['capture_steps']):.6f}\n")
            f.write(f"captured_episodes={int(eval_metrics.get('captured_episodes', 0))}\n")
            f.write(f"total_eval_episodes={int(eval_metrics.get('total_eval_episodes', 0))}\n")
