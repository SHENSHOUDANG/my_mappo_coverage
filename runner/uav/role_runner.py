"""
Role-based Runner for Multi-UAV Pursuit.

核心特性:
1) 读取分层配置（merged_cfg），不依赖全局扁平化参数。
2) 仅在初始化算法组件（Policy/Trainer/Buffer）时构建flat args。
3) 同角色共享策略：当前hunter共享一个policy；target可选是否训练。
"""

import os
import time
from itertools import chain
import argparse
import csv
import copy

import numpy as np
import torch
from tensorboardX import SummaryWriter

from utils.separated_buffer import SeparatedReplayBuffer


def _t2n(x):
    """
    输入:
        x (torch.Tensor): 任意形状张量。
    输出:
        np.ndarray: 转到CPU并detach后的numpy数组。
    物理意义:
        将网络输出从Torch计算图中分离，供环境交互与buffer写入使用。
    """
    return x.detach().cpu().numpy()


class RoleBasedRunner(object):
    """
    UAV Pursuit任务专用训练执行器。
    """

    def __init__(self, runner_cfg, merged_cfg):
        """
        输入:
            runner_cfg (dict):
                - envs: 训练环境向量封装（DummyVecEnv）
                - eval_envs: 评估环境向量封装或None
                - device: torch.device
                - run_dir: 结果输出目录Path
                - num_agents: 智能体总数（int）
            merged_cfg (EasyDict): 分层配置对象。
        输出:
            无（初始化runner内部状态）。
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
        self.val_episodes = int(self.cfg.eval.val_episodes)
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
        self.gif_dir = str(self.run_dir / "val_gifs")
        self.best_dir = str(self.run_dir / "best_models")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.gif_dir, exist_ok=True)
        os.makedirs(self.best_dir, exist_ok=True)
        self.writter = SummaryWriter(self.log_dir)

        self.log_csv_path = str(self.run_dir / "log.csv")
        self.val_csv_path = str(self.run_dir / "val.csv")
        self._init_csv_files()
        self.best_metrics = {
            "reward": -np.inf,
            "capture_rate": -np.inf,
            "capture_steps": np.inf,
        }

        # Step 4: 仅在算法组件初始化时构建flat args
        self.flat_args = self._build_flat_args_for_algorithm()

        # Step 5: 导入MAPPO算法组件（不修改算法实现）
        from algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
        from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        # Step 6: 角色定义与受控智能体集合
        self.agent_role = {aid: "hunter" for aid in range(self.num_hunters)}
        self.agent_role[self.target_index] = "target"
        self.controlled_agents = list(range(self.num_hunters))
        if self.target_trainable:
            self.controlled_agents.append(self.target_index)

        # Step 7: 构建“角色级别”的policy与trainer
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

        # Step 8: 构建“智能体级别”的buffer（同角色可共享同一个trainer）
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
        输入:
            无（读取 self.cfg 分层配置）。
        输出:
            argparse.Namespace: 仅供算法组件/Buffer构造使用的flat参数。
        物理意义:
            作为不改动MAPPO算法前提下的参数适配层。
        """
        # Step 1: 从原始config parser创建默认flat args
        args = argparse.Namespace()

        # Step 2: 写入算法训练所需字段（白名单映射）
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

        args.use_linear_lr_decay = self.cfg.schedule.use_linear_lr_decay
        args.save_interval = self.cfg.logging.save_interval
        args.log_interval = self.cfg.logging.log_interval
        args.use_eval = self.cfg.eval.use_eval
        args.eval_interval = self.cfg.eval.val_interval
        args.eval_episodes = self.cfg.eval.val_episodes
        args.save_gifs = self.cfg.render.save_gifs
        args.use_render = self.cfg.render.use_render
        args.render_episodes = self.cfg.render.render_episodes
        args.ifi = self.cfg.render.ifi
        args.model_dir = self.cfg.pretrained.model_dir
        return args

    def run(self):
        """
        功能:
            执行主训练循环，并在训练过程中记录CSV日志、周期性执行validation、
            保存普通检查点与最佳指标模型。
        输入:
            无。
        输出:
            无（执行完整训练流程）。
        """
        # Step 1: 预填充buffer第0步观测
        self.warmup()

        # Step 2: 计算总训练episode数
        total_episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        start_time = time.time()

        # Step 3: episode循环
        for episode in range(total_episodes):
            if self.use_linear_lr_decay:
                for role_name, trainer in self.role_trainers.items():
                    trainer.policy.lr_decay(episode, total_episodes)

            episode_hunter_reward = 0.0
            episode_target_reward = 0.0
            last_infos = None

            # Step 4: rollout采样
            for step in range(self.episode_length):
                collected = self.collect(step)
                obs, rewards, dones, infos = self.envs.step(collected["actions_env"])
                self.insert(obs, rewards, dones, collected)
                episode_hunter_reward += float(np.sum(rewards[:, : self.num_hunters, 0]))
                episode_target_reward += float(np.sum(rewards[:, self.target_index, 0]))
                last_infos = infos

            # Step 5: return计算与参数更新
            self.compute()
            train_infos = self.train()

            # Step 6: 保存与日志
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

            if episode % self.save_interval == 0 or episode == total_episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                fps = int(total_num_steps / (time.time() - start_time + 1e-6))
                print(
                    f"\nEnv {self.env_name} Algo {self.algorithm_name} Exp {self.experiment_name} "
                    f"Episode {episode}/{total_episodes} Steps {total_num_steps}/{self.num_env_steps} FPS {fps}\n"
                )
                self.log_train(train_infos, total_num_steps)

            if (episode + 1) % max(1, self.eval_interval) == 0:
                val_metrics = self.run_validation(episode, total_num_steps)
                self._append_val_csv(
                    episode=episode,
                    total_num_steps=total_num_steps,
                    val_reward=val_metrics["val_reward"],
                    capture_rate=val_metrics["capture_rate"],
                    capture_steps=val_metrics["capture_steps"],
                )
                self._maybe_save_best_models(episode, val_metrics)
                print(
                    f"[Validation] episode={episode}, "
                    f"reward={val_metrics['val_reward']:.4f}, "
                    f"capture_rate={val_metrics['capture_rate']:.4f}, "
                    f"capture_steps={val_metrics['capture_steps']:.2f}"
                )

    def warmup(self):
        """
        输入:
            无。
        输出:
            无（将初始观测写入每个受控agent的buffer[0]）。
        """
        # Step 1: 环境reset，获取初始观测
        obs = self.envs.reset()  # shape: [n_env, n_agent, obs_dim]

        # Step 2: 计算集中式critic需要的share_obs
        share_obs_all = np.array([list(chain(*o)) for o in obs], dtype=np.float32)

        # Step 3: 写入每个受控agent的buffer第0步
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
        输入:
            step (int): 当前episode内时间步索引。
        输出:
            dict:
                - actions_env (np.ndarray): shape=[n_env, n_agent, act_dim]，送入环境。
                - values/actions/action_log_probs/rnn_states/rnn_states_critic:
                  以agent_id为key的中间张量（numpy）。
        """
        # Step 1: 准备环境动作容器（未训练agent默认零动作）
        act_dim = self.envs.action_space[0].shape[0]
        actions_env = np.zeros(
            (self.n_rollout_threads, self.num_agents, act_dim), dtype=np.float32
        )

        # Step 2: 对每个受控agent，用其角色共享policy采样动作
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

        return {
            "actions_env": actions_env,
            "values": values,
            "actions": actions,
            "action_log_probs": action_log_probs,
            "rnn_states": rnn_states,
            "rnn_states_critic": rnn_states_critic,
        }

    def insert(self, obs, rewards, dones, collected):
        """
        输入:
            obs (np.ndarray): shape=[n_env, n_agent, obs_dim]。
            rewards (np.ndarray): shape=[n_env, n_agent, 1]。
            dones (np.ndarray): shape=[n_env, n_agent]。
            collected (dict): collect阶段保存的网络输出与动作。
        输出:
            无（写入replay buffer）。
        """
        # Step 1: 先计算集中式share_obs
        share_obs_all = np.array([list(chain(*o)) for o in obs], dtype=np.float32)

        # Step 2: 按受控agent逐个写buffer
        for agent_id in self.controlled_agents:
            done_mask = dones[:, agent_id].astype(bool)
            rnn_state = collected["rnn_states"][agent_id].copy()
            rnn_state_critic = collected["rnn_states_critic"][agent_id].copy()

            # 子步骤: done样本的RNN状态归零
            rnn_state[done_mask] = np.zeros(
                (done_mask.sum(), self.recurrent_N, self.hidden_size), dtype=np.float32
            )
            rnn_state_critic[done_mask] = np.zeros(
                (done_mask.sum(), self.recurrent_N, self.hidden_size), dtype=np.float32
            )

            # 子步骤: 构造mask
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
                collected["actions"][agent_id],
                collected["action_log_probs"][agent_id],
                collected["values"][agent_id],
                rewards[:, agent_id],
                masks,
            )

    @torch.no_grad()
    def compute(self):
        """
        输入:
            无。
        输出:
            无（为每个受控agent计算GAE/returns）。
        """
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
        输入:
            无。
        输出:
            dict: 训练日志字典，key为 role_agent 形式。
        """
        # Step 1: 对每个受控agent执行一次PPO更新
        # 注意: 同角色共享trainer，因此参数会被该角色多个agent样本连续更新。
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
        输入:
            无。
        输出:
            无（按角色保存actor/critic参数）。
        """
        for role, trainer in self.role_trainers.items():
            torch.save(trainer.policy.actor.state_dict(), os.path.join(self.save_dir, f"actor_{role}.pt"))
            torch.save(trainer.policy.critic.state_dict(), os.path.join(self.save_dir, f"critic_{role}.pt"))

    def log_train(self, train_infos, total_num_steps):
        """
        输入:
            train_infos (dict): train()返回的日志字典。
            total_num_steps (int): 当前累计环境步数。
        输出:
            无（写入tensorboard）。
        """
        # Step 1: 展开每个tag下的指标并写TensorBoard
        for tag, info in train_infos.items():
            for metric_name, metric_value in info.items():
                tb_key = f"{tag}/{metric_name}"
                self.writter.add_scalars(tb_key, {tb_key: metric_value}, total_num_steps)
