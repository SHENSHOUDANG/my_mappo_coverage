"""
Swarm simulation GUI for search + assignment + pursuit pipeline.

设计目标:
1) 提供单文件GUI入口，支持任务创建、预规划、下发执行、开始/暂停/重置。
2) 按 hybrid_decision_method 描述实现搜索-分配-执行主流程。
3) 复用 env_uav_pursuit 中 Target/Hunter 运动与策略 step/select_action 逻辑。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

project_root = Path(__file__).resolve().parents[1]
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from envs.env_uav_pursuit import ExplorerAgent, HunterAgent, TargetAgent, UAVPursuitEnv
from utils.util import load_config

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    tk = None
    ttk = None
    messagebox = None

try:
    from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy
    import torch
except Exception:
    RMAPPOPolicy = None
    torch = None


@dataclass
class AssignmentWeights:
    """
    功能:
        定义目标分配匹配成本的可调权重。
    输入:
        无（字段由GUI/配置赋值）。
    输出:
        无。
    """

    distance_weight: float = 1.0
    value_weight: float = 2.0
    endurance_weight: float = 1.0
    switch_weight: float = 0.5
    max_assign_dist: float = 250.0


@dataclass
class ExplorerRuntime:
    """
    功能:
        维护Explorer运行时状态。
    输入:
        agent (ExplorerAgent): Explorer对象。
    输出:
        无。
    """

    agent: ExplorerAgent
    state: str = "SEARCH"
    assigned_target: int = -1
    path: List[np.ndarray] = field(default_factory=list)
    path_index: int = 0
    resume_path_index: int = 0


@dataclass
class HunterRuntime:
    """
    功能:
        维护Hunter运行时状态。
    输入:
        agent (HunterAgent): Hunter对象。
    输出:
        无。
    """

    agent: HunterAgent
    standby_mode: str = "split"
    assigned_target: int = -1
    standby_path: List[np.ndarray] = field(default_factory=list)
    standby_index: int = 0
    standby_direction: int = 1
    standby_speed: float = 0.0
    last_target: int = -1
    zone_explorer: int = -1
    zone_offset: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


@dataclass
class TargetRuntime:
    """
    功能:
        维护Target运行时状态与任务池信息。
    输入:
        agent (TargetAgent): Target对象。
    输出:
        无。
    """

    agent: TargetAgent
    value: float = 1.0
    required_hunters: int = 1
    alive: bool = True
    in_pool: bool = False
    discovered: bool = False
    assigned_explorer: int = -1
    assigned_hunters: List[int] = field(default_factory=list)
    pursuit_started: bool = False
    assign_step: int = -1
    last_seen_step: int = -1
    last_seen_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    last_seen_vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


@dataclass
class MissionConfig:
    """
    功能:
        维护任务级配置参数。
    输入:
        无（字段由GUI与配置更新）。
    输出:
        无。
    """

    world_size: float = 100.0
    dt: float = 0.1
    max_steps: int = 3000
    hunters: int = 6
    explorers: int = 3
    targets: int = 5
    overlap_rate: float = 0.2
    hunters_wait_mode: str = "split"
    explorer_track_speed_scale: float = 1.2
    loss_timeout_steps: int = 40


@dataclass
class PursuitRuntime:
    """
    功能:
        维护单个Target追捕子任务的子环境与索引映射。
    输入:
        env (UAVPursuitEnv): 子任务环境。
        hunter_ids (List[int]): 全局Hunter ID按子环境局部顺序映射列表。
    输出:
        无。
    """

    env: UAVPursuitEnv
    hunter_ids: List[int] = field(default_factory=list)
    started: bool = False


class MinCostMatcher:
    """
    功能:
        提供最小成本匹配（匈牙利思想的最小费用增广实现）。
    输入:
        无。
    输出:
        无。
    """

    @staticmethod
    def solve(cost: np.ndarray) -> List[Tuple[int, int]]:
        """
        功能:
            计算最小成本匹配结果（支持矩形代价矩阵）。
        输入:
            cost (np.ndarray): shape=(n,m) 的代价矩阵。
        输出:
            List[Tuple[int,int]]: 匹配对(row_idx, col_idx)。
        """
        if cost.size == 0:
            return []
        arr = np.asarray(cost, dtype=np.float64)
        n, m = arr.shape
        transposed = False
        if n > m:
            arr = arr.T
            n, m = arr.shape
            transposed = True

        u = np.zeros(n + 1, dtype=np.float64)
        v = np.zeros(m + 1, dtype=np.float64)
        p = np.zeros(m + 1, dtype=np.int32)
        way = np.zeros(m + 1, dtype=np.int32)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = np.full(m + 1, np.inf, dtype=np.float64)
            used = np.zeros(m + 1, dtype=bool)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = np.inf
                j1 = 0
                for j in range(1, m + 1):
                    if used[j]:
                        continue
                    cur = arr[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(0, m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break

            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        pairs: List[Tuple[int, int]] = []
        for j in range(1, m + 1):
            if p[j] == 0:
                continue
            row = int(p[j] - 1)
            col = int(j - 1)
            if row < n:
                if transposed:
                    pairs.append((col, row))
                else:
                    pairs.append((row, col))
        return pairs


class LearnTargetActor:
    """
    功能:
        可选learn-target actor封装；无模型时自动回退零动作。
    输入:
        cfg (EasyDict): 合并配置。
        actor_path (Optional[str]): actor权重路径。
        obs_dim (int): 观测维度。
    输出:
        无。
    """

    def __init__(self, cfg, actor_path: Optional[str], obs_dim: int):
        self.enabled = False
        self.policy = None
        self.recurrent_N = 1
        self.hidden_size = 1
        self.rnn_states: Dict[int, np.ndarray] = {}

        if actor_path is None:
            return
        if RMAPPOPolicy is None or torch is None:
            return

        try:
            flat_args = self._build_flat_args_from_cfg(cfg)
            from gymnasium import spaces

            obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(int(obs_dim),), dtype=np.float32)
            act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.policy = RMAPPOPolicy(flat_args, obs_space, obs_space, act_space, device=device)
            ckpt = torch.load(str(actor_path), map_location=device)
            self.policy.actor.load_state_dict(ckpt)
            self.policy.actor.eval()
            self.recurrent_N = int(flat_args.recurrent_N)
            self.hidden_size = int(flat_args.hidden_size)
            self.enabled = True
        except Exception:
            self.enabled = False

    def _build_flat_args_from_cfg(self, merged_cfg):
        """
        功能:
            将分层配置映射为策略初始化所需扁平参数。
        输入:
            merged_cfg (EasyDict): 合并配置。
        输出:
            argparse.Namespace: 算法参数。
        """
        from runner.uav.role_runner import RoleBasedRunner

        class _Dummy(object):
            pass

        dummy = _Dummy()
        dummy.cfg = merged_cfg
        return RoleBasedRunner._build_flat_args_for_algorithm(dummy)

    def reset_target(self, target_id: int):
        """
        功能:
            重置目标的RNN状态。
        输入:
            target_id (int): 目标ID。
        输出:
            无。
        """
        self.rnn_states[int(target_id)] = np.zeros(
            (1, self.recurrent_N, self.hidden_size), dtype=np.float32
        )

    def act(self, target_id: int, obs: np.ndarray) -> np.ndarray:
        """
        功能:
            对learn目标执行一次策略前向推理。
        输入:
            target_id (int): 目标ID。
            obs (np.ndarray): shape=(obs_dim,)。
        输出:
            np.ndarray: shape=(2,) 归一化动作。
        """
        if (not self.enabled) or self.policy is None:
            return np.zeros(2, dtype=np.float32)
        tid = int(target_id)
        if tid not in self.rnn_states:
            self.reset_target(tid)
        rnn_state = self.rnn_states[tid]
        obs_batch = np.asarray(obs, dtype=np.float32)[None, :]
        masks = np.ones((1, 1), dtype=np.float32)
        with torch.no_grad():
            action_t, next_rnn = self.policy.act(obs_batch, rnn_state, masks, deterministic=True)
        self.rnn_states[tid] = next_rnn.detach().cpu().numpy().astype(np.float32)
        action = action_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return np.clip(action, -1.0, 1.0)


class SwarmSimulationCore:
    """
    功能:
        实现搜索-分配-追捕全流程的仿真核心。
    输入:
        cfg (EasyDict): 合并配置。
        seed (Optional[int]): 随机种子。
        target_actor_path (Optional[str]): learn-target actor路径。
    输出:
        无。
    """

    def __init__(self, cfg, seed: Optional[int] = None, target_actor_path: Optional[str] = None):
        self.cfg = cfg
        self.rng = np.random.RandomState(int(cfg.exp.seed if seed is None else seed))

        env_cfg = cfg.env
        multi_env_cfg = None
        if hasattr(cfg, "multi_infer") and hasattr(cfg.multi_infer, "env"):
            multi_env_cfg = cfg.multi_infer.env
        self.mission = MissionConfig(
            world_size=float(env_cfg.world_size),
            dt=float(env_cfg.dt),
            max_steps=int(cfg.eval.eval_episode_length if hasattr(cfg, "eval") else env_cfg.episode_length),
            hunters=max(1, int(env_cfg.max_hunters_num)),
            explorers=max(1, int(getattr(multi_env_cfg, "num_explorers", 3)) if multi_env_cfg is not None else 3),
            targets=max(1, int(getattr(multi_env_cfg, "num_targets", 5)) if multi_env_cfg is not None else 5),
            overlap_rate=0.2,
            hunters_wait_mode="split",
            explorer_track_speed_scale=1.2,
            loss_timeout_steps=40,
        )

        self.weights = AssignmentWeights()
        self.time_step = 0
        self.executing = False
        self.planned = False

        self.explorers: List[ExplorerRuntime] = []
        self.hunters: List[HunterRuntime] = []
        self.targets: List[TargetRuntime] = []
        self.pursuit_tasks: Dict[int, PursuitRuntime] = {}

        self.patrol_routes = self._load_patrol_routes(
            route_path=str(env_cfg.target_patrol_path),
            route_names=list(env_cfg.target_patrol_names),
        )
        self.target_actor = LearnTargetActor(cfg, target_actor_path, obs_dim=20)
        self.reset_world()

    def reset_world(self):
        """
        功能:
            重置仿真世界并初始化全部实体状态。
        输入:
            无。
        输出:
            无。
        """
        self.time_step = 0
        self.executing = False
        self.planned = False

        self.explorers = []
        self.hunters = []
        self.targets = []
        for runtime in self.pursuit_tasks.values():
            runtime.env.close()
        self.pursuit_tasks = {}

        explorer_cfg = getattr(self.cfg, "Explorer")
        for idx in range(int(self.mission.explorers)):
            agent = ExplorerAgent(
                agent_id=idx,
                max_speed=float(explorer_cfg.max_velo),
                safe_dis=float(explorer_cfg.safe_dis),
                control_mode="velocity",
                max_acc=0.0,
                max_turn_angle=180.0,
                min_turn_limit_velo=0.0,
                policy_type="search",
            )
            init_pos = self._sample_position()
            agent.reset(init_pos)
            self.explorers.append(ExplorerRuntime(agent=agent))

        hunter_cfg = getattr(self.cfg, "Hunter")
        for idx in range(int(self.mission.hunters)):
            agent = HunterAgent(
                agent_id=idx,
                max_speed=float(hunter_cfg.max_velo),
                safe_dis=float(hunter_cfg.safe_dis),
                control_mode=str(hunter_cfg.control_mode).lower(),
                max_acc=float(hunter_cfg.max_acc),
                max_turn_angle=float(hunter_cfg.max_turn_angle),
                min_turn_limit_velo=float(hunter_cfg.min_turn_limit_velo),
                policy_type="learn",
                block_length=0.0,
            )
            init_pos = self._sample_position()
            agent.reset(init_pos)
            speed = float(self.rng.uniform(0.0, float(hunter_cfg.max_velo)))
            self.hunters.append(HunterRuntime(agent=agent, standby_speed=speed))

        target_cfg = getattr(self.cfg, "Target")
        policy_source = str(self.cfg.env.target_policy_source).lower()
        for idx in range(int(self.mission.targets)):
            patrol_route = self.patrol_routes[idx % len(self.patrol_routes)] if len(self.patrol_routes) > 0 else None
            agent = TargetAgent(
                agent_id=idx,
                max_speed=float(target_cfg.max_velo),
                safe_dis=float(target_cfg.safe_dis),
                control_mode=str(target_cfg.control_mode).lower(),
                max_acc=float(target_cfg.max_acc),
                max_turn_angle=float(target_cfg.max_turn_angle),
                min_turn_limit_velo=float(target_cfg.min_turn_limit_velo),
                policy_type=policy_source,
                patrol_waypoints=patrol_route,
                patrol_routes=self.patrol_routes,
                switch_interval=max(1, int(self.cfg.env.target_switch_interval)),
                control_dt=float(self.mission.dt),
                world_size=float(self.mission.world_size),
                escape_dis=float(getattr(self.cfg.reward, "escape_radius", 30.0)),
                escape_gap_angle_bins=int(getattr(self.cfg.reward, "escape_gap_angle_bins", 360)),
                escape_gap_hunter_reward_scale=0.0,
                escape_gap_target_reward_scale=0.0,
                escape_gap_encircle_hunter_reward_scale=0.0,
                escape_gap_encircle_target_reward_scale=0.0,
                escape_gap_intercept_hunter_reward_scale=0.0,
                escape_gap_intercept_target_reward_scale=0.0,
                escape_gap_min_speed=float(getattr(self.cfg.reward, "escape_gap_min_speed", 0.2)),
                boundary_avoid_enable=bool(getattr(self.cfg.env, "target_boundary_avoid_enable", True)),
                boundary_influence_ratio=float(getattr(self.cfg.env, "target_boundary_influence_ratio", 0.30)),
                boundary_enter_ratio=float(getattr(self.cfg.env, "target_boundary_enter_ratio", 0.15)),
                boundary_exit_ratio=float(getattr(self.cfg.env, "target_boundary_exit_ratio", 0.22)),
                boundary_wall_gain=float(getattr(self.cfg.env, "target_boundary_wall_gain", 1.2)),
                boundary_corner_tangent_gain=float(getattr(self.cfg.env, "target_boundary_corner_tangent_gain", 0.8)),
                boundary_smooth_alpha=float(getattr(self.cfg.env, "target_boundary_smooth_alpha", 0.25)),
                boundary_lookahead_steps=int(getattr(self.cfg.env, "target_boundary_lookahead_steps", 5)),
            )
            init_pos = self._sample_position()
            if policy_source == "patrol" and patrol_route is not None and len(patrol_route) > 0:
                init_pos = patrol_route[0]
            agent.reset(init_pos)
            self.target_actor.reset_target(idx)
            value = float(self.rng.uniform(1.0, 10.0))
            required = int(np.clip(int(round(value / 2.0)), 1, 5))
            self.targets.append(
                TargetRuntime(
                    agent=agent,
                    value=value,
                    required_hunters=required,
                    alive=True,
                )
            )

    def apply_task_settings(self, mission: MissionConfig, weights: AssignmentWeights):
        """
        功能:
            应用任务设置并重置世界。
        输入:
            mission (MissionConfig): 新任务参数。
            weights (AssignmentWeights): 分配权重参数。
        输出:
            无。
        """
        self.mission = mission
        self.weights = weights
        self.reset_world()

    def plan_routes(self):
        """
        功能:
            执行搜索航线与待命航线预规划。
        输入:
            无。
        输出:
            无。
        """
        explorer_paths = self._build_explorer_search_paths(
            world_size=float(self.mission.world_size),
            num_explorers=int(self.mission.explorers),
            overlap_rate=float(self.mission.overlap_rate),
        )
        for idx, runtime in enumerate(self.explorers):
            runtime.path = explorer_paths[idx]
            runtime.path_index = 0
            runtime.resume_path_index = 0
            runtime.state = "SEARCH"
            runtime.assigned_target = -1
            if len(runtime.path) > 0:
                runtime.agent.reset(runtime.path[0].copy())

        if str(self.mission.hunters_wait_mode).lower() == "split":
            hunter_paths = self._build_hunter_split_paths(
                world_size=float(self.mission.world_size),
                num_hunters=int(self.mission.hunters),
            )
            for idx, runtime in enumerate(self.hunters):
                runtime.standby_mode = "split"
                runtime.zone_explorer = -1
                runtime.standby_path = hunter_paths[idx]
                runtime.standby_index = 0
                runtime.standby_direction = 1
                runtime.assigned_target = -1
                if len(runtime.standby_path) > 0:
                    runtime.agent.reset(runtime.standby_path[0].copy())
        else:
            self._assign_zone_groups()

        self.planned = True

    def dispatch_execute(self):
        """
        功能:
            将任务切换到执行状态（预规划后生效）。
        输入:
            无。
        输出:
            无。
        """
        if not self.planned:
            return
        self.executing = True

    def step_once(self):
        """
        功能:
            推进仿真一步，执行搜索、发现、分配与追捕。
        输入:
            无。
        输出:
            无。
        """
        if not self.executing:
            return
        if self.time_step >= int(self.mission.max_steps):
            self.executing = False
            return

        self.time_step += 1
        self._move_targets()
        self._move_explorers_and_hunters()
        self._update_discovery_pool()
        self._assignment_if_needed()
        self._step_pursuit_subtasks()
        self._update_pursuit_progress()

        alive_count = sum(1 for t in self.targets if t.alive)
        if alive_count <= 0:
            self.executing = False

    def get_summary(self) -> Dict[str, float]:
        """
        功能:
            获取运行状态汇总指标。
        输入:
            无。
        输出:
            Dict[str,float]: 指标字典。
        """
        alive_targets = sum(1 for t in self.targets if t.alive)
        in_pool = sum(1 for t in self.targets if t.alive and t.in_pool)
        pursuing = sum(1 for t in self.targets if t.alive and t.pursuit_started)
        free_hunters = sum(1 for h in self.hunters if h.assigned_target < 0)
        free_explorers = sum(1 for e in self.explorers if e.assigned_target < 0 and e.state == "SEARCH")
        captured = int(len(self.targets) - alive_targets)
        return {
            "step": float(self.time_step),
            "alive_targets": float(alive_targets),
            "pool_targets": float(in_pool),
            "pursuing_targets": float(pursuing),
            "captured_targets": float(captured),
            "free_hunters": float(free_hunters),
            "free_explorers": float(free_explorers),
        }

    def _sample_position(self) -> np.ndarray:
        """
        功能:
            在地图内随机采样位置。
        输入:
            无。
        输出:
            np.ndarray: shape=(2,)。
        """
        ws = float(self.mission.world_size)
        return self.rng.uniform(-ws * 0.9, ws * 0.9, size=(2,)).astype(np.float32)

    def _move_targets(self):
        """
        功能:
            推进所有存活Target一步，策略行为复用TargetAgent。
        输入:
            无。
        输出:
            无。
        """
        for tid, target in enumerate(self.targets):
            if not target.alive:
                continue
            if target.pursuit_started and tid in self.pursuit_tasks:
                continue
            active_hunters, active_mask = self._active_hunters_for_target(tid)
            action_from_policy = None
            if str(target.agent.policy_type).lower() == "learn":
                action_from_policy = np.zeros(2, dtype=np.float32)
            action = target.agent.select_action(
                step_count=int(self.time_step),
                action_from_policy=action_from_policy,
                rng=self.rng,
                hunters=active_hunters,
                active_hunter_mask=active_mask,
            )
            target.agent.step(action, dt=float(self.mission.dt), world_size=float(self.mission.world_size))

    def _move_explorers_and_hunters(self):
        """
        功能:
            推进Explorer与Hunter状态机运动。
        输入:
            无。
        输出:
            无。
        """
        for ex in self.explorers:
            if ex.state == "SEARCH":
                self._move_along_path(ex.agent, ex.path, ex)
            elif ex.state == "RETURN":
                self._move_along_path(ex.agent, ex.path, ex)
                if ex.path_index == ex.resume_path_index:
                    ex.state = "SEARCH"
            elif ex.state == "TRACK":
                tid = int(ex.assigned_target)
                if tid < 0 or tid >= len(self.targets) or (not self.targets[tid].alive):
                    ex.state = "RETURN"
                    ex.assigned_target = -1
                    continue
                target = self.targets[tid]
                speed = float(ex.agent.max_speed) * float(self.mission.explorer_track_speed_scale)
                self._move_towards(ex.agent, target.agent.position, speed=speed)

        for h in self.hunters:
            if h.assigned_target < 0:
                if h.standby_mode == "split":
                    self._move_hunter_split_standby(h)
                else:
                    self._move_hunter_zone_standby(h)
                continue

            tid = int(h.assigned_target)
            if tid < 0 or tid >= len(self.targets) or (not self.targets[tid].alive):
                self._release_hunter(h)
                continue

            target = self.targets[tid]
            if not target.pursuit_started:
                if target.assigned_explorer >= 0:
                    anchor = self.explorers[target.assigned_explorer].agent.position
                    self._move_towards(h.agent, anchor, speed=float(h.agent.max_speed))
                continue

            if tid in self.pursuit_tasks:
                continue

            chase_action = self._build_hunter_chase_action(h.agent, target.agent.position)
            h.agent.step(
                action_norm=chase_action,
                dt=float(self.mission.dt),
                world_size=float(self.mission.world_size),
            )

    def _step_pursuit_subtasks(self):
        """
        功能:
            推进所有已启动的追捕子任务（观测与step均复用UAVPursuitEnv）。
        输入:
            无。
        输出:
            无。
        """
        for tid, runtime in list(self.pursuit_tasks.items()):
            if tid < 0 or tid >= len(self.targets):
                continue
            target = self.targets[tid]
            if (not target.alive) or (not target.pursuit_started):
                continue

            env = runtime.env
            if not bool(runtime.started):
                self._sync_subenv_from_global(target_id=tid)
                runtime.started = True
            actions = np.zeros((env.agent_num, 2), dtype=np.float32)

            for local_hid, global_hid in enumerate(runtime.hunter_ids):
                if global_hid < 0 or global_hid >= len(self.hunters):
                    continue
                chase_action = self._build_hunter_chase_action(
                    hunter=env.hunters[local_hid],
                    target_pos=env.target.position,
                )
                actions[local_hid] = chase_action

            if str(env.target.policy_type).lower() == "learn":
                team_sees_target = bool(env._team_sees_target())
                obs_all = env._build_obs(team_sees_target=team_sees_target)
                target_obs = np.asarray(obs_all[env.target_index], dtype=np.float32)
                actions[env.target_index] = self.target_actor.act(tid, target_obs)

            env.step(actions)
            self._sync_target_task_from_subenv(tid)

    def _sync_subenv_from_global(self, target_id: int):
        """
        功能:
            将全局状态同步到追捕子环境（用于子任务正式启动前的首次对齐）。
        输入:
            target_id (int): 目标ID。
        输出:
            无。
        """
        runtime = self.pursuit_tasks.get(target_id, None)
        if runtime is None:
            return
        env = runtime.env
        target = self.targets[target_id]

        for local_hid, global_hid in enumerate(runtime.hunter_ids):
            if global_hid < 0 or global_hid >= len(self.hunters):
                continue
            global_h = self.hunters[global_hid].agent
            local_h = env.hunters[local_hid]
            local_h.position = np.asarray(global_h.position, dtype=np.float32).copy()
            local_h.velocity = np.asarray(global_h.velocity, dtype=np.float32).copy()
            local_h.heading = np.asarray(global_h.heading, dtype=np.float32).copy()
            local_h.alive = bool(global_h.alive)

        local_t = env.target
        global_t = target.agent
        local_t.position = np.asarray(global_t.position, dtype=np.float32).copy()
        local_t.velocity = np.asarray(global_t.velocity, dtype=np.float32).copy()
        local_t.heading = np.asarray(global_t.heading, dtype=np.float32).copy()
        local_t.alive = bool(global_t.alive)

    def _update_discovery_pool(self):
        """
        功能:
            更新目标发现池与共享信息。
        输入:
            无。
        输出:
            无。
        """
        for tid, target in enumerate(self.targets):
            if not target.alive:
                continue
            any_seen = False
            for ex in self.explorers:
                dist = float(np.linalg.norm(ex.agent.position - target.agent.position))
                perc = float(getattr(self.cfg.Explorer, "perception_radius", -1))
                perc = float(self.mission.world_size * 2.0) if perc <= 0 else perc
                if dist <= perc:
                    any_seen = True
                    target.discovered = True
                    target.in_pool = True
                    target.last_seen_step = int(self.time_step)
                    target.last_seen_pos = target.agent.position.copy()
                    target.last_seen_vel = target.agent.velocity.copy()
            if any_seen and target.assigned_explorer >= 0:
                target.pursuit_started = True

    def _assignment_if_needed(self):
        """
        功能:
            按任务池与空闲资源触发目标分配。
        输入:
            无。
        输出:
            无。
        """
        idle_explorer_ids = [
            idx for idx, ex in enumerate(self.explorers)
            if ex.state == "SEARCH" and ex.assigned_target < 0
        ]
        idle_hunter_ids = [idx for idx, h in enumerate(self.hunters) if h.assigned_target < 0]
        candidate_target_ids = [
            idx for idx, t in enumerate(self.targets)
            if t.alive and t.in_pool and t.assigned_explorer < 0
        ]
        if len(idle_explorer_ids) == 0 or len(idle_hunter_ids) == 0 or len(candidate_target_ids) == 0:
            return

        explorer_cost = self._build_explorer_cost_matrix(idle_explorer_ids, candidate_target_ids)
        explorer_pairs = MinCostMatcher.solve(explorer_cost)

        assigned_target_by_explorer: Dict[int, int] = {}
        accepted_targets: List[int] = []
        max_cost = 1e6
        for row, col in explorer_pairs:
            if row >= len(idle_explorer_ids) or col >= len(candidate_target_ids):
                continue
            if float(explorer_cost[row, col]) >= max_cost:
                continue
            eid = idle_explorer_ids[row]
            tid = candidate_target_ids[col]
            assigned_target_by_explorer[eid] = tid
            accepted_targets.append(tid)

        if len(accepted_targets) == 0:
            return

        expanded_slots: List[int] = []
        for tid in accepted_targets:
            req = int(np.clip(self.targets[tid].required_hunters, 1, 5))
            expanded_slots.extend([tid] * req)

        if len(expanded_slots) == 0:
            return

        hunter_cost = self._build_hunter_cost_matrix(idle_hunter_ids, expanded_slots)
        hunter_pairs = MinCostMatcher.solve(hunter_cost)

        target_hunter_map: Dict[int, List[int]] = {tid: [] for tid in accepted_targets}
        for row, col in hunter_pairs:
            if row >= len(idle_hunter_ids) or col >= len(expanded_slots):
                continue
            if float(hunter_cost[row, col]) >= max_cost:
                continue
            hid = idle_hunter_ids[row]
            tid = expanded_slots[col]
            target_hunter_map[tid].append(hid)

        for eid, tid in assigned_target_by_explorer.items():
            need = int(np.clip(self.targets[tid].required_hunters, 1, 5))
            picked_hunters = target_hunter_map.get(tid, [])
            if len(picked_hunters) < need:
                self.targets[tid].in_pool = False
                self.targets[tid].discovered = False
                continue

            use_hunters = picked_hunters[:need]
            ex = self.explorers[eid]
            ex.state = "TRACK"
            ex.assigned_target = int(tid)

            target = self.targets[tid]
            target.assigned_explorer = int(eid)
            target.assigned_hunters = [int(x) for x in use_hunters]
            target.assign_step = int(self.time_step)
            target.pursuit_started = False

            for hid in use_hunters:
                h = self.hunters[hid]
                h.last_target = int(h.assigned_target)
                h.assigned_target = int(tid)

            self._create_pursuit_task_env(target_id=int(tid), hunter_ids=use_hunters)

    def _update_pursuit_progress(self):
        """
        功能:
            更新追捕子任务终止、失败回收与资源释放。
        输入:
            无。
        输出:
            无。
        """
        capture_dist = float(self.cfg.env.capture_dis)
        for tid, target in enumerate(self.targets):
            if not target.alive:
                continue
            if target.assigned_explorer < 0:
                continue

            ex = self.explorers[target.assigned_explorer]
            perc = float(getattr(self.cfg.Explorer, "perception_radius", -1))
            perc = float(self.mission.world_size * 2.0) if perc <= 0 else perc
            seen_now = float(np.linalg.norm(ex.agent.position - target.agent.position)) <= perc
            if seen_now:
                target.last_seen_step = int(self.time_step)
                target.last_seen_pos = target.agent.position.copy()
                target.last_seen_vel = target.agent.velocity.copy()
                target.pursuit_started = True

            if (not target.pursuit_started) and (int(self.time_step) - int(target.assign_step) > int(self.mission.loss_timeout_steps)):
                self._abort_target_task(tid)
                continue

            if target.pursuit_started and (int(self.time_step) - int(target.last_seen_step) > int(self.mission.loss_timeout_steps)):
                self._abort_target_task(tid)
                continue

            active_capture = 0
            for hid in list(target.assigned_hunters):
                if hid < 0 or hid >= len(self.hunters):
                    continue
                h = self.hunters[hid]
                dist = float(np.linalg.norm(h.agent.position - target.agent.position))
                if dist <= capture_dist:
                    active_capture += 1

            if active_capture >= int(target.required_hunters):
                target.alive = False
                target.in_pool = False
                target.discovered = False
                self._release_target_resources(tid)

    def _abort_target_task(self, target_id: int):
        """
        功能:
            中止目标子任务并释放资源。
        输入:
            target_id (int): 目标ID。
        输出:
            无。
        """
        if target_id < 0 or target_id >= len(self.targets):
            return
        target = self.targets[target_id]
        target.in_pool = False
        target.discovered = False
        self._release_target_resources(target_id)

    def _release_target_resources(self, target_id: int):
        """
        功能:
            释放目标对应Explorer/Hunter资源并回到待命状态。
        输入:
            target_id (int): 目标ID。
        输出:
            无。
        """
        target = self.targets[target_id]
        if target.assigned_explorer >= 0:
            ex = self.explorers[target.assigned_explorer]
            ex.state = "RETURN"
            ex.assigned_target = -1
            ex.path_index = int(ex.resume_path_index)
        for hid in list(target.assigned_hunters):
            if hid < 0 or hid >= len(self.hunters):
                continue
            self._release_hunter(self.hunters[hid])
        if target_id in self.pursuit_tasks:
            self.pursuit_tasks[target_id].env.close()
            self.pursuit_tasks.pop(target_id, None)
        target.assigned_explorer = -1
        target.assigned_hunters = []
        target.pursuit_started = False

    def _create_pursuit_task_env(self, target_id: int, hunter_ids: List[int]):
        """
        功能:
            为指定目标创建追捕子任务环境，并将全局状态同步为子环境初始状态。
        输入:
            target_id (int): 目标ID。
            hunter_ids (List[int]): 分配到该目标的全局Hunter ID列表。
        输出:
            无。
        """
        if target_id < 0 or target_id >= len(self.targets):
            return
        if len(hunter_ids) <= 0:
            return

        target = self.targets[target_id]
        sub_cfg = copy.deepcopy(self.cfg)
        sub_cfg.env.max_hunters_num = int(len(hunter_ids))
        sub_cfg.env.world_size = float(self.mission.world_size)
        sub_cfg.env.episode_length = int(max(1, self.mission.max_steps))
        sub_cfg.env.target_policy_source = str(target.agent.policy_type).lower()

        sub_env = UAVPursuitEnv(sub_cfg)
        sub_env.seed(int(self.rng.randint(1, 10**9)))
        sub_env.reset(mode="initial")

        init_positions = np.zeros((sub_env.agent_num, 2), dtype=np.float32)
        for local_hid, global_hid in enumerate(hunter_ids):
            init_positions[local_hid] = self.hunters[int(global_hid)].agent.position.copy()
        init_positions[sub_env.target_index] = target.agent.position.copy()
        sub_env._reset_to_positions(init_positions)

        for local_hid, global_hid in enumerate(hunter_ids):
            global_hunter = self.hunters[int(global_hid)].agent
            local_hunter = sub_env.hunters[local_hid]
            local_hunter.velocity = np.asarray(global_hunter.velocity, dtype=np.float32).copy()
            local_hunter.heading = np.asarray(global_hunter.heading, dtype=np.float32).copy()
            local_hunter.alive = bool(global_hunter.alive)
        sub_env.target.velocity = np.asarray(target.agent.velocity, dtype=np.float32).copy()
        sub_env.target.heading = np.asarray(target.agent.heading, dtype=np.float32).copy()
        sub_env.target.alive = bool(target.agent.alive)
        sub_env.target.policy_type = str(target.agent.policy_type).lower()
        if str(sub_env.target.policy_type).lower() == "patrol":
            sub_env.target.patrol_routes = list(target.agent.patrol_routes)
            sub_env.target.patrol_waypoints = list(target.agent.patrol_waypoints)
            sub_env.target.route_index = int(target.agent.route_index)
            sub_env.target.patrol_index = int(target.agent.patrol_index)
            sub_env.target.route_episode_count = int(target.agent.route_episode_count)

        if target_id in self.pursuit_tasks:
            self.pursuit_tasks[target_id].env.close()
        self.pursuit_tasks[target_id] = PursuitRuntime(
            env=sub_env,
            hunter_ids=[int(x) for x in hunter_ids],
        )

    def _sync_target_task_from_subenv(self, target_id: int):
        """
        功能:
            将追捕子环境中的Hunter/Target状态回写到全局仿真。
        输入:
            target_id (int): 目标ID。
        输出:
            无。
        """
        runtime = self.pursuit_tasks.get(target_id, None)
        if runtime is None:
            return
        env = runtime.env

        for local_hid, global_hid in enumerate(runtime.hunter_ids):
            if global_hid < 0 or global_hid >= len(self.hunters):
                continue
            local_h = env.hunters[local_hid]
            global_h = self.hunters[global_hid].agent
            global_h.position = np.asarray(local_h.position, dtype=np.float32).copy()
            global_h.velocity = np.asarray(local_h.velocity, dtype=np.float32).copy()
            global_h.heading = np.asarray(local_h.heading, dtype=np.float32).copy()
            global_h.alive = bool(local_h.alive)
            global_h.trajectory.append(global_h.position.copy())

        target = self.targets[target_id]
        global_t = target.agent
        local_t = env.target
        global_t.position = np.asarray(local_t.position, dtype=np.float32).copy()
        global_t.velocity = np.asarray(local_t.velocity, dtype=np.float32).copy()
        global_t.heading = np.asarray(local_t.heading, dtype=np.float32).copy()
        global_t.alive = bool(local_t.alive)
        global_t.trajectory.append(global_t.position.copy())
        if str(global_t.policy_type).lower() == "patrol":
            global_t.patrol_index = int(local_t.patrol_index)
            global_t.route_index = int(local_t.route_index)
            global_t.route_episode_count = int(local_t.route_episode_count)

        if not bool(local_t.alive):
            target.alive = False
            target.in_pool = False
            target.discovered = False
            self._release_target_resources(target_id)

    def _release_hunter(self, hunter: HunterRuntime):
        """
        功能:
            释放单个Hunter到待命状态。
        输入:
            hunter (HunterRuntime): Hunter运行态。
        输出:
            无。
        """
        hunter.last_target = int(hunter.assigned_target)
        hunter.assigned_target = -1
        if hunter.standby_mode == "split" and len(hunter.standby_path) > 0:
            if hunter.standby_index < len(hunter.standby_path):
                hunter.agent.position = hunter.standby_path[hunter.standby_index].copy()
            hunter.agent.velocity[:] = 0.0

    def _move_along_path(self, agent: ExplorerAgent, path: List[np.ndarray], runtime: ExplorerRuntime):
        """
        功能:
            按规划航点推进Explorer（简化运动，不启用动力学约束）。
        输入:
            agent (ExplorerAgent): Explorer对象。
            path (List[np.ndarray]): 航线航点。
            runtime (ExplorerRuntime): 运行状态。
        输出:
            无。
        """
        if len(path) == 0:
            return
        idx = int(runtime.path_index) % len(path)
        target = path[idx]
        speed = float(agent.max_speed)
        arrived = self._move_towards(agent, target, speed)
        if arrived:
            runtime.path_index = (idx + 1) % len(path)
            if runtime.state == "SEARCH":
                runtime.resume_path_index = int(runtime.path_index)

    def _move_hunter_split_standby(self, runtime: HunterRuntime):
        """
        功能:
            按split待命航线推进Hunter。
        输入:
            runtime (HunterRuntime): Hunter运行状态。
        输出:
            无。
        """
        if len(runtime.standby_path) == 0:
            return
        idx = int(runtime.standby_index)
        target = runtime.standby_path[idx]
        speed = float(max(0.0, runtime.standby_speed))
        arrived = self._move_towards(runtime.agent, target, speed=speed)
        if not arrived:
            return
        if idx <= 0:
            runtime.standby_direction = 1
        elif idx >= len(runtime.standby_path) - 1:
            runtime.standby_direction = -1
        runtime.standby_index = int(np.clip(idx + runtime.standby_direction, 0, len(runtime.standby_path) - 1))

    def _move_hunter_zone_standby(self, runtime: HunterRuntime):
        """
        功能:
            按zone待命模式跟随对应Explorer。
        输入:
            runtime (HunterRuntime): Hunter运行状态。
        输出:
            无。
        """
        if runtime.zone_explorer < 0 or runtime.zone_explorer >= len(self.explorers):
            return
        anchor = self.explorers[runtime.zone_explorer].agent.position + runtime.zone_offset
        runtime.agent.position = np.clip(anchor, -self.mission.world_size, self.mission.world_size).astype(np.float32)
        runtime.agent.velocity[:] = 0.0
        runtime.agent.trajectory.append(runtime.agent.position.copy())

    def _move_towards(self, agent, target_pos: np.ndarray, speed: float) -> bool:
        """
        功能:
            以给定速度将agent推进至目标点。
        输入:
            agent (BaseAgent): Agent对象。
            target_pos (np.ndarray): 目标位置 shape=(2,)。
            speed (float): 速度上限（米/秒）。
        输出:
            bool: 是否到达目标点。
        """
        dt = float(self.mission.dt)
        vec = np.asarray(target_pos, dtype=np.float32) - np.asarray(agent.position, dtype=np.float32)
        dist = float(np.linalg.norm(vec))
        if dist <= 1e-6:
            agent.velocity[:] = 0.0
            agent.trajectory.append(agent.position.copy())
            return True
        step_dist = float(max(0.0, speed) * dt)
        if step_dist >= dist:
            new_pos = np.asarray(target_pos, dtype=np.float32)
            agent.velocity = (vec / max(dt, 1e-6)).astype(np.float32)
            agent.position = np.clip(new_pos, -self.mission.world_size, self.mission.world_size).astype(np.float32)
            agent.trajectory.append(agent.position.copy())
            return True
        direction = vec / dist
        agent.velocity = (direction * float(max(0.0, speed))).astype(np.float32)
        agent.position = np.clip(
            agent.position + agent.velocity * dt,
            -self.mission.world_size,
            self.mission.world_size,
        ).astype(np.float32)
        agent.trajectory.append(agent.position.copy())
        return False

    def _build_hunter_chase_action(self, hunter: HunterAgent, target_pos: np.ndarray) -> np.ndarray:
        """
        功能:
            根据目标位置构造Hunter追捕动作（兼容velocity/acceleration）。
        输入:
            hunter (HunterAgent): Hunter对象。
            target_pos (np.ndarray): 目标位置。
        输出:
            np.ndarray: shape=(2,) 归一化动作。
        """
        vec = np.asarray(target_pos, dtype=np.float32) - np.asarray(hunter.position, dtype=np.float32)
        dist = float(np.linalg.norm(vec))
        if dist <= 1e-6:
            return np.zeros(2, dtype=np.float32)
        direction = vec / dist
        if str(hunter.control_mode).lower() == "acceleration":
            desired_vel = direction * float(hunter.max_speed)
            acc = desired_vel - np.asarray(hunter.velocity, dtype=np.float32)
            if float(hunter.max_acc) <= 1e-8:
                return np.zeros(2, dtype=np.float32)
            return np.clip(acc / float(hunter.max_acc), -1.0, 1.0).astype(np.float32)
        return np.clip(direction, -1.0, 1.0).astype(np.float32)

    def _build_explorer_search_paths(self, world_size: float, num_explorers: int, overlap_rate: float) -> List[List[np.ndarray]]:
        """
        功能:
            生成弓字覆盖航线并切分为M条子航线。
        输入:
            world_size (float): 地图半边长。
            num_explorers (int): Explorer数量。
            overlap_rate (float): 感知重叠率。
        输出:
            List[List[np.ndarray]]: 每个Explorer对应航线。
        """
        perc = float(getattr(self.cfg.Explorer, "perception_radius", -1))
        if perc <= 0:
            perc = float(max(5.0, self.cfg.Explorer.safe_dis))
        spacing = float(max(2.0, 2.0 * perc - perc * float(np.clip(overlap_rate, 0.0, 0.95))))
        margin = float(max(2.0, perc * 0.5))
        y_vals = []
        y = float(world_size - margin)
        while y >= -float(world_size - margin):
            y_vals.append(y)
            y -= spacing
        if len(y_vals) == 0:
            y_vals = [0.0]

        x_min = -float(world_size - margin)
        x_max = float(world_size - margin)
        full_route: List[np.ndarray] = []
        left_to_right = True
        for yy in y_vals:
            if left_to_right:
                full_route.append(np.array([x_min, yy], dtype=np.float32))
                full_route.append(np.array([x_max, yy], dtype=np.float32))
            else:
                full_route.append(np.array([x_max, yy], dtype=np.float32))
                full_route.append(np.array([x_min, yy], dtype=np.float32))
            left_to_right = not left_to_right

        paths: List[List[np.ndarray]] = [[] for _ in range(num_explorers)]
        for idx, wp in enumerate(full_route):
            eid = int(idx % num_explorers)
            paths[eid].append(wp.copy())
        for eid in range(num_explorers):
            if len(paths[eid]) == 0:
                paths[eid] = [self._sample_position()]
        return paths

    def _build_hunter_split_paths(self, world_size: float, num_hunters: int) -> List[List[np.ndarray]]:
        """
        功能:
            构建split模式下Hunter纵向待命航线。
        输入:
            world_size (float): 地图半边长。
            num_hunters (int): Hunter数量。
        输出:
            List[List[np.ndarray]]: 每个Hunter的待命航线。
        """
        margin = max(2.0, world_size * 0.05)
        xs = np.linspace(-world_size + margin, world_size - margin, num=max(1, num_hunters)).astype(np.float32)
        y1 = -world_size + margin
        y2 = world_size - margin
        out = []
        for xx in xs:
            out.append([
                np.array([xx, y1], dtype=np.float32),
                np.array([xx, y2], dtype=np.float32),
            ])
        return out

    def _assign_zone_groups(self):
        """
        功能:
            构建zone模式下Hunter编组与方阵偏移。
        输入:
            无。
        输出:
            无。
        """
        for hid, runtime in enumerate(self.hunters):
            eid = int(hid % max(1, len(self.explorers)))
            runtime.standby_mode = "zone"
            runtime.zone_explorer = eid

        group_map: Dict[int, List[int]] = {}
        for hid, runtime in enumerate(self.hunters):
            group_map.setdefault(runtime.zone_explorer, []).append(hid)

        spacing = max(2.0, float(self.cfg.Hunter.safe_dis) * 0.5)
        for eid, member_ids in group_map.items():
            n = len(member_ids)
            side = int(math.ceil(math.sqrt(max(1, n))))
            slots = []
            for r in range(side):
                for c in range(side):
                    x = (c - (side - 1) / 2.0) * spacing
                    y = -(r + 1) * spacing
                    slots.append(np.array([x, y], dtype=np.float32))
            self.rng.shuffle(slots)
            for i, hid in enumerate(member_ids):
                runtime = self.hunters[hid]
                runtime.zone_offset = slots[i].copy()
                anchor = self.explorers[eid].agent.position + runtime.zone_offset
                runtime.agent.reset(np.clip(anchor, -self.mission.world_size, self.mission.world_size).astype(np.float32))

    def _build_explorer_cost_matrix(self, explorer_ids: List[int], target_ids: List[int]) -> np.ndarray:
        """
        功能:
            构建Explorer-Target匹配代价矩阵。
        输入:
            explorer_ids (List[int]): 空闲Explorer列表。
            target_ids (List[int]): 待分配Target列表。
        输出:
            np.ndarray: shape=(E,T) 代价矩阵。
        """
        high = 1e6
        mat = np.full((len(explorer_ids), len(target_ids)), high, dtype=np.float32)
        for i, eid in enumerate(explorer_ids):
            ex = self.explorers[eid]
            for j, tid in enumerate(target_ids):
                tgt = self.targets[tid]
                if not tgt.alive:
                    continue
                dist = float(np.linalg.norm(ex.agent.position - tgt.last_seen_pos))
                if dist > float(self.weights.max_assign_dist):
                    continue
                value_term = (10.0 - float(tgt.value))
                endurance_term = 0.0
                switch_term = 0.0
                cost = (
                    float(self.weights.distance_weight) * dist
                    + float(self.weights.value_weight) * value_term
                    + float(self.weights.endurance_weight) * endurance_term
                    + float(self.weights.switch_weight) * switch_term
                )
                mat[i, j] = float(cost)
        return mat

    def _build_hunter_cost_matrix(self, hunter_ids: List[int], target_slots: List[int]) -> np.ndarray:
        """
        功能:
            构建Hunter-扩展Target槽位匹配代价矩阵。
        输入:
            hunter_ids (List[int]): 空闲Hunter列表。
            target_slots (List[int]): 扩展后的Target槽位列表。
        输出:
            np.ndarray: shape=(H,S) 代价矩阵。
        """
        high = 1e6
        mat = np.full((len(hunter_ids), len(target_slots)), high, dtype=np.float32)
        for i, hid in enumerate(hunter_ids):
            hunter = self.hunters[hid]
            for j, tid in enumerate(target_slots):
                tgt = self.targets[tid]
                if not tgt.alive:
                    continue
                dist = float(np.linalg.norm(hunter.agent.position - tgt.last_seen_pos))
                if dist > float(self.weights.max_assign_dist):
                    continue
                value_term = (10.0 - float(tgt.value))
                endurance_term = 0.0
                switch_term = 0.0 if hunter.last_target < 0 or hunter.last_target == tid else 1.0
                cost = (
                    float(self.weights.distance_weight) * dist
                    + float(self.weights.value_weight) * value_term
                    + float(self.weights.endurance_weight) * endurance_term
                    + float(self.weights.switch_weight) * switch_term
                )
                mat[i, j] = float(cost)
        return mat

    def _active_hunters_for_target(self, target_id: int) -> Tuple[List[HunterAgent], np.ndarray]:
        """
        功能:
            收集指定目标当前关联的Hunter列表与激活掩码。
        输入:
            target_id (int): 目标ID。
        输出:
            Tuple[List[HunterAgent], np.ndarray]: (Hunter对象列表, 激活掩码)。
        """
        target = self.targets[target_id]
        hunter_objs: List[HunterAgent] = []
        mask: List[bool] = []
        for hid in target.assigned_hunters:
            if hid < 0 or hid >= len(self.hunters):
                continue
            hunter_objs.append(self.hunters[hid].agent)
            mask.append(True)
        return hunter_objs, np.asarray(mask, dtype=bool)

    def _build_target_learn_obs(self, target_id: int, hunters: List[HunterAgent]) -> np.ndarray:
        """
        功能:
            构建learn-target推理观测（简化版本）。
        输入:
            target_id (int): 目标ID。
            hunters (List[HunterAgent]): 当前关联Hunter列表。
        输出:
            np.ndarray: shape=(20,)。
        """
        target = self.targets[target_id].agent
        obs = np.zeros(20, dtype=np.float32)
        ws = float(max(1.0, self.mission.world_size))
        obs[0:2] = target.position / ws
        obs[2:4] = target.velocity / max(1e-6, float(target.max_speed))
        cap = min(4, len(hunters))
        for idx in range(cap):
            rel = hunters[idx].position - target.position
            obs[4 + idx * 4: 6 + idx * 4] = rel / ws
            obs[6 + idx * 4: 8 + idx * 4] = hunters[idx].velocity / max(1e-6, float(hunters[idx].max_speed))
        return obs

    def _load_patrol_routes(self, route_path: str, route_names: List[str]) -> List[List[np.ndarray]]:
        """
        功能:
            从JSON读取巡逻路线并转换为航点列表。
        输入:
            route_path (str): 巡逻文件路径。
            route_names (List[str]): 指定路线名，支持'all'。
        输出:
            List[List[np.ndarray]]: 巡逻路线集合。
        """
        path = Path(route_path)
        if not path.is_absolute():
            path = project_root / path
        routes: List[List[np.ndarray]] = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                routes = self._parse_route_json(data, route_names)
            except Exception:
                routes = []
        if len(routes) > 0:
            return routes
        return [
            [
                np.array([-30.0, -30.0], dtype=np.float32),
                np.array([30.0, -30.0], dtype=np.float32),
                np.array([30.0, 30.0], dtype=np.float32),
                np.array([-30.0, 30.0], dtype=np.float32),
            ]
        ]

    def _parse_route_json(self, data, route_names: List[str]) -> List[List[np.ndarray]]:
        """
        功能:
            解析巡逻文件内容为标准路线列表。
        输入:
            data (Any): JSON加载结果。
            route_names (List[str]): 路线名过滤。
        输出:
            List[List[np.ndarray]]: 路线集合。
        """
        raw_routes: Dict[str, List] = {}
        if isinstance(data, dict):
            if "routes" in data and isinstance(data["routes"], dict):
                raw_routes = dict(data["routes"])
            else:
                for k, v in data.items():
                    if isinstance(v, list):
                        raw_routes[str(k)] = v
        elif isinstance(data, list):
            for idx, route in enumerate(data):
                raw_routes[f"route_{idx}"] = route

        names = [str(x) for x in route_names]
        selected_keys = list(raw_routes.keys()) if "all" in [x.lower() for x in names] else [k for k in raw_routes.keys() if k in names]
        if len(selected_keys) == 0:
            selected_keys = list(raw_routes.keys())

        out: List[List[np.ndarray]] = []
        for key in selected_keys:
            points = raw_routes.get(key, [])
            route: List[np.ndarray] = []
            for point in points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    route.append(np.array([float(point[0]), float(point[1])], dtype=np.float32))
                elif isinstance(point, dict) and "x" in point and "y" in point:
                    route.append(np.array([float(point["x"]), float(point["y"])], dtype=np.float32))
            if len(route) > 0:
                out.append(route)
        return out


class SwarmSimGUI:
    """
    功能:
        提供任务控制、参数配置与可视化界面。
    输入:
        sim (SwarmSimulationCore): 仿真核心实例。
    输出:
        无。
    """

    def __init__(self, sim: SwarmSimulationCore):
        if tk is None:
            raise RuntimeError("tkinter is unavailable in current environment")
        self.sim = sim
        self.root = tk.Tk()
        self.root.title("Swarm Search + Pursuit Simulator")
        self.running = False

        self._build_ui()
        self._schedule_loop()

    def _build_ui(self):
        """
        功能:
            构建GUI布局、参数控件与绘图区域。
        输入:
            无。
        输出:
            无。
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        left = ttk.Frame(self.root, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self.root, padding=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.inputs: Dict[str, tk.StringVar] = {}

        self._add_input(left, "world_size", str(self.sim.mission.world_size))
        self._add_input(left, "hunters", str(self.sim.mission.hunters))
        self._add_input(left, "explorers", str(self.sim.mission.explorers))
        self._add_input(left, "targets", str(self.sim.mission.targets))
        self._add_input(left, "max_steps", str(self.sim.mission.max_steps))
        self._add_input(left, "dt", str(self.sim.mission.dt))
        self._add_input(left, "overlap_rate", str(self.sim.mission.overlap_rate))
        self._add_input(left, "wait_mode(split/zone)", str(self.sim.mission.hunters_wait_mode))
        self._add_input(left, "track_speed_scale", str(self.sim.mission.explorer_track_speed_scale))
        self._add_input(left, "loss_timeout", str(self.sim.mission.loss_timeout_steps))

        self._add_input(left, "w_distance", str(self.sim.weights.distance_weight))
        self._add_input(left, "w_value", str(self.sim.weights.value_weight))
        self._add_input(left, "w_endurance", str(self.sim.weights.endurance_weight))
        self._add_input(left, "w_switch", str(self.sim.weights.switch_weight))
        self._add_input(left, "max_assign_dist", str(self.sim.weights.max_assign_dist))

        row_buttons = ttk.Frame(left)
        row_buttons.pack(fill=tk.X, pady=(8, 4))
        ttk.Button(row_buttons, text="新建任务", command=self._on_apply).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_buttons, text="预规划", command=self._on_plan).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_buttons, text="下发执行", command=self._on_dispatch).pack(side=tk.LEFT, padx=2)

        row_buttons2 = ttk.Frame(left)
        row_buttons2.pack(fill=tk.X, pady=(4, 4))
        ttk.Button(row_buttons2, text="开始", command=self._on_start).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_buttons2, text="暂停", command=self._on_pause).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_buttons2, text="单步", command=self._on_step).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_buttons2, text="重置", command=self._on_reset).pack(side=tk.LEFT, padx=2)

        row_buttons3 = ttk.Frame(left)
        row_buttons3.pack(fill=tk.X, pady=(4, 4))
        ttk.Button(row_buttons3, text="保存倾向", command=self._on_save_profile).pack(side=tk.LEFT, padx=2)
        ttk.Button(row_buttons3, text="加载倾向", command=self._on_load_profile).pack(side=tk.LEFT, padx=2)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var, wraplength=320).pack(fill=tk.X, pady=(10, 0))

        self.figure = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_scene()

    def _add_input(self, parent, key: str, default: str):
        """
        功能:
            添加一行参数输入控件。
        输入:
            parent (ttk.Frame): 父容器。
            key (str): 参数名。
            default (str): 默认值。
        输出:
            无。
        """
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=1)
        ttk.Label(frame, text=key, width=18).pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        self.inputs[key] = var
        ttk.Entry(frame, textvariable=var, width=16).pack(side=tk.RIGHT)

    def _read_mission_and_weights(self) -> Tuple[MissionConfig, AssignmentWeights]:
        """
        功能:
            从输入框读取任务参数与权重参数。
        输入:
            无。
        输出:
            Tuple[MissionConfig, AssignmentWeights]: 配置对象。
        """
        mission = MissionConfig(
            world_size=float(self.inputs["world_size"].get()),
            dt=float(self.inputs["dt"].get()),
            max_steps=int(float(self.inputs["max_steps"].get())),
            hunters=max(1, int(float(self.inputs["hunters"].get()))),
            explorers=max(1, int(float(self.inputs["explorers"].get()))),
            targets=max(1, int(float(self.inputs["targets"].get()))),
            overlap_rate=float(self.inputs["overlap_rate"].get()),
            hunters_wait_mode=str(self.inputs["wait_mode(split/zone)"].get()).strip().lower(),
            explorer_track_speed_scale=float(self.inputs["track_speed_scale"].get()),
            loss_timeout_steps=max(1, int(float(self.inputs["loss_timeout"].get()))),
        )
        if mission.hunters_wait_mode not in ("split", "zone"):
            mission.hunters_wait_mode = "split"

        weights = AssignmentWeights(
            distance_weight=float(self.inputs["w_distance"].get()),
            value_weight=float(self.inputs["w_value"].get()),
            endurance_weight=float(self.inputs["w_endurance"].get()),
            switch_weight=float(self.inputs["w_switch"].get()),
            max_assign_dist=float(self.inputs["max_assign_dist"].get()),
        )
        return mission, weights

    def _on_apply(self):
        """
        功能:
            应用任务参数并重新初始化场景。
        输入:
            无。
        输出:
            无。
        """
        try:
            mission, weights = self._read_mission_and_weights()
            self.running = False
            self.sim.apply_task_settings(mission, weights)
            self.status_var.set("Task initialized")
            self._draw_scene()
        except Exception as e:
            if messagebox is not None:
                messagebox.showerror("参数错误", str(e))

    def _on_plan(self):
        """
        功能:
            执行预规划。
        输入:
            无。
        输出:
            无。
        """
        self.sim.plan_routes()
        self.status_var.set("Planned")
        self._draw_scene()

    def _on_dispatch(self):
        """
        功能:
            下发执行。
        输入:
            无。
        输出:
            无。
        """
        self.sim.dispatch_execute()
        self.status_var.set("Dispatched")

    def _on_start(self):
        """
        功能:
            开始连续仿真。
        输入:
            无。
        输出:
            无。
        """
        self.running = True

    def _on_pause(self):
        """
        功能:
            暂停连续仿真。
        输入:
            无。
        输出:
            无。
        """
        self.running = False

    def _on_step(self):
        """
        功能:
            手动执行一步仿真。
        输入:
            无。
        输出:
            无。
        """
        self.sim.step_once()
        self._draw_scene()

    def _on_reset(self):
        """
        功能:
            重置任务。
        输入:
            无。
        输出:
            无。
        """
        self.running = False
        self.sim.reset_world()
        self.status_var.set("Reset")
        self._draw_scene()

    def _on_save_profile(self):
        """
        功能:
            保存当前分配权重为JSON文件。
        输入:
            无。
        输出:
            无。
        """
        _, weights = self._read_mission_and_weights()
        out_dir = project_root / "results" / "swarm_profiles"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "latest_profile.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(weights.__dict__, f, ensure_ascii=False, indent=2)
        self.status_var.set(f"Saved profile: {out_path}")

    def _on_load_profile(self):
        """
        功能:
            加载分配权重配置到输入框。
        输入:
            无。
        输出:
            无。
        """
        in_path = project_root / "results" / "swarm_profiles" / "latest_profile.json"
        if not in_path.exists():
            self.status_var.set("No profile file")
            return
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = {
            "distance_weight": "w_distance",
            "value_weight": "w_value",
            "endurance_weight": "w_endurance",
            "switch_weight": "w_switch",
            "max_assign_dist": "max_assign_dist",
        }
        for key, input_key in mapping.items():
            if key in data and input_key in self.inputs:
                self.inputs[input_key].set(str(data[key]))
        self.status_var.set(f"Loaded profile: {in_path}")

    def _schedule_loop(self):
        """
        功能:
            GUI事件循环中的定时更新逻辑。
        输入:
            无。
        输出:
            无。
        """
        if self.running:
            self.sim.step_once()
            if not self.sim.executing:
                self.running = False
            self._draw_scene()
        self.root.after(50, self._schedule_loop)

    def _draw_scene(self):
        """
        功能:
            绘制当前仿真场景与状态信息。
        输入:
            无。
        输出:
            无。
        """
        self.ax.clear()
        ws = float(self.sim.mission.world_size)
        self.ax.set_xlim(-ws, ws)
        self.ax.set_ylim(-ws, ws)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.2)

        for ex in self.sim.explorers:
            p = ex.agent.position
            color = "tab:green" if ex.state == "SEARCH" else ("tab:orange" if ex.state == "TRACK" else "tab:gray")
            self.ax.scatter([p[0]], [p[1]], c=color, s=45, marker="^")
            if len(ex.path) > 1:
                path_np = np.asarray(ex.path)
                self.ax.plot(path_np[:, 0], path_np[:, 1], color="tab:green", alpha=0.15)

        for h in self.sim.hunters:
            p = h.agent.position
            color = "tab:blue" if h.assigned_target < 0 else "tab:red"
            self.ax.scatter([p[0]], [p[1]], c=color, s=35, marker="o")
            if h.standby_mode == "split" and len(h.standby_path) > 1:
                path_np = np.asarray(h.standby_path)
                self.ax.plot(path_np[:, 0], path_np[:, 1], color="tab:blue", alpha=0.12)

        for tid, t in enumerate(self.sim.targets):
            if not t.alive:
                continue
            p = t.agent.position
            color = "tab:purple" if t.in_pool else "black"
            self.ax.scatter([p[0]], [p[1]], c=color, s=40, marker="x")
            self.ax.text(float(p[0]) + 1.0, float(p[1]) + 1.0, f"T{tid}", fontsize=8)

        summary = self.sim.get_summary()
        status_text = (
            f"step={int(summary['step'])}, alive={int(summary['alive_targets'])}, "
            f"pool={int(summary['pool_targets'])}, pursuing={int(summary['pursuing_targets'])}, "
            f"captured={int(summary['captured_targets'])}, "
            f"freeH={int(summary['free_hunters'])}, freeE={int(summary['free_explorers'])}"
        )
        self.ax.set_title(status_text)
        self.canvas.draw_idle()

    def run(self):
        """
        功能:
            启动GUI主循环。
        输入:
            无。
        输出:
            无。
        """
        self.root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    """
    功能:
        构建命令行解析器。
    输入:
        无。
    输出:
        argparse.ArgumentParser: 参数解析器。
    """
    parser = argparse.ArgumentParser(description="Swarm simulation GUI entry")
    parser.add_argument("--config_file", type=str, required=True, help="Path to yaml config file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--target_actor", type=str, default=None, help="Optional actor for target learn policy")
    return parser


def main():
    """
    功能:
        程序主入口，加载配置并启动GUI。
    输入:
        无。
    输出:
        无。
    """
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config_file)

    sim = SwarmSimulationCore(cfg=cfg, seed=args.seed, target_actor_path=args.target_actor)
    app = SwarmSimGUI(sim)
    app.run()


if __name__ == "__main__":
    main()
