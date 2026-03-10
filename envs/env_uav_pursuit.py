"""
Multi-UAV Pursuit 环境（Hunter-only版本）

与现有 MAPPO 代码的对接契约:
1) reset() -> List[np.ndarray], 长度=agent_num, 每个元素 shape=(obs_dim,)
2) step(actions) -> [obs, rewards, dones, infos]
   - obs: List[np.ndarray], shape=(agent_num, obs_dim)
   - rewards: List[List[float]], shape=(agent_num, 1)
   - dones: List[bool], shape=(agent_num,)
   - infos: List[dict], shape=(agent_num,)
"""

from typing import List, Optional
import json
import os
from collections import deque

import numpy as np
import matplotlib

# 仅在无显示环境下回退到Agg，避免影响交互式可视化脚本。
if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge


def _safe_distance_penalty(
    dist,
    safe_dis,
    collision_dis,
    collision_penalty_k,
    safe_zone_penalty_scale,
):
    """
    功能:
        根据安全距离与碰撞距离计算单agent的距离风险惩罚。
    输入:
        dist (float): 两agent间欧式距离（米）。
        safe_dis (float): 当前agent安全距离阈值（米）。
        collision_dis (float): 硬碰撞距离阈值（米）。
        collision_penalty_k (float): 最大碰撞惩罚系数（硬碰撞）。
        safe_zone_penalty_scale (float): 危险区惩罚相对硬碰撞惩罚的缩放系数。
    输出:
        float: 非负惩罚值（调用方再加负号累计到reward）。
    """
    safe_val = float(max(0.0, safe_dis))
    coll_val = float(max(0.0, collision_dis))
    if dist >= safe_val:    # 安全区
        return 0.0
    if dist <= coll_val:    # 碰撞区
        return float(max(0.0, collision_penalty_k))
    
    # 危险惩罚区: 线性增长，但上限由safe_zone_penalty_scale控制，显著低于硬碰撞惩罚。
    denom = max(safe_val - coll_val, 1e-6)
    ratio = float(np.clip((safe_val - dist) / denom, 0.0, 1.0))
    scale = float(np.clip(safe_zone_penalty_scale, 0.0, 1.0))
    return float(max(0.0, collision_penalty_k)) * scale * ratio


class BaseAgent(object):
    """
    Agent基础类，负责动作选择与运动学更新。

    坐标与单位:
    - position: 全局笛卡尔坐标(米)，原点在地图中心。
    - velocity: 全局速度(米/秒)。
    - action_norm: 归一化动作，范围[-1,1]，无单位。
      实际速度 = action_norm * max_speed。
    """

    def __init__(
        self,
        agent_id: int,
        role: str,
        max_speed: float,
        safe_dis: float,
        control_mode: str = "velocity",
        max_acc: float = 0.0,
        max_turn_angle: float = 180.0,
        min_turn_limit_velo: float = 0.0,
        policy_type: str = "learn",
        policy_net=None,
        action_update_interval: int = 1,
    ):
        self.agent_id = int(agent_id)
        self.role = role
        self.max_speed = float(max_speed)
        self.safe_dis = float(max(0.0, safe_dis))
        self.control_mode = str(control_mode).lower()
        self.max_acc = float(max(0.0, max_acc))
        self.max_turn_angle = float(max_turn_angle)
        self.min_turn_limit_velo = float(max(0.0, min_turn_limit_velo))
        # 统一将转角上限转换为弧度（配置中建议使用角度值）
        self.max_turn_angle_rad = np.deg2rad(self.max_turn_angle)
        self.policy_type = str(policy_type).lower()
        self.policy_net = policy_net
        self.action_update_interval = max(1, int(action_update_interval))

        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.heading = np.array([1.0, 0.0], dtype=np.float32)
        self.trajectory: List[np.ndarray] = []
        self.alive = True

        self._cached_random_action = np.zeros(2, dtype=np.float32)
        self._last_random_refresh_step = -1

        # print(f"{role} Agent {agent_id}: Policy type: {policy_type}")

    def reset(self, init_pos: np.ndarray):
        """
        输入:
            init_pos (np.ndarray): shape=(2,), 全局坐标(米)。
        输出:
            无。
        """
        self.position = np.asarray(init_pos, dtype=np.float32).copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.heading = np.array([1.0, 0.0], dtype=np.float32)
        self.trajectory = [self.position.copy()]
        self.alive = True
        self._cached_random_action = np.zeros(2, dtype=np.float32)
        self._last_random_refresh_step = -1

    def select_action(
        self,
        step_count: int,
        action_from_policy: Optional[np.ndarray],
        rng: np.random.RandomState,
        hunters: Optional[List["HunterAgent"]] = None,
        active_hunter_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        输入:
            step_count (int): 当前环境步。
            action_from_policy (Optional[np.ndarray]):
                learn策略对应网络输出，shape=(2,), 归一化动作。
            rng (np.random.RandomState): 随机数发生器。
            hunters (Optional[List[HunterAgent]]): Hunter列表（基类默认不使用）。
            active_hunter_mask (Optional[np.ndarray]): Hunter激活掩码（基类默认不使用）。
        输出:
            np.ndarray: shape=(2,), 归一化动作。
        """
        if not self.alive:
            return np.zeros(2, dtype=np.float32)

        if self.policy_type == "learn":
            if action_from_policy is None:
                return np.zeros(2, dtype=np.float32)
            return np.clip(np.asarray(action_from_policy, dtype=np.float32), -1.0, 1.0)

        if self.policy_type == "random":
            if (
                self._last_random_refresh_step < 0
                or step_count - self._last_random_refresh_step >= self.action_update_interval
            ):
                theta = rng.uniform(0.0, 2.0 * np.pi)
                magnitude = rng.uniform(0.0, 1.0)
                self._cached_random_action = np.array(
                    [np.cos(theta), np.sin(theta)], dtype=np.float32
                ) * magnitude
                self._last_random_refresh_step = step_count
            return self._cached_random_action.copy()

        return np.zeros(2, dtype=np.float32)

    def step(self, action_norm: np.ndarray, dt: float, world_size: float):
        """
        输入:
            action_norm (np.ndarray): shape=(2,), 归一化动作。
            dt (float): 时间步长(秒)。
            world_size (float): 地图半边长(米)，用于边界裁剪。
        输出:
            无（内部更新position/velocity/trajectory）。
        """
        if not self.alive:
            self.velocity[:] = 0.0
            self.trajectory.append(self.position.copy())
            return

        u = np.clip(np.asarray(action_norm, dtype=np.float32), -1.0, 1.0)

        if self.control_mode == "acceleration":
            # acceleration模式: 动作表示归一化加速度，速度由积分得到并受max_speed约束。
            acc = u * self.max_acc
            self.velocity = self.velocity + acc * float(dt)
            speed = float(np.linalg.norm(self.velocity))
            if speed > self.max_speed and speed > 1e-8:
                self.velocity = self.velocity / speed * self.max_speed
        else:
            # velocity模式: 动作表示目标速度；Target的random/patrol策略允许自由转向。
            desired_velocity = u * self.max_speed
            bypass_turn_limit = (
                self.role == "target"
                and self.policy_type in ("random", "patrol", "greedy", "escape")
            )
            self.velocity = desired_velocity if bypass_turn_limit else self._apply_turn_limit(self.velocity, desired_velocity)

        self.position = self.position + self.velocity * float(dt)
        self.position = np.clip(self.position, -float(world_size), float(world_size))
        speed = float(np.linalg.norm(self.velocity))
        if speed > 1e-8:
            self.heading = (self.velocity / speed).astype(np.float32)
        self.trajectory.append(self.position.copy())

        self.speed = speed

    def _apply_turn_limit(self, current_velocity: np.ndarray, desired_velocity: np.ndarray) -> np.ndarray:
        """
        功能:
            在velocity控制模式下限制速度方向的单步转角。
        输入:
            current_velocity (np.ndarray): 当前速度，shape=(2,)。
            desired_velocity (np.ndarray): 目标速度，shape=(2,)。
        输出:
            np.ndarray: 转角受限后的速度向量，shape=(2,)。
        """
        # Step 1: 无需限幅的情况直接返回目标速度
        max_turn = float(max(0.0, self.max_turn_angle_rad))
        if max_turn >= np.pi:
            return desired_velocity.astype(np.float32)

        cur_norm = float(np.linalg.norm(current_velocity))
        des_norm = float(np.linalg.norm(desired_velocity))
        if cur_norm < 1e-8 or des_norm < 1e-8:
            return desired_velocity.astype(np.float32)
        if cur_norm <= self.min_turn_limit_velo:
            return desired_velocity.astype(np.float32)

        # Step 2: 计算当前/目标方向夹角并进行裁剪
        theta_cur = float(np.arctan2(current_velocity[1], current_velocity[0]))
        theta_des = float(np.arctan2(desired_velocity[1], desired_velocity[0]))
        delta = theta_des - theta_cur
        delta = float(np.arctan2(np.sin(delta), np.cos(delta)))  # wrap到[-pi,pi]
        delta_clipped = float(np.clip(delta, -max_turn, max_turn))

        # Step 3: 输出受限方向 + 原目标速度幅值
        theta_new = theta_cur + delta_clipped
        return np.array(
            [np.cos(theta_new), np.sin(theta_new)],
            dtype=np.float32,
        ) * np.float32(des_norm)


class HunterAgent(BaseAgent):
    def __init__(
        self,
        agent_id: int,
        max_speed: float,
        safe_dis: float,
        control_mode: str = "velocity",
        max_acc: float = 0.0,
        max_turn_angle: float = 180.0,
        min_turn_limit_velo: float = 0.0,
        policy_type: str = "learn",
        policy_net=None,
        block_length: float = 0.0,
    ):
        """
        功能:
            初始化Hunter智能体。
        输入:
            agent_id (int): Hunter编号。
            max_speed (float): 最大速度（米/秒）。
            safe_dis (float): 安全距离阈值（米）。
            control_mode (str): 控制模式（velocity/acceleration）。
            max_acc (float): acceleration模式下最大加速度（米/秒²）。
            max_turn_angle (float): velocity模式下最大转角（度）。
            min_turn_limit_velo (float): 速度超过该阈值时才启用转角限制（米/秒）。
            policy_type (str): 策略类型。
            policy_net (Any): 可选策略网络。
            block_length (float): 围捕建模的单侧拦截长度（米）。
        输出:
            无。
        """
        super().__init__(
            agent_id=agent_id,
            role="hunter",
            max_speed=max_speed,
            safe_dis=safe_dis,
            control_mode=control_mode,
            max_acc=max_acc,
            max_turn_angle=max_turn_angle,
            min_turn_limit_velo=min_turn_limit_velo,
            policy_type=policy_type,
            policy_net=policy_net,
        )
        self.block_length = float(max(0.0, block_length))

    def compute_intercept_angle_range(self, target_position: np.ndarray):
        """
        功能:
            基于Target位置计算当前Hunter在Target视角下的拦截角区间。
        输入:
            target_position (np.ndarray): Target位置，shape=(2,)。
        输出:
            Optional[tuple[float, float, float]]: (中心角, 半宽角, 距离)；无效时返回None。
        """
        if (not bool(self.alive)) or float(self.block_length) <= 0.0:
            return None
        target_pos = np.asarray(target_position, dtype=np.float32)
        rel = np.asarray(self.position, dtype=np.float32) - target_pos
        dist = float(np.linalg.norm(rel))
        if dist <= 1e-6:
            return None
        center_angle = float(np.arctan2(rel[1], rel[0]))
        half_angle = float(np.arctan(float(self.block_length) / max(dist, 1e-6)))
        return center_angle, half_angle, dist

    def compute_intercept_segment(self, target_position: np.ndarray):
        """
        功能:
            计算围捕可视化用的拦截线段端点（过Hunter并垂直于target->hunter方向）。
        输入:
            target_position (np.ndarray): Target位置，shape=(2,)。
        输出:
            Optional[tuple[np.ndarray, np.ndarray]]: 两个端点；无效时返回None。
        """
        if (not bool(self.alive)) or float(self.block_length) <= 0.0:
            return None
        target_pos = np.asarray(target_position, dtype=np.float32)
        rel = np.asarray(self.position, dtype=np.float32) - target_pos
        dist = float(np.linalg.norm(rel))
        if dist <= 1e-6:
            return None
        rel_dir = rel / dist
        perp_dir = np.array([-rel_dir[1], rel_dir[0]], dtype=np.float32)
        p1 = np.asarray(self.position, dtype=np.float32) + perp_dir * float(self.block_length)
        p2 = np.asarray(self.position, dtype=np.float32) - perp_dir * float(self.block_length)
        return p1.astype(np.float32), p2.astype(np.float32)


class ExplorerAgent(BaseAgent):
    def __init__(
        self,
        agent_id: int,
        max_speed: float,
        safe_dis: float,
        control_mode: str = "velocity",
        max_acc: float = 0.0,
        max_turn_angle: float = 180.0,
        min_turn_limit_velo: float = 0.0,
        policy_type: str = "learn",
        policy_net=None,
    ):
        """
        功能:
            初始化Explorer智能体（当前hunter-only任务中默认不启用）。
        输入:
            agent_id (int): Explorer编号。
            max_speed (float): 最大速度（米/秒）。
            safe_dis (float): 安全距离阈值（米）。
            control_mode (str): 控制模式（velocity/acceleration）。
            max_acc (float): acceleration模式下最大加速度（米/秒²）。
            max_turn_angle (float): velocity模式下最大转角（度）。
            min_turn_limit_velo (float): 速度超过该阈值时才启用转角限制（米/秒）。
            policy_type (str): 策略类型。
            policy_net (Any): 可选策略网络。
        输出:
            无。
        """
        super().__init__(
            agent_id=agent_id,
            role="explorer",
            max_speed=max_speed,
            safe_dis=safe_dis,
            control_mode=control_mode,
            max_acc=max_acc,
            max_turn_angle=max_turn_angle,
            min_turn_limit_velo=min_turn_limit_velo,
            policy_type=policy_type,
            policy_net=policy_net,
        )


class TargetAgent(BaseAgent):
    def __init__(
        self,
        agent_id: int,
        max_speed: float,
        safe_dis: float,
        control_mode: str = "velocity",
        max_acc: float = 0.0,
        max_turn_angle: float = 180.0,
        min_turn_limit_velo: float = 0.0,
        policy_type: str = "learn",
        policy_net=None,
        action_update_interval: int = 1,
        patrol_waypoints: Optional[List[np.ndarray]] = None,
        patrol_routes: Optional[List[List[np.ndarray]]] = None,
        switch_interval: int = 1,
        control_dt: float = 1.0,
        world_size: float = 1.0,
        escape_dis: float = 0.0,
        escape_gap_angle_bins: int = 360,
        escape_gap_hunter_reward_scale: float = 0.0,
        escape_gap_target_reward_scale: float = 0.0,
        escape_gap_encircle_hunter_reward_scale: float = 0.0,
        escape_gap_encircle_target_reward_scale: float = 0.0,
        escape_gap_intercept_hunter_reward_scale: float = 0.0,
        escape_gap_intercept_target_reward_scale: float = 0.0,
        escape_gap_min_speed: float = 0.2,
        boundary_avoid_enable: bool = True,
        boundary_influence_ratio: float = 0.30,
        boundary_enter_ratio: float = 0.15,
        boundary_exit_ratio: float = 0.22,
        boundary_wall_gain: float = 1.2,
        boundary_corner_tangent_gain: float = 0.8,
        boundary_smooth_alpha: float = 0.25,
        boundary_lookahead_steps: int = 5,
    ):
        """
        功能:
            初始化Target智能体，支持learn/patrol/random/greedy/escape策略。
        输入:
            agent_id (int): Target编号。
            max_speed (float): 最大速度（米/秒）。
            safe_dis (float): 安全距离阈值（米）。
            control_mode (str): 控制模式（velocity/acceleration）。
            max_acc (float): acceleration模式下最大加速度（米/秒²）。
            max_turn_angle (float): velocity模式下最大转角（度）。
            min_turn_limit_velo (float): 速度超过该阈值时才启用转角限制（米/秒）。
            policy_type (str): learn/patrol/random/greedy/escape。
            policy_net (Any): learn模式下可选策略网络。
            action_update_interval (int): random策略重采样间隔（步）。
            patrol_waypoints (Optional[List[np.ndarray]]): 巡逻航点（全局坐标，米）。
            patrol_routes (Optional[List[List[np.ndarray]]]): 可选巡逻路线集合。
            switch_interval (int): 路线切换间隔（按episode计）。
            control_dt (float): 环境控制步长dt（秒），用于patrol速度缩放。
            world_size (float): 地图半边长（米），用于random策略边界修正。
            escape_dis (float): 围捕分析半径（米）。
            escape_gap_angle_bins (int): 360度离散角bins数量。
            escape_gap_hunter_reward_scale (float): Hunter侧escape_gap_reward缩放系数。
            escape_gap_target_reward_scale (float): Target侧escape_gap_reward缩放系数。
            escape_gap_encircle_hunter_reward_scale (float): Hunter侧包围质量奖励缩放系数。
            escape_gap_encircle_target_reward_scale (float): Target侧缺口质量奖励缩放系数。
            escape_gap_intercept_hunter_reward_scale (float): Hunter侧拦截奖励缩放系数。
            escape_gap_intercept_target_reward_scale (float): Target侧逃逸方向奖励缩放系数。
            escape_gap_min_speed (float): 计算逃脱方向奖励的最低Target速度阈值（米/秒）。
            boundary_avoid_enable (bool): 是否启用边界软约束。
            boundary_influence_ratio (float): 软约束生效半径占world_size比例。
            boundary_enter_ratio (float): 边界保护进入阈值占world_size比例。
            boundary_exit_ratio (float): 边界保护退出阈值占world_size比例。
            boundary_wall_gain (float): 边界法向修正增益。
            boundary_corner_tangent_gain (float): 角落切向滑移增益。
            boundary_smooth_alpha (float): 动作EMA平滑系数（0~1）。
            boundary_lookahead_steps (int): 多步前瞻步数。
        输出:
            无。
        """
        super().__init__(
            agent_id=agent_id,
            role="target",
            max_speed=max_speed,
            safe_dis=safe_dis,
            control_mode=control_mode,
            max_acc=max_acc,
            max_turn_angle=max_turn_angle,
            min_turn_limit_velo=min_turn_limit_velo,
            policy_type=policy_type,
            policy_net=policy_net,
            action_update_interval=action_update_interval,
        )
        self.patrol_waypoints = patrol_waypoints or []
        self.patrol_routes = patrol_routes or (
            [self.patrol_waypoints] if len(self.patrol_waypoints) > 0 else []
        )
        self.switch_interval = max(1, int(switch_interval))
        self.control_dt = float(max(1e-6, control_dt))
        self.world_size = float(max(1e-6, world_size))
        self.escape_dis = float(max(0.0, escape_dis))
        self.escape_gap_angle_bins = int(max(16, escape_gap_angle_bins))
        self.escape_gap_hunter_reward_scale = float(max(0.0, escape_gap_hunter_reward_scale))
        self.escape_gap_target_reward_scale = float(max(0.0, escape_gap_target_reward_scale))
        self.escape_gap_encircle_hunter_reward_scale = float(max(0.0, escape_gap_encircle_hunter_reward_scale))
        self.escape_gap_encircle_target_reward_scale = float(max(0.0, escape_gap_encircle_target_reward_scale))
        self.escape_gap_intercept_hunter_reward_scale = float(max(0.0, escape_gap_intercept_hunter_reward_scale))
        self.escape_gap_intercept_target_reward_scale = float(max(0.0, escape_gap_intercept_target_reward_scale))
        self.escape_gap_min_speed = float(max(0.0, escape_gap_min_speed))
        self.boundary_avoid_enable = bool(boundary_avoid_enable)
        self.boundary_influence_ratio = float(max(0.01, boundary_influence_ratio))
        self.boundary_enter_ratio = float(max(0.0, boundary_enter_ratio))
        self.boundary_exit_ratio = float(max(self.boundary_enter_ratio, boundary_exit_ratio))
        self.boundary_wall_gain = float(max(0.0, boundary_wall_gain))
        self.boundary_corner_tangent_gain = float(max(0.0, boundary_corner_tangent_gain))
        self.boundary_smooth_alpha = float(np.clip(boundary_smooth_alpha, 0.0, 1.0))
        self.boundary_lookahead_steps = int(max(1, boundary_lookahead_steps))
        self.boundary_mode_active = False
        self.last_boundary_action = np.zeros(2, dtype=np.float32)
        self.route_episode_count = 0
        self.route_index = 0
        self.patrol_index = 0
        self.max_escape_gap_angle = 0.0
        self.max_escape_gap_center_angle = 0.0
        self.max_escape_gap_start_angle = 0.0
        self.max_escape_gap_end_angle = 0.0
        self.escape_gap_metric_valid = False
        self.last_escape_gap_blocked_mask = np.zeros(self.escape_gap_angle_bins, dtype=bool)
        self.last_encircling_hunter_ids = []
        self.last_escape_gap_encircle_score = 0.0
        self.last_escape_gap_open_score = 0.0
        self.last_escape_gap_hunter_direction_score = 0.0
        self.last_escape_gap_target_direction_score = 0.0

    def reset(
        self,
        init_pos: np.ndarray,
        force_route_index: Optional[int] = None,
        force_route_episode_count: Optional[int] = None,
    ):
        """
        功能:
            重置Target状态并将巡逻索引归零。
        输入:
            init_pos (np.ndarray): shape=(2,), 全局坐标（米）。
            force_route_index (Optional[int]): 强制使用的巡逻路线索引。
            force_route_episode_count (Optional[int]): 强制设置的路线episode计数。
        输出:
            无。
        """
        super().reset(init_pos)
        self.max_escape_gap_angle = 0.0
        self.max_escape_gap_center_angle = 0.0
        self.max_escape_gap_start_angle = 0.0
        self.max_escape_gap_end_angle = 0.0
        self.escape_gap_metric_valid = False
        self.last_escape_gap_blocked_mask = np.zeros(self.escape_gap_angle_bins, dtype=bool)
        self.last_encircling_hunter_ids = []
        self.last_escape_gap_encircle_score = 0.0
        self.last_escape_gap_open_score = 0.0
        self.last_escape_gap_hunter_direction_score = 0.0
        self.last_escape_gap_target_direction_score = 0.0
        self.boundary_mode_active = False
        self.last_boundary_action = np.zeros(2, dtype=np.float32)
        if force_route_episode_count is not None:
            self.route_episode_count = int(force_route_episode_count)
        else:
            self.route_episode_count += 1

        if self.policy_type == "patrol" and len(self.patrol_routes) > 0:
            # 每次reset后episode计数+1，与switch_interval取模为0时切换到下一条路线。
            if force_route_index is not None:
                self.route_index = int(force_route_index) % len(self.patrol_routes)
            elif self.route_episode_count % self.switch_interval == 0:
                self.route_index = (self.route_index + 1) % len(self.patrol_routes)
            self.patrol_waypoints = self.patrol_routes[self.route_index]
            self.patrol_index = 0
            if len(self.patrol_waypoints) > 0:
                self.position = self.patrol_waypoints[0].copy()
                self.trajectory = [self.position.copy()]
                self.velocity[:] = 0.0
        else:
            self.patrol_index = 0

    def select_action(
        self,
        step_count: int,
        action_from_policy: Optional[np.ndarray],
        rng: np.random.RandomState,
        hunters: Optional[List[HunterAgent]] = None,
        active_hunter_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        功能:
            根据Target策略类型返回动作。
        输入:
            step_count (int): 当前环境步。
            action_from_policy (Optional[np.ndarray]): learn策略动作。
            rng (np.random.RandomState): 随机数生成器。
            hunters (Optional[List[HunterAgent]]): Hunter列表。
            active_hunter_mask (Optional[np.ndarray]): Hunter激活掩码，shape=(num_hunters,)。
        输出:
            np.ndarray: shape=(2,), 归一化动作。
        """
        if self.policy_type == "patrol":
            return self._patrol_action()
        if self.policy_type == "random":
            raw_random_action = super().select_action(step_count, action_from_policy, rng)
            return self._action_with_boundary_avoidance(raw_random_action)
        if self.policy_type == "greedy":
            return self._greedy_action(hunters=hunters, active_hunter_mask=active_hunter_mask)
        if self.policy_type == "escape":
            return self._escape_action(hunters=hunters, active_hunter_mask=active_hunter_mask)
        return super().select_action(step_count, action_from_policy, rng)

    def _action_with_boundary_avoidance(self, base_action: np.ndarray) -> np.ndarray:
        """
        功能:
            对动作做边界连续软约束修正（含滞回、角落切向滑移和动作平滑）。
        输入:
            base_action (np.ndarray): shape=(2,), 原始动作（归一化）。
        输出:
            np.ndarray: shape=(2,), 修正后的动作（归一化）。
        """
        base_action = np.clip(np.asarray(base_action, dtype=np.float32), -1.0, 1.0)
        if not bool(self.boundary_avoid_enable):
            self.last_boundary_action = base_action.copy()
            return base_action

        # Step 1: 多步前瞻边界距离，提前介入避免“撞墙后急拐”。
        min_boundary_dist, ref_pos = self._predict_min_boundary_distance(
            base_action=base_action,
            horizon_steps=int(self.boundary_lookahead_steps),
        )
        world_size = float(max(1e-6, self.world_size))
        influence_dist = float(np.clip(self.boundary_influence_ratio, 0.01, 1.0) * world_size)
        enter_dist = float(np.clip(self.boundary_enter_ratio, 0.0, 1.0) * world_size)
        exit_dist = float(np.clip(self.boundary_exit_ratio, 0.0, 1.0) * world_size)
        exit_dist = max(exit_dist, enter_dist)

        # Step 2: 滞回切换，防止阈值附近来回抖动。
        if self.boundary_mode_active:
            if min_boundary_dist >= exit_dist:
                self.boundary_mode_active = False
        else:
            if min_boundary_dist <= enter_dist:
                self.boundary_mode_active = True
        need_avoid = bool(self.boundary_mode_active or (min_boundary_dist < influence_dist))
        if not need_avoid:
            out = self._smooth_action(base_action)
            self.last_boundary_action = out.copy()
            return out

        # Step 3: 连续墙面斥力 + 角落切向滑移，避免“中心/边缘”二值切换。
        wall_vec, corner_weight = self._build_wall_avoidance_vector(
            ref_pos=ref_pos,
            influence_dist=influence_dist,
        )
        wall_norm = float(np.linalg.norm(wall_vec))
        if wall_norm <= 1e-8:
            out = self._smooth_action(base_action)
            self.last_boundary_action = out.copy()
            return out

        wall_dir = wall_vec / wall_norm
        adjusted = base_action + float(self.boundary_wall_gain) * wall_dir
        if corner_weight > 0.0 and float(self.boundary_corner_tangent_gain) > 0.0:
            tan_1 = np.array([-wall_dir[1], wall_dir[0]], dtype=np.float32)
            tan_2 = -tan_1
            tan_dir = tan_1 if float(np.dot(tan_1, base_action)) >= float(np.dot(tan_2, base_action)) else tan_2
            adjusted = adjusted + float(self.boundary_corner_tangent_gain) * float(corner_weight) * tan_dir

        out = np.clip(adjusted, -1.0, 1.0).astype(np.float32)
        out = self._smooth_action(out)
        self.last_boundary_action = out.copy()
        return out

    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        """
        功能:
            对动作施加EMA平滑，减少边界附近方向抖动。
        输入:
            action (np.ndarray): 原始动作，shape=(2,)。
        输出:
            np.ndarray: 平滑后动作，shape=(2,)。
        """
        alpha = float(np.clip(self.boundary_smooth_alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0).astype(np.float32)
        mixed = (1.0 - alpha) * np.asarray(self.last_boundary_action, dtype=np.float32) + alpha * np.asarray(action, dtype=np.float32)
        return np.clip(mixed, -1.0, 1.0).astype(np.float32)

    def _predict_min_boundary_distance(self, base_action: np.ndarray, horizon_steps: int):
        """
        功能:
            预测若保持当前动作执行多步时，轨迹到边界的最小余量。
        输入:
            base_action (np.ndarray): 归一化动作，shape=(2,)。
            horizon_steps (int): 前瞻步数。
        输出:
            tuple[float, np.ndarray]: (最小边界余量(米), 对应位置shape=(2,))。
        """
        dt = float(max(1e-6, self.control_dt))
        ws = float(max(1e-6, self.world_size))
        action = np.clip(np.asarray(base_action, dtype=np.float32), -1.0, 1.0)
        pos = np.asarray(self.position, dtype=np.float32).copy()
        vel = np.asarray(self.velocity, dtype=np.float32).copy()

        min_dist = float("inf")
        min_pos = pos.copy()
        for _ in range(int(max(1, horizon_steps))):
            if self.control_mode == "acceleration":
                vel = vel + action * float(self.max_acc) * dt
                speed = float(np.linalg.norm(vel))
                if speed > float(self.max_speed) and speed > 1e-8:
                    vel = vel / speed * float(self.max_speed)
            else:
                vel = action * float(self.max_speed)
            pos = pos + vel * dt
            dist = float(max(0.0, ws - max(abs(float(pos[0])), abs(float(pos[1])))))
            if dist < min_dist:
                min_dist = dist
                min_pos = pos.copy()
        return float(min_dist), np.asarray(min_pos, dtype=np.float32)

    def _build_wall_avoidance_vector(self, ref_pos: np.ndarray, influence_dist: float):
        """
        功能:
            计算参考位置下的墙面斥力向量与角落权重。
        输入:
            ref_pos (np.ndarray): 参考位置，shape=(2,)。
            influence_dist (float): 边界影响范围（米）。
        输出:
            tuple[np.ndarray, float]: (法向斥力向量, 角落权重[0,1])。
        """
        ws = float(max(1e-6, self.world_size))
        p = np.asarray(ref_pos, dtype=np.float32)
        inf = float(max(1e-6, influence_dist))

        wall_dists = [
            float(ws - float(p[0])),  # right
            float(ws + float(p[0])),  # left
            float(ws - float(p[1])),  # top
            float(ws + float(p[1])),  # bottom
        ]
        wall_normals = [
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]

        wall_vec = np.zeros(2, dtype=np.float32)
        weights = []
        for d, normal in zip(wall_dists, wall_normals):
            ratio = float(np.clip((inf - float(d)) / inf, 0.0, 1.0))
            weight = ratio * ratio
            weights.append(weight)
            wall_vec = wall_vec + float(weight) * normal

        sorted_weights = sorted(weights, reverse=True)
        corner_weight = float(np.clip(sorted_weights[1], 0.0, 1.0)) if len(sorted_weights) >= 2 else 0.0
        return wall_vec, corner_weight

    def _random_action_with_boundary_avoidance(self, random_action: np.ndarray) -> np.ndarray:
        """
        功能:
            random策略边界修正兼容接口（内部复用通用边界修正函数）。
        输入:
            random_action (np.ndarray): shape=(2,), 原始随机动作（归一化）。
        输出:
            np.ndarray: shape=(2,), 修正后的动作（归一化）。
        """
        return self._action_with_boundary_avoidance(random_action)

    def _greedy_action(
        self,
        hunters: Optional[List[HunterAgent]],
        active_hunter_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        功能:
            greedy策略：选取最近Hunter并朝其反方向运动，再叠加边界兜底修正。
        输入:
            hunters (Optional[List[HunterAgent]]): Hunter列表。
            active_hunter_mask (Optional[np.ndarray]): Hunter激活掩码。
        输出:
            np.ndarray: shape=(2,), 归一化动作。
        """
        if hunters is None or len(hunters) == 0:
            return self._default_hold_action()

        nearest_vec = None
        nearest_dist = float("inf")
        for hid, hunter in enumerate(hunters):
            is_active = True
            if active_hunter_mask is not None and hid < len(active_hunter_mask):
                is_active = bool(active_hunter_mask[hid])
            if (not is_active) or (not bool(hunter.alive)):
                continue
            vec = np.asarray(self.position, dtype=np.float32) - np.asarray(hunter.position, dtype=np.float32)
            dist = float(np.linalg.norm(vec))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_vec = vec

        if nearest_vec is None:
            return self._default_hold_action()

        vec_norm = float(np.linalg.norm(nearest_vec))
        if vec_norm <= 1e-8:
            return self._default_hold_action()
        else:
            away_dir = (nearest_vec / vec_norm).astype(np.float32)
        return self._action_with_boundary_avoidance(away_dir)

    def _escape_action(
        self,
        hunters: Optional[List[HunterAgent]],
        active_hunter_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        功能:
            escape策略：受Hunter与边界斥力场共同作用，沿最终合力方向运动。
        输入:
            hunters (Optional[List[HunterAgent]]): Hunter列表。
            active_hunter_mask (Optional[np.ndarray]): Hunter激活掩码。
        输出:
            np.ndarray: shape=(2,), 归一化动作。
        """
        safe_hunters = [] if hunters is None else list(hunters)
        if active_hunter_mask is None:
            safe_mask = np.ones(len(safe_hunters), dtype=bool)
        else:
            safe_mask = np.asarray(active_hunter_mask, dtype=bool)

        # Step 1: 计算Hunter斥力（与Hunter连线反向，距离越近权重越大）。
        hunter_repulse = np.zeros(2, dtype=np.float32)
        has_active_hunter = False
        hunter_repulse_power = 2.0
        for hid, hunter in enumerate(safe_hunters):
            is_active = bool(safe_mask[hid]) if hid < len(safe_mask) else True
            if (not is_active) or (not bool(hunter.alive)):
                continue
            has_active_hunter = True
            rel = np.asarray(self.position, dtype=np.float32) - np.asarray(hunter.position, dtype=np.float32)
            dist = float(np.linalg.norm(rel))
            if dist <= 1e-6:
                continue
            dir_away = rel / dist
            weight = 1.0 / max(dist ** hunter_repulse_power, 1e-6)
            hunter_repulse += (float(weight) * dir_away).astype(np.float32)

        # Step 2: 计算边界斥力（离墙越近斥力越强）。
        world_size = float(max(1e-6, self.world_size))
        influence_dist = float(np.clip(self.boundary_influence_ratio, 0.01, 1.0) * world_size)
        boundary_repulse, _ = self._build_wall_avoidance_vector(
            ref_pos=np.asarray(self.position, dtype=np.float32),
            influence_dist=influence_dist,
        )

        # Step 3: 合力方向作为escape目标方向。
        force = np.asarray(hunter_repulse, dtype=np.float32) + float(self.boundary_wall_gain) * np.asarray(
            boundary_repulse, dtype=np.float32
        )
        force_norm = float(np.linalg.norm(force))
        if (not has_active_hunter) or force_norm <= 1e-8:
            return self._default_hold_action()
        escape_dir = (force / force_norm).astype(np.float32)
        return self._action_with_boundary_avoidance(escape_dir)

    def _default_hold_action(self) -> np.ndarray:
        """
        功能:
            greedy/escape无有效方向时的默认动作：
            - acceleration模式：输出反向减速度使速度衰减到0；
            - velocity模式：输出零速度动作。
        输入:
            无。
        输出:
            np.ndarray: shape=(2,), 归一化动作。
        """
        if self.control_mode == "acceleration":
            speed = float(np.linalg.norm(self.velocity))
            if speed <= 1e-8 or float(self.max_acc) <= 1e-8:
                return np.zeros(2, dtype=np.float32)
            brake_acc = -np.asarray(self.velocity, dtype=np.float32)
            return np.clip(brake_acc / float(self.max_acc), -1.0, 1.0).astype(np.float32)
        return np.zeros(2, dtype=np.float32)

    def _patrol_action(self) -> np.ndarray:
        """
        功能:
            计算patrol模式下动作（velocity/acceleration模式分别处理）。
        输入:
            无（内部使用self.position与self.patrol_waypoints）。
        输出:
            np.ndarray: shape=(2,), 归一化方向动作。
        """
        # Step 1: 空路线保护
        if not self.patrol_waypoints:
            return np.zeros(2, dtype=np.float32)

        # Step 2: 计算当前航点相对向量与距离
        waypoint = self.patrol_waypoints[self.patrol_index]
        vec = waypoint - self.position
        dist = float(np.linalg.norm(vec))

        # Step 3: 到达判定（acc模式放宽阈值，减轻转向与惯性导致的绕圈）
        if self.control_mode == "acceleration":
            arrive_thre = max(float(self.max_speed) * 1.2, float(self.safe_dis) * 0.8)
        else:
            arrive_thre = float(self.max_speed) * float(self.control_dt)

        if dist <= arrive_thre:  # 到达当前航点
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_waypoints)
            waypoint = self.patrol_waypoints[self.patrol_index]  # 下一个航点
            vec = waypoint - self.position
            dist = float(np.linalg.norm(vec))
        if dist <= 1e-6:
            return np.zeros(2, dtype=np.float32)

        # Step 4: velocity模式：临近航点时按剩余距离降低动作幅值，避免越点往返
        direction = (vec / dist).astype(np.float32)
        if self.control_mode == "velocity":
            step_reachable_dist = max(float(self.max_speed) * float(self.control_dt), 1e-6)
            speed_scale = float(np.clip(dist / step_reachable_dist, 0.0, 1.0))
            return (direction * speed_scale).astype(np.float32)

        # Step 5: acceleration模式：求“逼近期望速度”的加速度方向，抵消原速度惯性
        if self.control_mode == "acceleration":
            desired_speed = float(np.clip(dist, 0.0, float(self.max_speed)))
            desired_velocity = direction * desired_speed
            acc_target = desired_velocity - self.velocity
            if float(np.linalg.norm(acc_target)) <= 1e-6 or float(self.max_acc) <= 1e-6:
                return np.zeros(2, dtype=np.float32)
            return np.clip(acc_target / float(self.max_acc), -1.0, 1.0).astype(np.float32)

        # Step 6: 其他模式回退为单位方向
        return direction

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        """
        功能:
            将角度归一化到[-pi, pi]区间。
        输入:
            angle_rad (float): 原始角度（弧度）。
        输出:
            float: 归一化后角度（弧度）。
        """
        return float(np.arctan2(np.sin(float(angle_rad)), np.cos(float(angle_rad))))

    @staticmethod
    def _shift_to_2pi(angle_rad: float) -> float:
        """
        功能:
            将角度平移到[0, 2pi)区间（以-pi对应0索引，便于与离散bin对齐）。
        输入:
            angle_rad (float): 原始角度（弧度）。
        输出:
            float: 平移后的角度。
        """
        return float(np.mod(float(angle_rad) + np.pi, 2.0 * np.pi))

    def _angle_to_bin_index(self, angle_rad: float, bins: int) -> int:
        """
        功能:
            将角度映射到离散角bin索引。
        输入:
            angle_rad (float): 角度（弧度）。
            bins (int): 离散角bin数量。
        输出:
            int: 对应bin索引。
        """
        b = int(max(1, bins))
        bin_width = float(2.0 * np.pi / b)
        shifted = self._shift_to_2pi(angle_rad)
        idx = int(np.floor(shifted / max(bin_width, 1e-8)))
        return int(np.clip(idx, 0, b - 1))

    def compute_max_escape_gap(self, hunters: List[HunterAgent], active_hunter_mask: np.ndarray):
        """
        功能:
            计算Target周围最大潜在可逃脱空间夹角，并更新内部缓存供reward/render调用。
        输入:
            hunters (List[HunterAgent]): Hunter列表。
            active_hunter_mask (np.ndarray): Hunter激活掩码，shape=(num_hunters,)。
        输出:
            dict: 包含metric_valid、active_alive_hunters、encircling_hunters、max_gap_angle等信息。
        """
        self.max_escape_gap_angle = 0.0
        self.max_escape_gap_center_angle = 0.0
        self.max_escape_gap_start_angle = 0.0
        self.max_escape_gap_end_angle = 0.0
        self.escape_gap_metric_valid = False
        self.last_escape_gap_blocked_mask = np.zeros(self.escape_gap_angle_bins, dtype=bool)
        self.last_encircling_hunter_ids = []
        self.last_escape_gap_encircle_score = 0.0
        self.last_escape_gap_open_score = 0.0
        self.last_escape_gap_hunter_direction_score = 0.0
        self.last_escape_gap_target_direction_score = 0.0

        if not bool(self.alive):
            return {
                "metric_valid": False,
                "active_alive_hunters": 0,
                "encircling_hunters": 0,
                "max_gap_angle": 0.0,
                "max_gap_center_angle": 0.0,
                "max_gap_start_angle": 0.0,
                "max_gap_end_angle": 0.0,
                "encircling_hunter_ids": [],
            }

        # Step 1: 统计active且alive的Hunter数量（<=1时不参与指标统计）
        active_alive_hunters = 0
        for hid, hunter in enumerate(hunters):
            is_active = bool(active_hunter_mask[hid]) if hid < len(active_hunter_mask) else True
            if is_active and bool(hunter.alive):
                active_alive_hunters += 1
        if active_alive_hunters <= 1:
            return {
                "metric_valid": False,
                "active_alive_hunters": int(active_alive_hunters),
                "encircling_hunters": 0,
                "max_gap_angle": 0.0,
                "max_gap_center_angle": 0.0,
                "max_gap_start_angle": 0.0,
                "max_gap_end_angle": 0.0,
                "encircling_hunter_ids": [],
            }

        # Step 2: 筛选escape_dis内围捕Hunter并构造角覆盖区间
        angle_centers = []
        angle_halves = []
        encircling_hunter_ids = []
        for hid, hunter in enumerate(hunters):
            is_active = bool(active_hunter_mask[hid]) if hid < len(active_hunter_mask) else True
            if (not is_active) or (not bool(hunter.alive)):
                continue
            rel = np.asarray(hunter.position, dtype=np.float32) - np.asarray(self.position, dtype=np.float32)
            dist = float(np.linalg.norm(rel))
            if float(self.escape_dis) > 0.0 and dist > float(self.escape_dis):
                continue
            angle_info = hunter.compute_intercept_angle_range(np.asarray(self.position, dtype=np.float32))
            if angle_info is None:
                continue
            center_angle, half_angle, _ = angle_info
            angle_centers.append(float(center_angle))
            angle_halves.append(float(max(0.0, half_angle)))
            encircling_hunter_ids.append(int(hid))

        self.last_encircling_hunter_ids = list(encircling_hunter_ids)
        bins = int(max(16, self.escape_gap_angle_bins))
        bin_width = float(2.0 * np.pi / bins)
        blocked_mask = np.zeros(bins, dtype=bool)
        for center_angle, half_angle in zip(angle_centers, angle_halves):
            if half_angle <= 0.0:
                continue
            if half_angle >= np.pi:
                blocked_mask[:] = True
                break
            start_angle = float(center_angle - half_angle)
            end_angle = float(center_angle + half_angle)
            if (end_angle - start_angle) >= (2.0 * np.pi):
                blocked_mask[:] = True
                break
            start_shift = self._shift_to_2pi(start_angle)
            end_shift = self._shift_to_2pi(end_angle)
            start_idx = self._angle_to_bin_index(start_angle, bins)
            end_idx = self._angle_to_bin_index(end_angle, bins)
            if start_shift <= end_shift:
                blocked_mask[start_idx : end_idx + 1] = True
            else:
                blocked_mask[start_idx:] = True
                blocked_mask[: end_idx + 1] = True
        self.last_escape_gap_blocked_mask = blocked_mask.copy()

        # Step 3: 在环形离散空间中搜索最长连续未阻塞区间
        if np.all(blocked_mask):
            best_len = 0
            best_start = 0
        elif not np.any(blocked_mask):
            best_len = bins
            best_start = 0
        else:
            doubled = np.concatenate([blocked_mask, blocked_mask], axis=0)
            best_len = 0
            best_start = 0
            run_len = 0
            run_start = 0
            for idx, is_blocked in enumerate(doubled):
                if not bool(is_blocked):
                    if run_len == 0:
                        run_start = idx
                    run_len += 1
                    candidate_len = min(run_len, bins)
                    if candidate_len > best_len:
                        best_len = int(candidate_len)
                        best_start = int(run_start)
                else:
                    run_len = 0
            best_start = int(best_start % bins)

        max_gap_angle = float(best_len * bin_width)
        max_gap_start_angle = float(-np.pi + best_start * bin_width)
        max_gap_center_angle = self._wrap_angle(max_gap_start_angle + 0.5 * max_gap_angle)
        max_gap_end_angle = self._wrap_angle(max_gap_start_angle + max_gap_angle)

        self.max_escape_gap_angle = float(max_gap_angle)
        self.max_escape_gap_center_angle = float(max_gap_center_angle)
        self.max_escape_gap_start_angle = self._wrap_angle(max_gap_start_angle)
        self.max_escape_gap_end_angle = float(max_gap_end_angle)
        self.escape_gap_metric_valid = True
        return {
            "metric_valid": True,
            "active_alive_hunters": int(active_alive_hunters),
            "encircling_hunters": int(len(encircling_hunter_ids)),
            "max_gap_angle": float(self.max_escape_gap_angle),
            "max_gap_center_angle": float(self.max_escape_gap_center_angle),
            "max_gap_start_angle": float(self.max_escape_gap_start_angle),
            "max_gap_end_angle": float(self.max_escape_gap_end_angle),
            "encircling_hunter_ids": list(encircling_hunter_ids),
        }

    def compute_escape_gap_reward(self, hunters: List[HunterAgent], active_hunter_mask: np.ndarray):
        """
        功能:
            计算escape_gap拆分奖励（包围质量 + 拦截），并返回应分配奖励的围捕Hunter索引。
        输入:
            hunters (List[HunterAgent]): Hunter列表。
            active_hunter_mask (np.ndarray): Hunter激活掩码，shape=(num_hunters,)。
        输出:
            tuple[float, float, float, float, List[int], dict]:
                (Hunter包围奖励值, Target包围奖励值, Hunter拦截奖励值, Target拦截奖励值, 奖励接收Hunter索引, 诊断信息)。
        """

        # Step 0: 全部缩放系数为0时直接跳过。
        if (
            self.escape_gap_encircle_hunter_reward_scale == 0.0
            and self.escape_gap_encircle_target_reward_scale == 0.0
            and self.escape_gap_intercept_hunter_reward_scale == 0.0
            and self.escape_gap_intercept_target_reward_scale == 0.0
            and self.escape_gap_hunter_reward_scale == 0.0
            and self.escape_gap_target_reward_scale == 0.0
        ):
            gap_info = self.compute_max_escape_gap([], None)
            return 0.0, 0.0, 0.0, 0.0, [], gap_info
        
        gap_info = self.compute_max_escape_gap(hunters, active_hunter_mask)
        if (not bool(gap_info.get("metric_valid", False))) or int(gap_info.get("active_alive_hunters", 0)) <= 1:
            return 0.0, 0.0, 0.0, 0.0, [], gap_info
        encircling_hunter_ids = list(gap_info.get("encircling_hunter_ids", []))
        if len(encircling_hunter_ids) <= 1:
            return 0.0, 0.0, 0.0, 0.0, [], gap_info

        max_gap_angle = float(gap_info.get("max_gap_angle", 0.0))
        gap_open_score = float(np.clip(max_gap_angle / (2.0 * np.pi), 0.0, 1.0))
        encircle_score = 1.0 - gap_open_score

        # Step 1: 包围质量奖励（仅依赖encircle_score / gap_open_score）。
        hunter_encircle_reward_value = (
            float(self.escape_gap_encircle_hunter_reward_scale)
            * float(max(0.0, encircle_score))
        )
        target_encircle_reward_value = (
            float(self.escape_gap_encircle_target_reward_scale)
            * float(max(0.0, gap_open_score))
        )

        # Step 2: 拦截奖励（仅依赖direction_score；Target速度过小时跳过）。
        speed = float(np.linalg.norm(np.asarray(self.velocity, dtype=np.float32)))
        target_direction_score = 0.0
        hunter_direction_score = 0.0
        hunter_intercept_reward_value = 0.0
        target_intercept_reward_value = 0.0
        direction_reward_enabled = bool(speed >= float(self.escape_gap_min_speed) and max_gap_angle > 1e-6)
        if direction_reward_enabled:
            # - Target: 鼓励朝逃脱缺口中心方向运动。
            # - Hunter: 鼓励Target运动方向与逃脱缺口中心方向相反。
            target_move_angle = float(np.arctan2(float(self.velocity[1]), float(self.velocity[0])))
            gap_center = float(gap_info.get("max_gap_center_angle", 0.0))
            angle_delta = float(self._wrap_angle(target_move_angle - gap_center))
            target_direction_score = 0.5 * (1.0 + float(np.cos(angle_delta)))
            hunter_direction_score = 0.5 * (1.0 + float(np.cos(angle_delta - np.pi)))
            hunter_intercept_reward_value = (
                float(self.escape_gap_intercept_hunter_reward_scale)
                * float(max(0.0, hunter_direction_score))
            )
            target_intercept_reward_value = (
                float(self.escape_gap_intercept_target_reward_scale)
                * float(max(0.0, target_direction_score))
            )

        gap_info["encircle_score"] = float(encircle_score)
        gap_info["gap_open_score"] = float(gap_open_score)
        gap_info["hunter_direction_score"] = float(hunter_direction_score)
        gap_info["target_direction_score"] = float(target_direction_score)
        gap_info["direction_reward_enabled"] = bool(direction_reward_enabled)
        self.last_escape_gap_encircle_score = float(encircle_score)
        self.last_escape_gap_open_score = float(gap_open_score)
        self.last_escape_gap_hunter_direction_score = float(hunter_direction_score)
        self.last_escape_gap_target_direction_score = float(target_direction_score)
        return (
            hunter_encircle_reward_value,
            target_encircle_reward_value,
            hunter_intercept_reward_value,
            target_intercept_reward_value,
            encircling_hunter_ids,
            gap_info,
        )


class UAVPursuitEnv(object):
    def __init__(self, config):
        """
        功能:
            初始化Hunter-only追逃环境。
        输入:
            config (EasyDict): 分层配置对象。
        输出:
            无。
        """
        # 步骤1：读取配置分组
        self.config = config
        env_cfg = config.env
        hunter_cfg = config.Hunter
        target_cfg = config.Target
        reward_cfg = config.reward

        # 步骤2：初始化环境物理参数与奖励参数
        self.world_size = float(env_cfg.world_size)
        self.default_world_size = float(self.world_size)
        self.dt = float(env_cfg.dt)
        self.max_steps = int(env_cfg.episode_length)
        self.num_hunters = int(env_cfg.max_hunters_num)
        self.num_explorers = int(env_cfg.num_explorers)
        if self.num_explorers != 0:
            raise ValueError("当前阶段仅实现 hunter-only: num_explorers 必须为 0")

        self.target_index = self.num_hunters
        self.agent_num = self.num_hunters + 1
        self.neighbor_N = int(env_cfg.neighbor_N)
        self.neighbor_N = max(0, self.neighbor_N)
        self.coord_summary_obs_enable = bool(env_cfg.coord_summary_obs_enable)
        self.coord_topk_hunters = int(max(1, int(env_cfg.coord_topk_hunters)))
        self.neighbor_feat_dim = 6  # [dx,dy,dvx,dvy,d,valid]
        self.target_feat_dim = 6    # [dx,dy,dvx,dvy,d,visible]
        self.coord_summary_feat_dim = 2 if bool(self.coord_summary_obs_enable) else 0
        self.obs_dim = (
            4
            + self.neighbor_N * self.neighbor_feat_dim
            + self.target_feat_dim
            + 5
            + self.coord_summary_feat_dim
        )
        self.action_dim = 2

        self.capture_dis = float(env_cfg.capture_dis)
        self.capture_step = int(env_cfg.capture_step)
        self.collision_dis = float(env_cfg.collision_dis)
        self.hunters_in_zone = bool(env_cfg.hunters_in_zone)
        self.target_avoid_hunter_zone = bool(env_cfg.target_avoid_hunter_zone)
        self.target_hunter_zone_min_dis = float(max(0.0, env_cfg.target_hunter_zone_min_dis))
        self.target_pos_init_guidance_step = int(env_cfg.target_pos_init_guidance_step)
        self.target_pos_guidance = bool(env_cfg.target_pos_guidance)
        self.noisy_target_pos_std = float(env_cfg.noisy_target_pos_std)
        self.noisy_target_vel_std = float(env_cfg.noisy_target_vel_std)
        self.target_policy_source = str(env_cfg.target_policy_source).lower()
        self.target_switch_interval = int(env_cfg.target_switch_interval)
        self.target_boundary_avoid_enable = bool(getattr(env_cfg, "target_boundary_avoid_enable", True))
        self.target_boundary_influence_ratio = float(getattr(env_cfg, "target_boundary_influence_ratio", 0.30))
        self.target_boundary_enter_ratio = float(getattr(env_cfg, "target_boundary_enter_ratio", 0.15))
        self.target_boundary_exit_ratio = float(getattr(env_cfg, "target_boundary_exit_ratio", 0.22))
        self.target_boundary_wall_gain = float(getattr(env_cfg, "target_boundary_wall_gain", 1.2))
        self.target_boundary_corner_tangent_gain = float(
            getattr(env_cfg, "target_boundary_corner_tangent_gain", 0.8)
        )
        self.target_boundary_smooth_alpha = float(getattr(env_cfg, "target_boundary_smooth_alpha", 0.25))
        self.target_boundary_lookahead_steps = int(getattr(env_cfg, "target_boundary_lookahead_steps", 5))

        self.hunter_perception_radius = float(hunter_cfg.perception_radius)
        self.target_perception_radius = float(target_cfg.perception_radius)
        self.target_safe_dis = float(max(self.collision_dis, float(target_cfg.safe_dis)))
        self.collision_penalty_k = float(reward_cfg.collision_penalty_k)
        self.safe_zone_penalty_scale = float(reward_cfg.safe_zone_penalty_scale)
        self.collision_penalty_cap = float(reward_cfg.collision_penalty_cap)
        self.speed_penalty = float(reward_cfg.speed_penalty)
        self.base_far_scale = float(reward_cfg.base_far_scale)
        self.base_near_scale = float(reward_cfg.base_near_scale)
        self.base_streak_scale = float(reward_cfg.base_streak_scale)
        self.base_streak_cap = int(reward_cfg.base_streak_cap)
        self.base_reward_topk_enable = bool(reward_cfg.base_reward_topk_enable)
        self.base_reward_topk_k = int(max(1, int(reward_cfg.base_reward_topk_k)))
        self.base_reward_non_topk_scale = float(
            np.clip(float(reward_cfg.base_reward_non_topk_scale), 0.0, 1.0)
        )
        self.base_reward_mode = str(getattr(reward_cfg, "base_reward_mode", "legacy")).lower()
        if self.base_reward_mode not in ("legacy", "delta_window"):
            raise ValueError(
                "Unsupported reward.base_reward_mode: {} (choices: legacy, delta_window)".format(
                    str(self.base_reward_mode)
                )
            )
        self.base_delta_window = int(max(1, int(getattr(reward_cfg, "base_delta_window", 10))))
        self.base_delta_hunter_scale = float(getattr(reward_cfg, "base_delta_hunter_scale", 1.0))
        self.base_delta_target_scale = float(getattr(reward_cfg, "base_delta_target_scale", 1.0))
        self.base_delta_norm_scale = float(max(1e-6, float(getattr(reward_cfg, "base_delta_norm_scale", self.capture_dis))))
        self.hunter_capture_reward = float(reward_cfg.hunter_capture_reward)
        self.target_captured_penalty = float(reward_cfg.target_captured_penalty)
        capture_reward_allocation = str(
            getattr(reward_cfg, "capture_reward_allocation", "team")
        ).lower()
        if capture_reward_allocation == "naive":
            capture_reward_allocation = "team"
        if capture_reward_allocation not in ("team", "alone", "encircle"):
            raise ValueError(
                "Unsupported reward.capture_reward_allocation: {} (choices: team, alone, encircle)".format(
                    str(capture_reward_allocation)
                )
            )
        self.capture_reward_allocation = str(capture_reward_allocation)
        self.target_collision_penalty = float(reward_cfg.target_collision_penalty)
        escape_block_scale = float(max(0.0, reward_cfg.escape_block_scale))
        escape_block_length = float(max(0.0, self.capture_dis * escape_block_scale))
        escape_gap_enable = bool(reward_cfg.escape_gap_enable)
        escape_gap_hunter_reward_scale = float(max(0.0, reward_cfg.escape_gap_hunter_reward_scale))
        escape_gap_target_reward_scale = float(max(0.0, reward_cfg.escape_gap_target_reward_scale))
        escape_gap_encircle_hunter_reward_scale = float(
            max(
                0.0,
                getattr(
                    reward_cfg,
                    "escape_gap_encircle_hunter_reward_scale",
                    escape_gap_hunter_reward_scale,
                ),
            )
        )
        escape_gap_encircle_target_reward_scale = float(
            max(
                0.0,
                getattr(
                    reward_cfg,
                    "escape_gap_encircle_target_reward_scale",
                    escape_gap_target_reward_scale,
                ),
            )
        )
        escape_gap_intercept_hunter_reward_scale = float(
            max(
                0.0,
                getattr(
                    reward_cfg,
                    "escape_gap_intercept_hunter_reward_scale",
                    escape_gap_hunter_reward_scale,
                ),
            )
        )
        escape_gap_intercept_target_reward_scale = float(
            max(
                0.0,
                getattr(
                    reward_cfg,
                    "escape_gap_intercept_target_reward_scale",
                    escape_gap_target_reward_scale,
                ),
            )
        )
        if not escape_gap_enable:
            escape_gap_hunter_reward_scale = 0.0
            escape_gap_target_reward_scale = 0.0
            escape_gap_encircle_hunter_reward_scale = 0.0
            escape_gap_encircle_target_reward_scale = 0.0
            escape_gap_intercept_hunter_reward_scale = 0.0
            escape_gap_intercept_target_reward_scale = 0.0
        escape_radius = float(max(0.0, reward_cfg.escape_radius))
        escape_gap_angle_bins = int(max(16, int(reward_cfg.escape_gap_angle_bins)))
        escape_gap_min_speed = float(max(0.0, reward_cfg.escape_gap_min_speed))
        self.hunter_zone_spacing = float(
            max(self.collision_dis * 3.0, float(hunter_cfg.safe_dis) * 1.2, 1e-6)
        )
        self.hunter_zone_offsets = self._build_hunter_zone_offsets(
            max_hunters_num=int(self.num_hunters),
            spacing=float(self.hunter_zone_spacing),
        )

        # 步骤3：初始化运行时状态缓存
        self.rng = np.random.RandomState()
        self.position_rng = np.random.RandomState()
        self.base_seed = 0
        self.step_count = 0
        self.episode_count = 0
        self.capture_counter = np.zeros(self.num_hunters, dtype=np.int32)
        self.done = np.zeros(self.agent_num, dtype=bool)

        self.active_hunter_mask = np.ones(self.num_hunters, dtype=bool)
        self.active_num_hunters = self.num_hunters
        self.hunter_distance_histories = [deque(maxlen=int(self.base_delta_window)) for _ in range(self.num_hunters)]
        self.task_seed = None
        self.regen_scope = "train"
        self.target_route_id = 0
        self.initial_reset_count = 0

        self.shared_target_pos = np.zeros(2, dtype=np.float32)
        self.shared_target_vel = np.zeros(2, dtype=np.float32)
        self.shared_target_valid = False
        self.last_seen_age = 0
        self.last_episode_captured = False
        self.last_capture_step = None
        self.last_target_collided = False
        self.last_collision_pairs = []
        self.last_boundary_collision_agents = []
        self.hunter_reward_sum = 0.0
        self.target_reward_sum = 0.0
        self.reward_step_count = 0
        self.hunter_reward_last = 0.0
        self.target_reward_last = 0.0
        for hid in range(self.num_hunters):
            self.hunter_distance_histories[hid].clear()
        self.last_coord_summary_cache = np.zeros(
            (self.agent_num, self.coord_summary_feat_dim),
            dtype=np.float32,
        )
        self.active_target_patrol_names = list(env_cfg.target_patrol_names)
        self.last_reset_mode = "initial"
        self.last_coord_summary_cache = np.zeros(
            (self.agent_num, self.coord_summary_feat_dim),
            dtype=np.float32,
        )

        # 步骤4：初始化Agent对象
        patrol_routes = self._load_patrol_routes(
            env_cfg.target_patrol_path, list(env_cfg.target_patrol_names)
        )
        if len(self._last_loaded_patrol_route_names) > 0:
            self.active_target_patrol_names = list(self._last_loaded_patrol_route_names)
        self.hunters = [
            HunterAgent(
                i,
                max_speed=float(hunter_cfg.max_velo),
                safe_dis=float(max(self.collision_dis, float(hunter_cfg.safe_dis))),
                control_mode=str(hunter_cfg.control_mode).lower(),
                max_acc=float(hunter_cfg.max_acc),
                max_turn_angle=float(hunter_cfg.max_turn_angle),
                min_turn_limit_velo=float(hunter_cfg.min_turn_limit_velo),
                policy_type="learn",
                block_length=float(escape_block_length),
            )
            for i in range(self.num_hunters)
        ]
        self.target = TargetAgent(
            agent_id=self.target_index,
            max_speed=float(target_cfg.max_velo),
            safe_dis=float(self.target_safe_dis),
            control_mode=str(target_cfg.control_mode).lower(),
            max_acc=float(target_cfg.max_acc),
            max_turn_angle=float(target_cfg.max_turn_angle),
            min_turn_limit_velo=float(target_cfg.min_turn_limit_velo),
            policy_type=self.target_policy_source,
            action_update_interval=max(1, self.target_switch_interval),
            patrol_waypoints=patrol_routes[0] if patrol_routes else None,
            patrol_routes=patrol_routes,
            switch_interval=max(1, self.target_switch_interval),
            control_dt=float(self.dt),
            world_size=float(self.world_size),
            escape_dis=float(escape_radius),
            escape_gap_angle_bins=int(escape_gap_angle_bins),
            escape_gap_hunter_reward_scale=float(escape_gap_hunter_reward_scale),
            escape_gap_target_reward_scale=float(escape_gap_target_reward_scale),
            escape_gap_encircle_hunter_reward_scale=float(escape_gap_encircle_hunter_reward_scale),
            escape_gap_encircle_target_reward_scale=float(escape_gap_encircle_target_reward_scale),
            escape_gap_intercept_hunter_reward_scale=float(escape_gap_intercept_hunter_reward_scale),
            escape_gap_intercept_target_reward_scale=float(escape_gap_intercept_target_reward_scale),
            escape_gap_min_speed=float(escape_gap_min_speed),
            boundary_avoid_enable=bool(self.target_boundary_avoid_enable),
            boundary_influence_ratio=float(self.target_boundary_influence_ratio),
            boundary_enter_ratio=float(self.target_boundary_enter_ratio),
            boundary_exit_ratio=float(self.target_boundary_exit_ratio),
            boundary_wall_gain=float(self.target_boundary_wall_gain),
            boundary_corner_tangent_gain=float(self.target_boundary_corner_tangent_gain),
            boundary_smooth_alpha=float(self.target_boundary_smooth_alpha),
            boundary_lookahead_steps=int(self.target_boundary_lookahead_steps),
        )
        self.patrol_routes = patrol_routes
        self.default_target_policy_source = str(self.target_policy_source)
        self.default_target_patrol_names = list(env_cfg.target_patrol_names)
        self.default_target_patrol_path = str(env_cfg.target_patrol_path)
        self.default_hunters_in_zone = bool(self.hunters_in_zone)

        self.train_split_cfg = config.domain_randomization.train_split

    @property
    def agents(self):
        """
        功能:
            获取环境中全部agent列表。
        输入:
            无。
        输出:
            List[BaseAgent]: [hunters..., target]。
        """
        return self.hunters + [self.target]

    def seed(self, seed):
        """
        功能:
            设置环境随机种子。
        输入:
            seed (int): 随机种子。
        输出:
            无。
        """
        self.base_seed = int(seed)
        self.rng.seed(self.base_seed)
        self.position_rng.seed(self.base_seed)

    def set_regen_scope(self, scope: str):
        """
        功能:
            设置regen采样使用的配置分支（train/eval）。
        输入:
            scope (str): 采样分支名称。
        输出:
            无。
        """
        self.regen_scope = str(scope).lower()

    def reset(self, mode: str = "initial", task_spec: Optional[dict] = None):
        """
        功能:
            重置环境到episode初始状态并返回初始观测。
        输入:
            mode (str): reset模式，支持initial/recover/regen。
            task_spec (Optional[dict]): 可选任务规格，供regen或initial覆盖当前任务参数。
        输出:
            List[np.ndarray]: shape=(agent_num, obs_dim)。
        """
        mode_val = str(mode).lower()
        self.last_reset_mode = str(mode_val)
        if mode_val == "regen":
            sampled_spec = task_spec if task_spec is not None else self._sample_regen_task_spec()
            self._apply_task_spec(sampled_spec, reset_position_rng=True)
            self._reset_target_route_state_for_recover()
            self.initial_reset_count = 0
            return self._reset_with_sampled_positions()

        if mode_val == "recover":
            recover_seed = self.base_seed if self.task_seed is None else int(self.task_seed)
            self.rng.seed(int(recover_seed))
            self.position_rng.seed(int(recover_seed))
            self._reset_target_route_state_for_recover()
            return self._reset_with_sampled_positions()

        if mode_val != "initial":
            raise ValueError(f"Unsupported reset mode: {mode}")

        # train阶段：按initial reset次数触发周期性regen
        if self.regen_scope == "train" and task_spec is None and bool(self.train_split_cfg.enable):
            self.initial_reset_count += 1
            interval_hit = int(self.initial_reset_count) % int(self.train_split_cfg.regen_interval_episode) == 0
            random_hit = float(self.rng.uniform(0.0, 1.0)) <= float(self.train_split_cfg.regen_prob)
            if interval_hit and random_hit:
                sampled_spec = self._sample_regen_task_spec()
                self._apply_task_spec(sampled_spec, reset_position_rng=True)
                self._reset_target_route_state_for_recover()
                self.initial_reset_count = 0
                return self._reset_with_sampled_positions()

        if task_spec is not None:
            self._apply_task_spec(task_spec, reset_position_rng=True)
            self._reset_target_route_state_for_recover()
            self.initial_reset_count = 0
        return self._reset_with_sampled_positions()

    def _reset_with_sampled_positions(self):
        """
        功能:
            使用当前位置随机流采样初始位置并执行reset主逻辑。
            若Target初始capture_dis范围内存在active Hunter，则最多重采样10次。
        输入:
            无。
        输出:
            List[np.ndarray]: shape=(agent_num, obs_dim)。
        """
        # Step 1: 采样初始位置，并做“Target捕获圈内无Hunter”约束重试
        max_retry = 10
        init_positions = None
        for _ in range(max_retry):
            sampled = self._sample_initial_positions()
            if self._is_valid_initial_positions(sampled):
                init_positions = sampled
                break
            init_positions = sampled

        # Step 2: 使用最终采样结果执行reset
        return self._reset_to_positions(init_positions)

    def _reset_to_positions(self, init_positions: np.ndarray):
        """
        功能:
            将环境重置到指定初始位置，并按照active_hunter_mask启停Hunter。
        输入:
            init_positions (np.ndarray): shape=(agent_num,2)，各agent初始位置。
        输出:
            List[np.ndarray]: shape=(agent_num, obs_dim)。
        """
        # 步骤1：清空episode状态
        self.episode_count += 1
        self.step_count = 0
        self.done[:] = False
        self.capture_counter[:] = 0
        self.shared_target_valid = False
        self.shared_target_pos[:] = 0.0
        self.shared_target_vel[:] = 0.0
        self.last_seen_age = 0
        self.last_episode_captured = False
        self.last_capture_step = None
        self.last_target_collided = False
        self.last_collision_pairs = []
        self.last_boundary_collision_agents = []
        self.hunter_reward_sum = 0.0
        self.target_reward_sum = 0.0
        self.reward_step_count = 0
        self.hunter_reward_last = 0.0
        self.target_reward_last = 0.0

        # 步骤2：按给定位置初始化所有agent
        if init_positions.shape != (self.agent_num, 2):
            raise ValueError(
                f"init_positions shape mismatch: expected {(self.agent_num, 2)}, got {init_positions.shape}"
            )

        for hid, hunter in enumerate(self.hunters):
            hunter.reset(np.asarray(init_positions[hid], dtype=np.float32))
            if not bool(self.active_hunter_mask[hid]):
                hunter.alive = False
                hunter.position[:] = 0.0
                hunter.velocity[:] = 0.0
                hunter.heading = np.array([1.0, 0.0], dtype=np.float32)
                hunter.trajectory = [hunter.position.copy()]
                self.capture_counter[hid] = 0

        self.target.reset(np.asarray(init_positions[self.target_index], dtype=np.float32))

        # 步骤3：返回初始观测
        return self._build_obs(team_sees_target=False)

    def _apply_task_spec(self, task_spec: dict, reset_position_rng: bool = True):
        """
        功能:
            应用任务规格（激活Hunter数量、地图尺寸、Target策略与巡逻路线、初始化seed、Hunter初始化方式）。
        输入:
            task_spec (dict): 任务规格字典。
            reset_position_rng (bool): 是否重置初始位置采样随机流。
        输出:
            无。
        """
        spec = dict(task_spec or {})

        # Step 0: 应用任务指定world_size；未指定时回退到环境初始world_size
        world_size = float(spec.get("world_size", self.default_world_size))
        self.world_size = float(max(1e-6, world_size))
        self.target.world_size = float(self.world_size)

        active_num_hunters = int(spec.get("num_hunters", self.num_hunters))
        active_num_hunters = int(np.clip(active_num_hunters, 1, self.num_hunters))
        self.active_num_hunters = active_num_hunters
        self.active_hunter_mask[:] = False
        self.active_hunter_mask[: self.active_num_hunters] = True

        # 兼容hunters_in_zone与hunter_in_zone两种键名。
        hunters_in_zone = spec.get("hunters_in_zone", spec.get("hunter_in_zone", self.default_hunters_in_zone))
        self.hunters_in_zone = bool(hunters_in_zone)

        target_policy_source = str(spec.get("target_policy_source", self.default_target_policy_source)).lower()
        target_patrol_path = str(spec.get("target_patrol_path", self.default_target_patrol_path))
        target_patrol_names = list(spec.get("target_patrol_names", self.default_target_patrol_names))
        target_route_id = int(spec.get("target_route_id", 0))

        patrol_routes = self._load_patrol_routes(target_patrol_path, target_patrol_names)
        if target_policy_source == "patrol" and len(patrol_routes) == 0:
            target_policy_source = "random"
        self.target_policy_source = target_policy_source
        self.target.policy_type = target_policy_source
        self.target.patrol_routes = patrol_routes
        self.target.patrol_waypoints = patrol_routes[0] if len(patrol_routes) > 0 else []
        self.target.route_index = 0
        self.target.route_episode_count = 0
        self.target.patrol_index = 0
        self.patrol_routes = patrol_routes
        self.active_target_patrol_names = (
            list(self._last_loaded_patrol_route_names)
            if len(self._last_loaded_patrol_route_names) > 0
            else list(target_patrol_names)
        )
        self.target_route_id = 0
        if len(self.patrol_routes) > 0:
            self.target_route_id = int(np.clip(target_route_id, 0, len(self.patrol_routes) - 1))
            self.target.route_index = int(self.target_route_id)
            self.target.patrol_waypoints = self.patrol_routes[self.target_route_id]

        seed_val = spec.get("seed", None)
        if seed_val is not None:
            self.task_seed = int(seed_val)
            if reset_position_rng:
                self.position_rng.seed(self.task_seed)

    def _reset_target_route_state_for_recover(self):
        """
        功能:
            recover前重置Target巡逻状态，保证固定route下每次reset一致。
        输入:
            无。
        输出:
            无。
        """
        if self.target.policy_type != "patrol":
            return
        if len(self.patrol_routes) == 0:
            return
        route_id = int(np.clip(self.target_route_id, 0, len(self.patrol_routes) - 1))
        self.target.route_index = route_id
        self.target.route_episode_count = 0
        self.target.patrol_index = 0
        self.target.patrol_waypoints = self.patrol_routes[route_id]

    def _sample_initial_positions(self):
        """
        功能:
            采样所有agent初始位置。
        输入:
            无。
        输出:
            np.ndarray: shape=(agent_num,2)。
        """
        # Step 1: 常规模式下全部agent在全图均匀随机采样。
        if not bool(self.hunters_in_zone):
            return self.position_rng.uniform(
                -self.world_size + self.collision_dis * 2, self.world_size - self.collision_dis * 2, size=(self.agent_num, 2)
            ).astype(np.float32)

        # Step 2: hunters_in_zone模式下，Hunter采用“固定阵列偏移 + 随机zone中心”。
        init_positions = np.zeros((self.agent_num, 2), dtype=np.float32)
        active_ids = [hid for hid in range(self.num_hunters) if bool(self.active_hunter_mask[hid])]
        active_count = int(max(1, len(active_ids)))
        total_slots = int(self.hunter_zone_offsets.shape[0])

        # 每次reset都随机重排Hunter与槽位映射；recover通过重置随机种子保证结果可复现。
        slot_indices = self.position_rng.permutation(total_slots).astype(np.int32)[:active_count]

        zone_center = self._sample_hunter_zone_center(slot_indices=slot_indices)
        target_pos = self._sample_target_position_with_zone_constraint(zone_center)
        init_positions[self.target_index] = target_pos
        for local_idx, hid in enumerate(active_ids):
            offset_idx = int(slot_indices[local_idx]) if local_idx < len(slot_indices) else 0
            init_positions[int(hid)] = zone_center + self.hunter_zone_offsets[offset_idx]

        # 非active hunter位置仅用于占位，不影响episode行为。
        for hid in range(self.num_hunters):
            if bool(self.active_hunter_mask[hid]):
                continue
            init_positions[hid] = zone_center

        ws = float(self.world_size)
        init_positions[:, 0] = np.clip(init_positions[:, 0], -ws, ws)
        init_positions[:, 1] = np.clip(init_positions[:, 1], -ws, ws)
        return init_positions.astype(np.float32)

    def _build_hunter_zone_offsets(self, max_hunters_num: int, spacing: float) -> np.ndarray:
        """
        功能:
            预计算Hunter固定排布阵列的相对偏移（以zone中心为原点）。
        输入:
            max_hunters_num (int): 最大Hunter数量。
            spacing (float): 阵列最小间距（米）。
        输出:
            np.ndarray: shape=(m*m,2)，m=ceil(sqrt(max_hunters_num)) 的方阵槽位偏移。
        """
        n = int(max(1, max_hunters_num))
        step = float(max(1e-6, spacing))

        # Step 1: 生成不小于max_hunters_num且最接近的平方槽位数m*m。
        side = int(np.ceil(np.sqrt(float(n))))
        slots = int(side * side)

        # Step 2: 构造方阵槽位并以方阵中心为原点。
        center_shift = 0.5 * float(side - 1)
        offsets = []
        for row in range(side):
            for col in range(side):
                dx = (float(col) - center_shift) * step
                dy = (float(row) - center_shift) * step
                offsets.append((dx, dy))
        offsets = np.asarray(offsets[:slots], dtype=np.float32)

        # Step 3: 按离中心距离排序，便于不同active数量下优先取紧凑阵型。
        order = np.argsort(np.asarray([float(x * x + y * y) for x, y in offsets], dtype=np.float32))
        offsets = offsets[order]
        return offsets

    def _sample_hunter_zone_center(self, slot_indices: np.ndarray) -> np.ndarray:
        """
        功能:
            在地图内随机采样Hunter阵列中心，保证阵列整体尽量不越界。
        输入:
            slot_indices (np.ndarray): 当前active hunters对应的槽位索引。
        输出:
            np.ndarray: shape=(2,)，zone中心坐标。
        """
        ws = float(self.world_size)
        if self.hunter_zone_offsets is None or self.hunter_zone_offsets.shape[0] <= 0:
            return self.position_rng.uniform(-ws, ws, size=(2,)).astype(np.float32)

        # Step 1: 仅根据当前实际使用槽位偏移计算边界余量。
        if slot_indices is None or len(slot_indices) == 0:
            used_offsets = self.hunter_zone_offsets[:1]
        else:
            safe_idx = np.asarray(slot_indices, dtype=np.int32)
            safe_idx = np.clip(safe_idx, 0, self.hunter_zone_offsets.shape[0] - 1)
            used_offsets = self.hunter_zone_offsets[safe_idx]
        min_x = float(np.min(used_offsets[:, 0]))
        max_x = float(np.max(used_offsets[:, 0]))
        min_y = float(np.min(used_offsets[:, 1]))
        max_y = float(np.max(used_offsets[:, 1]))

        # Step 2: 计算可采样中心区间，若区间退化则回退到原点。
        low_x = -ws - min_x
        high_x = ws - max_x
        low_y = -ws - min_y
        high_y = ws - max_y
        if low_x > high_x or low_y > high_y:
            return np.zeros(2, dtype=np.float32)

        cx = float(self.position_rng.uniform(low_x, high_x))
        cy = float(self.position_rng.uniform(low_y, high_y))
        return np.asarray([cx, cy], dtype=np.float32)

    def _sample_target_position_with_zone_constraint(self, zone_center: np.ndarray) -> np.ndarray:
        """
        功能:
            采样Target初始位置；启用约束时，保证与Hunter zone中心距离不小于最小阈值。
        输入:
            zone_center (np.ndarray): Hunter阵列中心，shape=(2,)。
        输出:
            np.ndarray: Target初始位置，shape=(2,)。
        """
        ws = float(self.world_size)
        if (not bool(self.hunters_in_zone)) or (not bool(self.target_avoid_hunter_zone)):
            return self.position_rng.uniform(-ws, ws, size=(2,)).astype(np.float32)

        min_dis = float(max(0.0, self.target_hunter_zone_min_dis))
        max_retry = 10
        candidate = None
        for _ in range(max_retry):
            sampled = self.position_rng.uniform(-ws, ws, size=(2,)).astype(np.float32)
            candidate = sampled
            if float(np.linalg.norm(sampled - np.asarray(zone_center, dtype=np.float32))) >= min_dis:
                return sampled
        return candidate if candidate is not None else np.zeros(2, dtype=np.float32)

    def _is_valid_initial_positions(self, init_positions: np.ndarray) -> bool:
        """
        功能:
            校验初始位置是否满足“Target的capture_dis范围内无active Hunter”约束。
        输入:
            init_positions (np.ndarray): 候选初始位置，shape=(agent_num,2)。
        输出:
            bool: True表示满足约束，False表示需要重采样。
        """
        # Step 1: 形状保护，异常输入直接判定为无效。
        if init_positions is None or init_positions.shape != (self.agent_num, 2):
            return False

        # Step 2: 遍历active Hunter并判断是否落入Target捕获半径。
        target_pos = np.asarray(init_positions[self.target_index], dtype=np.float32)
        capture_dis_safe = max(float(self.capture_dis), 0.0)
        for hid in range(self.num_hunters):
            if not bool(self.active_hunter_mask[hid]):
                continue
            hunter_pos = np.asarray(init_positions[hid], dtype=np.float32)
            dist = float(np.linalg.norm(hunter_pos - target_pos))
            if dist <= capture_dis_safe:
                return False
        return True

    def _sample_regen_task_spec(self):
        """
        功能:
            按当前regen_scope从配置中采样一组任务规格。
        输入:
            无。
        输出:
            dict: 采样得到的任务规格。
        """
        split_cfg = self.train_split_cfg
        if not bool(split_cfg.enable):
            return {
                "num_hunters": int(self.active_num_hunters),
                "hunters_in_zone": bool(self.hunters_in_zone),
                "world_size": float(self.world_size),
                "target_policy_source": str(self.target.policy_type),
                "target_patrol_path": str(self.default_target_patrol_path),
                "target_patrol_names": list(self.default_target_patrol_names),
                "target_route_id": int(self.target_route_id),
                "seed": self.base_seed if self.task_seed is None else int(self.task_seed),
            }

        hunter_choices = list(split_cfg.hunter_count_choices)
        raw_zone_choices = list(getattr(split_cfg, "hunters_in_zone_choices", [self.default_hunters_in_zone]))
        hunters_in_zone_choices = []
        for raw in raw_zone_choices:
            if isinstance(raw, bool):
                hunters_in_zone_choices.append(bool(raw))
                continue
            if isinstance(raw, (int, np.integer)):
                hunters_in_zone_choices.append(bool(int(raw) != 0))
                continue
            if isinstance(raw, str):
                s = raw.strip().lower()
                if s in {"1", "true", "t", "yes", "y", "on"}:
                    hunters_in_zone_choices.append(True)
                    continue
                if s in {"0", "false", "f", "no", "n", "off"}:
                    hunters_in_zone_choices.append(False)
                    continue
            hunters_in_zone_choices.append(bool(raw))
        if len(hunters_in_zone_choices) == 0:
            hunters_in_zone_choices = [bool(self.default_hunters_in_zone)]
        policy_choices = [str(x).lower() for x in list(split_cfg.target_policy_choices)]
        patrol_choices = list(split_cfg.patrol_name_choices)
        seed_range = list(split_cfg.seed_range)

        num_hunters = int(self.rng.choice(hunter_choices))
        hunters_in_zone = bool(self.rng.choice(hunters_in_zone_choices))
        target_policy_source = str(self.rng.choice(policy_choices))

        target_patrol_names = list(self.default_target_patrol_names)
        target_route_id = 0
        if target_policy_source == "patrol":
            if len(patrol_choices) > 0:
                chosen_name = str(self.rng.choice(patrol_choices))
                target_patrol_names = [chosen_name]
            target_route_id = 0

        seed_min = int(min(seed_range[0], seed_range[1]))
        seed_max = int(max(seed_range[0], seed_range[1]))
        seed_val = int(self.rng.randint(seed_min, seed_max + 1))

        return {
            "num_hunters": int(np.clip(num_hunters, 1, self.num_hunters)),
            "hunters_in_zone": bool(hunters_in_zone),
            "world_size": float(self.world_size),
            "target_policy_source": str(target_policy_source),
            "target_patrol_path": str(self.default_target_patrol_path),
            "target_patrol_names": list(target_patrol_names),
            "target_route_id": int(target_route_id),
            "seed": int(seed_val),
        }

    def step(self, actions):
        """
        功能:
            推进环境一步，完成动作执行、奖励计算和终止判定。
        输入:
            actions (np.ndarray | list): shape=(agent_num,2)，归一化动作。
        输出:
            list: [obs, rewards, dones, infos]。
        """
        # 步骤1：汇总每个agent的最终执行动作
        raw_actions = np.asarray(actions, dtype=np.float32).reshape(self.agent_num, 2)

        selected_actions = []
        for i, agent in enumerate(self.agents):
            policy_action = raw_actions[i] if agent.policy_type == "learn" else None
            if isinstance(agent, TargetAgent):
                selected_actions.append(
                    agent.select_action(
                        self.step_count,
                        policy_action,
                        self.rng,
                        hunters=self.hunters,
                        active_hunter_mask=self.active_hunter_mask,
                    )
                )
            else:
                selected_actions.append(agent.select_action(self.step_count, policy_action, self.rng))

        # 步骤2：执行运动学更新
        for agent, action in zip(self.agents, selected_actions):
            agent.step(action, self.dt, self.world_size)

        # 步骤3：碰撞、可见性、记忆更新、捕获判定
        target_collided, collision_rewards = self._handle_collision()
        self.last_target_collided = bool(target_collided)
        team_sees_target = self._team_sees_target()
        self._update_shared_target_memory(team_sees_target)

        captured = False
        if self.target.alive and not target_collided:
            captured = self._update_capture_counter()
            if captured:
                self.target.alive = False
                self.target.velocity[:] = 0.0
                self.last_episode_captured = True
                if self.last_capture_step is None:
                    self.last_capture_step = int(self.step_count + 1)

        # 步骤4：奖励聚合（通过统一函数返回总奖励与子项）
        rewards, reward_terms = self._compute_rewards(captured, collision_rewards)
        active_hunter_rewards = [
            float(rewards[i]) for i in range(self.num_hunters) if bool(self.active_hunter_mask[i])
        ]
        hunter_reward_mean = float(np.mean(active_hunter_rewards)) if len(active_hunter_rewards) > 0 else 0.0
        target_reward_value = float(rewards[self.target_index]) if self.target_index < rewards.shape[0] else 0.0
        self.hunter_reward_sum += hunter_reward_mean
        self.target_reward_sum += target_reward_value
        self.reward_step_count += 1
        self.hunter_reward_last = hunter_reward_mean
        self.target_reward_last = target_reward_value

        # 步骤5：终止条件判定
        self.step_count += 1
        timeout = self.step_count >= self.max_steps
        all_hunters_dead = not any(
            bool(self.active_hunter_mask[i]) and bool(h.alive) for i, h in enumerate(self.hunters)
        )
        episode_end = timeout or captured or target_collided or all_hunters_dead

        if episode_end:
            self.done[:] = True
        else:
            self.done = np.array([not a.alive for a in self.agents], dtype=bool)

        # 步骤6：打包并返回标准接口数据
        obs = self._build_obs(team_sees_target)
        rews = [[float(r)] for r in rewards]
        dones = [bool(d) for d in self.done]
        infos = [
            {
                "role": a.role,
                "alive": bool(a.alive),
                "captured": bool(captured),
                "target_collided": bool(target_collided),
                "team_sees_target": bool(team_sees_target),
                "reward_total": float(reward_terms["total"][i]),
                "reward_collision": float(reward_terms["collision_reward"][i]),
                "reward_speed_penalty": float(reward_terms["speed_penalty_reward"][i]),
                "reward_hunter_base": float(reward_terms["hunter_base_reward"][i]),
                "reward_target_base": float(reward_terms["target_base_reward"][i]),
                "reward_hunter_streak": float(reward_terms["hunter_streak_reward"][i]),
                "reward_target_streak": float(reward_terms["target_streak_reward"][i]),
                "reward_capture": float(reward_terms["capture_reward"][i]),
                "reward_escape_gap": float(reward_terms["escape_gap_reward"][i]),
                "reward_escape_gap_hunter": float(reward_terms["escape_gap_hunter_reward"][i]),
                "reward_escape_gap_target": float(reward_terms["escape_gap_target_reward"][i]),
                "reward_escape_gap_encircle": float(reward_terms["escape_gap_encircle_reward"][i]),
                "reward_escape_gap_intercept": float(reward_terms["escape_gap_intercept_reward"][i]),
                "reward_escape_gap_encircle_hunter": float(reward_terms["escape_gap_encircle_hunter_reward"][i]),
                "reward_escape_gap_encircle_target": float(reward_terms["escape_gap_encircle_target_reward"][i]),
                "reward_escape_gap_intercept_hunter": float(reward_terms["escape_gap_intercept_hunter_reward"][i]),
                "reward_escape_gap_intercept_target": float(reward_terms["escape_gap_intercept_target_reward"][i]),
                "max_escape_gap_angle": float(self.target.max_escape_gap_angle),
                "max_escape_gap_center_angle": float(self.target.max_escape_gap_center_angle),
                "max_escape_gap_metric_valid": bool(self.target.escape_gap_metric_valid),
                "escape_gap_encircle_score": float(self.target.last_escape_gap_encircle_score),
                "escape_gap_open_score": float(self.target.last_escape_gap_open_score),
                "escape_gap_hunter_direction_score": float(self.target.last_escape_gap_hunter_direction_score),
                "escape_gap_target_direction_score": float(self.target.last_escape_gap_target_direction_score),
                "coord_self_is_topk": float(
                    self.last_coord_summary_cache[i, 0]
                    if (
                        bool(self.coord_summary_obs_enable)
                        and self.last_coord_summary_cache.shape[1] >= 1
                    )
                    else 0.0
                ),
                "coord_hunters_in_escape_radius_count": float(
                    self.last_coord_summary_cache[i, 1]
                    if (
                        bool(self.coord_summary_obs_enable)
                        and self.last_coord_summary_cache.shape[1] >= 2
                    )
                    else 0.0
                ),
                "active_agent": bool(
                    True if a.role != "hunter" else self.active_hunter_mask[int(i)]
                ),
            }
            for i, a in enumerate(self.agents)
        ]
        return [obs, rews, dones, infos]

    def render(self, mode="rgb_array", title=None):
        """
        功能:
            渲染当前场景，用于训练/评估GIF生成。
        输入:
            mode (str): 渲染模式，支持\"rgb_array\"与\"human\"。
            title (str | None): 兼容参数，当前渲染标题由环境状态自动生成。
        输出:
            np.ndarray | None: mode为rgb_array时返回RGB图像；human时返回None。
        """
        # Step 1: 仅支持两种渲染模式
        if mode not in ("rgb_array", "human"):
            raise NotImplementedError(f"Unsupported render mode: {mode}")

        # Step 2: 创建画布并设置坐标范围
        fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
        ws = float(self.world_size)
        ax.set_xlim(-ws, ws)
        ax.set_ylim(-ws, ws)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.3)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # Step 3: 标题信息（显示当前episode进度、瞬时奖励与累计奖励）
        render_title = (
            f"Prog. {int(self.step_count)}/{int(self.max_steps)} | "
            f"Hut rwd {float(self.hunter_reward_last):.3f} / {float(self.hunter_reward_sum):.3f} | "
            f"Tgt rwd {float(self.target_reward_last):.3f} / {float(self.target_reward_sum):.3f}"
        )
        ax.set_title(render_title)

        # Step 4: 绘制Target完整巡逻轨迹（patrol模式）
        if self.target.policy_type == "patrol" and len(self.target.patrol_waypoints) > 0:
            patrol_points = np.asarray(self.target.patrol_waypoints, dtype=np.float32)
            if patrol_points.ndim == 2 and patrol_points.shape[0] > 0:
                ax.plot(
                    patrol_points[:, 0],
                    patrol_points[:, 1],
                    color="#9467bd",
                    linestyle="-",
                    linewidth=1.4,
                    alpha=0.65,
                )
                if patrol_points.shape[0] > 1:
                    ax.plot(
                        [float(patrol_points[-1, 0]), float(patrol_points[0, 0])],
                        [float(patrol_points[-1, 1]), float(patrol_points[0, 1])],
                        color="#9467bd",
                        linestyle="-",
                        linewidth=1.0,
                        alpha=0.45,
                    )
                ax.scatter(
                    patrol_points[:, 0],
                    patrol_points[:, 1],
                    c=["#9467bd"],
                    marker="x",
                    s=26,
                    alpha=0.75,
                )

        # Step 5: 绘制轨迹渐隐、当前位置、速度向量、感知半径和碰撞半径
        hunter_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#17becf", "#8c564b"]
        hunter_idx = 0
        velocity_arrow_len = max(1e-6, float(self.world_size) * 0.08)
        for agent_idx, agent in enumerate(self.agents):
            if agent.role == "hunter" and not bool(self.active_hunter_mask[agent_idx]):
                continue
            if agent.role == "hunter":
                color = hunter_colors[hunter_idx % len(hunter_colors)]
                radius = float(self.hunter_perception_radius)
                marker = "o"
                hunter_idx += 1
            else:
                color = "#d62728"
                radius = float(self.target_perception_radius)
                marker = "s"

            traj = np.asarray(agent.trajectory, dtype=np.float32)
            if len(traj) >= 2:
                for i in range(1, len(traj)):
                    alpha = max(0.08, float(i) / float(len(traj)))
                    ax.plot(
                        traj[i - 1 : i + 1, 0],
                        traj[i - 1 : i + 1, 1],
                        color=color,
                        linewidth=2.0,
                        alpha=0.65 * alpha,
                    )

            pos = np.asarray(agent.position, dtype=np.float32)
            alive = bool(agent.alive)
            ax.scatter(
                [pos[0]],
                [pos[1]],
                c=[color],
                marker=marker,
                s=70,
                alpha=1.0 if alive else 0.35,
                edgecolors="black",
                linewidths=0.5,
            )
            if radius >= 0:
                ax.add_patch(
                    plt.Circle(
                        (float(pos[0]), float(pos[1])),
                        radius,
                        color=color,
                        fill=False,
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.30 if alive else 0.15,
                    )
                )
            if agent.role != "target" and float(agent.safe_dis) > 0.0:
                ax.add_patch(
                    plt.Circle(
                        (float(pos[0]), float(pos[1])),
                        float(agent.safe_dis),
                        color="#7f7f7f",
                        fill=True,
                        linestyle="-.",
                        linewidth=1.0,
                        alpha=0.30 if alive else 0.12,
                    )
                )
            if agent.role == "target":
                ax.add_patch(
                    plt.Circle(
                        (float(pos[0]), float(pos[1])),
                        float(self.capture_dis),
                        color="#bcbd22",
                        fill=False,
                        linestyle="-.",
                        linewidth=1.1,
                        alpha=0.45 if alive else 0.2,
                    )
                )
            ax.add_patch(
                plt.Circle(
                    (float(pos[0]), float(pos[1])),
                    float(self.collision_dis),
                    color="black",
                    fill=False,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.20 if alive else 0.10,
                )
            )

            vel = np.asarray(agent.velocity, dtype=np.float32)
            v_norm = float(np.linalg.norm(vel))
            heading = np.asarray(agent.heading, dtype=np.float32)
            h_norm = float(np.linalg.norm(heading))
            if h_norm < 1e-8:
                heading = np.array([1.0, 0.0], dtype=np.float32)
            else:
                heading = heading / h_norm
            vec = heading * velocity_arrow_len
            ax.arrow(
                float(pos[0]),
                float(pos[1]),
                float(vec[0]),
                float(vec[1]),
                width=max(0.02 * self.collision_dis, 0.05),
                head_width=max(0.22 * self.collision_dis, 0.35),
                head_length=max(0.28 * self.collision_dis, 0.45),
                length_includes_head=True,
                color=color,
                alpha=0.85 if alive else 0.35,
            )
            ax.text(
                float(pos[0] + vec[0]),
                float(pos[1] + vec[1]),
                f"{v_norm:.1f}",
                fontsize=7,
                color=color,
                alpha=0.9 if alive else 0.5,
            )

            turn_active = (
                agent.control_mode == "velocity"
                and v_norm > float(agent.min_turn_limit_velo)
                and float(agent.max_turn_angle_rad) < np.pi
            )
            if turn_active:
                theta = float(np.arctan2(heading[1], heading[0]))
                left_theta = theta + float(agent.max_turn_angle_rad)
                right_theta = theta - float(agent.max_turn_angle_rad)
                for th in (left_theta, right_theta):
                    turn_vec = np.array([np.cos(th), np.sin(th)], dtype=np.float32) * velocity_arrow_len
                    ax.plot(
                        [float(pos[0]), float(pos[0] + turn_vec[0])],
                        [float(pos[1]), float(pos[1] + turn_vec[1])],
                        color=color,
                        linestyle="--",
                        linewidth=0.9,
                        alpha=0.45 if alive else 0.2,
                    )

        # Step 6: 绘制围捕几何（escape_radius、拦截区间、最大可逃脱扇区）
        gap_info = self.target.compute_max_escape_gap(self.hunters, self.active_hunter_mask)
        if bool(gap_info.get("metric_valid", False)) and int(gap_info.get("active_alive_hunters", 0)) > 1:
            target_pos = np.asarray(self.target.position, dtype=np.float32)
            escape_dis = float(max(0.0, self.target.escape_dis))
            if escape_dis > 0.0:
                ax.add_patch(
                    plt.Circle(
                        (float(target_pos[0]), float(target_pos[1])),
                        escape_dis,
                        color="#ff9896",
                        fill=False,
                        linestyle="-",
                        linewidth=1.0,
                        alpha=0.50,
                    )
                )

            for hid in list(gap_info.get("encircling_hunter_ids", [])):
                if hid < 0 or hid >= self.num_hunters:
                    continue
                segment = self.hunters[hid].compute_intercept_segment(target_pos)
                if segment is None:
                    continue
                p1, p2 = segment
                ax.plot(
                    [float(p1[0]), float(p2[0])],
                    [float(p1[1]), float(p2[1])],
                    color="#ff7f0e",
                    linestyle="-",
                    linewidth=1.4,
                    alpha=0.75,
                )

            max_gap_angle = float(gap_info.get("max_gap_angle", 0.0))
            if escape_dis > 0.0 and max_gap_angle > 1e-6:
                start_rad = float(gap_info.get("max_gap_start_angle", 0.0))
                theta1_deg = float(np.degrees(start_rad))
                theta2_deg = float(theta1_deg + np.degrees(max_gap_angle))
                ax.add_patch(
                    Wedge(
                        center=(float(target_pos[0]), float(target_pos[1])),
                        r=escape_dis,
                        theta1=theta1_deg,
                        theta2=theta2_deg,
                        facecolor="#ff9896",
                        edgecolor="#d62728",
                        linewidth=1.0,
                        alpha=0.22,
                    )
                )

        # Step 7: 绘制Pursuit组接收到的Target位置（共享记忆）
        if self.shared_target_valid:
            shared_pos = np.asarray(self.shared_target_pos, dtype=np.float32)
            ax.scatter(
                [shared_pos[0]],
                [shared_pos[1]],
                c=["#e377c2"],
                marker="*",
                s=140,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.5,
            )
            for hid, hunter in enumerate(self.hunters):
                if not bool(self.active_hunter_mask[hid]):
                    continue
                if not hunter.alive:
                    continue
                ax.plot(
                    [hunter.position[0], shared_pos[0]],
                    [hunter.position[1], shared_pos[1]],
                    color="#e377c2",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.25,
                )

        # Step 8: 绘制碰撞/抓捕事件提示
        if self.last_target_collided:
            tp = np.asarray(self.target.position, dtype=np.float32)
            ax.scatter([tp[0]], [tp[1]], c=["black"], marker="X", s=120, alpha=0.95)
            ax.text(tp[0], tp[1], " COLL", fontsize=7, color="black")
        if self.last_episode_captured and self.last_capture_step is not None:
            tp = np.asarray(self.target.position, dtype=np.float32)
            ax.scatter(
                [tp[0]],
                [tp[1]],
                c=["gold"],
                marker="*",
                s=180,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.6,
            )
            ax.text(tp[0], tp[1], " CAP", fontsize=7, color="#8c6d1f")
        for i, j in self.last_collision_pairs:
            pi = np.asarray(self.agents[i].position, dtype=np.float32)
            pj = np.asarray(self.agents[j].position, dtype=np.float32)
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color="black", linestyle="-.", linewidth=1.0, alpha=0.5)
        for agent_id in self.last_boundary_collision_agents:
            if agent_id < 0 or agent_id >= self.agent_num:
                continue
            p = np.asarray(self.agents[agent_id].position, dtype=np.float32)
            ax.scatter(
                [p[0]],
                [p[1]],
                c=["#8c564b"],
                marker="D",
                s=80,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.6,
            )
            ax.text(p[0], p[1], " BND", fontsize=7, color="#8c564b")

        # Step 9: 绘制图例
        legend_handles = []
        for hid in range(self.num_hunters):
            if not bool(self.active_hunter_mask[hid]):
                continue
            h_color = hunter_colors[hid % len(hunter_colors)]
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=h_color,
                    markeredgecolor="black",
                    markersize=7,
                    label=f"Hunter-{hid}",
                )
            )
        target_policy_type = str(self.target.policy_type).lower()
        if target_policy_type == "patrol":
            route_name = "NA"
            if len(self.active_target_patrol_names) > 0 and len(self.patrol_routes) > 0:
                route_id = int(np.clip(self.target_route_id, 0, len(self.active_target_patrol_names) - 1))
                route_name = str(self.active_target_patrol_names[route_id])
            target_label = f"Target ({target_policy_type}:{route_name})"
        else:
            target_label = f"Target ({target_policy_type})"

        legend_handles.extend(
            [
                Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728", markeredgecolor="black", markersize=7, label=target_label),
                Line2D([0], [0], color="#1f77b4", lw=1.0, linestyle=":", label="Perception Radius"),
                Line2D([0], [0], color="#7f7f7f", lw=1.0, linestyle="-.", label="Safe Distance"),
                Line2D([0], [0], color="#bcbd22", lw=1.1, linestyle="-.", label="Target Capture Range"),
                Line2D([0], [0], color="black", lw=1.0, linestyle="--", label="Collision Radius"),
                Line2D([0], [0], color="#1f77b4", lw=1.8, marker=">", markersize=6, label="Velocity Vector"),
                Line2D([0], [0], color="#1f77b4", lw=1.0, linestyle="--", label="Turn Limit"),
                Line2D([0], [0], color="#9467bd", lw=1.4, linestyle="-", label="Patrol Route"),
                Line2D([0], [0], color="#ff9896", lw=1.0, linestyle="-", label="Escape Radius"),
                Line2D([0], [0], color="#ff7f0e", lw=1.4, linestyle="-", label="Intercept Segment"),
                Line2D([0], [0], color="#d62728", lw=1.0, linestyle="-", label="Max Escape Gap"),
                Line2D([0], [0], marker="*", color="w", markerfacecolor="#e377c2", markeredgecolor="black", markersize=9, label="Shared Target Pos"),
                Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="black", markersize=9, label="Capture Event"),
                Line2D([0], [0], marker="X", color="w", markerfacecolor="black", markeredgecolor="black", markersize=8, label="Collision Event"),
                Line2D([0], [0], marker="D", color="w", markerfacecolor="#8c564b", markeredgecolor="black", markersize=8, label="Boundary Collision"),
            ]
        )
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            framealpha=0.85,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(right=0.72)

        # Step 10: 返回RGB数组或直接展示
        if mode == "human":
            plt.show(block=False)
            plt.pause(0.001)
            plt.close(fig)
            return None

        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
        plt.close(fig)
        return image

    def close(self):
        """
        功能:
            关闭环境绘图资源。
        输入:
            无。
        输出:
            无。
        """
        plt.close("all")

    def _team_sees_target(self):
        """
        功能:
            判断本步是否有Hunter观测到Target。
        输入:
            无。
        输出:
            bool: True表示至少一个Hunter可见Target。
        """
        if not self.target.alive:
            return False
        for hid, h in enumerate(self.hunters):
            if not bool(self.active_hunter_mask[hid]):
                continue
            if not h.alive:
                continue
            if self.hunter_perception_radius < 0:
                return True
            if np.linalg.norm(h.position - self.target.position) <= self.hunter_perception_radius:
                return True
        return False

    def _update_shared_target_memory(self, team_sees_target):
        """
        功能:
            更新追捕组共享的Target状态记忆。
        输入:
            team_sees_target (bool): 本步是否真实观测到Target。
        输出:
            无。
        """
        if not self.target.alive:
            self.shared_target_valid = False
            self.last_seen_age = 0
            return

        if team_sees_target or self.step_count < self.target_pos_init_guidance_step:
            self.shared_target_pos = self.target.position.copy()
            self.shared_target_vel = self.target.velocity.copy()
            self.shared_target_valid = True
            self.last_seen_age = 0
            return

        if not self.target_pos_guidance:
            self.shared_target_valid = False
            self.last_seen_age += 1
            return

        pos_std = self.world_size * self.noisy_target_pos_std / 100.0
        vel_std = self.target.max_speed * self.noisy_target_vel_std / 100.0
        self.shared_target_pos = self.target.position + self.rng.normal(0, pos_std, size=2).astype(np.float32)
        self.shared_target_vel = self.target.velocity + self.rng.normal(0, vel_std, size=2).astype(np.float32)
        self.shared_target_pos = np.clip(self.shared_target_pos, -self.world_size, self.world_size)
        self.shared_target_valid = True
        self.last_seen_age += 1

    def _update_capture_counter(self):
        """
        功能:
            更新Hunter连续捕获计数并判定是否捕获成功。
        输入:
            无。
        输出:
            bool: True表示满足capture_step连续捕获条件。
        """
        for i, h in enumerate(self.hunters):
            if not bool(self.active_hunter_mask[i]):
                self.capture_counter[i] = 0
                continue
            if not h.alive:
                self.capture_counter[i] = 0
                continue
            d = float(np.linalg.norm(h.position - self.target.position))
            self.capture_counter[i] = self.capture_counter[i] + 1 if d <= self.capture_dis else 0
        return bool(np.any(self.capture_counter >= self.capture_step))

    def _handle_collision(self):
        """
        功能:
            扫描agent间距离与边界距离并计算风险惩罚/硬碰撞结果，同时标记失活agent。
            规则:
            - Target不参与与其他agent的两两碰撞判定；
            - 所有active agent均参与边界风险与边界碰撞判定；
            - 当Target为learn策略时，边界碰撞不死亡而是反弹，并施加边界碰撞惩罚。
        输入:
            无。
        输出:
            tuple:
                - bool: Target是否发生碰撞。
                - np.ndarray: 碰撞奖励分量，shape=(agent_num,)。
        """
        target_collided = False
        collision_pairs = []
        boundary_collision_agents = []
        agents = self.agents
        disable = [False] * self.agent_num
        collision_rewards = np.zeros(self.agent_num, dtype=np.float32)
        for i in range(self.agent_num):
            if i < self.num_hunters and (not bool(self.active_hunter_mask[i])):
                continue
            if not agents[i].alive: # 已发生碰撞的Agent不重新处理碰撞
                continue

            # Step 1: 与边界的风险惩罚和硬碰撞判定
            boundary_dist = self._distance_to_nearest_boundary(agents[i].position)
            collision_rewards[i] -= _safe_distance_penalty(
                dist=boundary_dist,
                safe_dis=float(agents[i].safe_dis),
                collision_dis=float(self.collision_dis),
                collision_penalty_k=float(self.collision_penalty_k),
                safe_zone_penalty_scale=float(self.safe_zone_penalty_scale),
            )
            if boundary_dist <= float(self.collision_dis):
                boundary_collision_agents.append(int(i))
                if i == self.target_index:
                    collision_rewards[i] -= float(self.target_collision_penalty)
                    if str(self.target.policy_type).lower() == "learn":
                        self._bounce_target_from_boundary()
                    else:
                        disable[i] = True
                        target_collided = True
                else:
                    disable[i] = True

            for j in range(i + 1, self.agent_num):
                if j < self.num_hunters and (not bool(self.active_hunter_mask[j])):
                    continue
                if not agents[j].alive: # 已发生碰撞的Agent不重新处理碰撞
                    continue
                if i == self.target_index or j == self.target_index:
                    continue

                dist = float(np.linalg.norm(agents[i].position - agents[j].position))

                # Step 2: 距离进入safe_dis即开始风险惩罚，越接近collision_dis惩罚越大
                collision_rewards[i] -= _safe_distance_penalty(
                    dist=dist,
                    safe_dis=float(agents[i].safe_dis),
                    collision_dis=float(self.collision_dis),
                    collision_penalty_k=float(self.collision_penalty_k),
                    safe_zone_penalty_scale=float(self.safe_zone_penalty_scale),
                )
                collision_rewards[j] -= _safe_distance_penalty(
                    dist=dist,
                    safe_dis=float(agents[j].safe_dis),
                    collision_dis=float(self.collision_dis),
                    collision_penalty_k=float(self.collision_penalty_k),
                    safe_zone_penalty_scale=float(self.safe_zone_penalty_scale),
                )

                # Step 3: 小于collision_dis直接触发硬碰撞
                if dist <= self.collision_dis:
                    collision_pairs.append((int(i), int(j)))
                    disable[i] = True
                    disable[j] = True

        for idx, d in enumerate(disable):
            if d:
                agents[idx].alive = False
                agents[idx].velocity[:] = 0.0
        if self.collision_penalty_cap > 0:
            collision_rewards = np.maximum(collision_rewards, -float(self.collision_penalty_cap))
        self.last_collision_pairs = collision_pairs
        self.last_boundary_collision_agents = boundary_collision_agents
        return target_collided, collision_rewards

    def _get_capture_success_hunter_ids(self):
        """
        功能:
            获取本步触发捕获条件的Hunter索引集合（active且alive且capture_counter达阈值）。
        输入:
            无。
        输出:
            List[int]: 成功触发捕获条件的Hunter索引列表。
        """
        captor_ids = []
        for hid in range(self.num_hunters):
            if not bool(self.active_hunter_mask[hid]):
                continue
            if not bool(self.hunters[hid].alive):
                continue
            if int(self.capture_counter[hid]) >= int(self.capture_step):
                captor_ids.append(int(hid))
        return captor_ids

    def _get_encircle_hunter_ids_for_capture(self):
        """
        功能:
            获取捕获时参与围捕几何的Hunter索引（active且alive，优先使用escape_radius筛选）。
        输入:
            无。
        输出:
            List[int]: 参与围捕的Hunter索引列表。
        """
        encircle_ids = []
        escape_radius = float(getattr(self.target, "escape_dis", 0.0))
        for hid in range(self.num_hunters):
            if not bool(self.active_hunter_mask[hid]):
                continue
            hunter = self.hunters[hid]
            if not bool(hunter.alive):
                continue
            dist = float(np.linalg.norm(np.asarray(hunter.position) - np.asarray(self.target.position)))
            if escape_radius > 0.0 and dist > escape_radius:
                continue
            encircle_ids.append(int(hid))

        if len(encircle_ids) == 0:
            encircle_ids = self._get_capture_success_hunter_ids()
        return encircle_ids

    def _compute_capture_encircle_quality(self, encircle_hunter_ids):
        """
        功能:
            基于围捕Hunter在Target周围的角分布，计算包围质量分数（0~1）。
        输入:
            encircle_hunter_ids (List[int]): 参与围捕的Hunter索引列表。
        输出:
            float: 包围质量分数，1表示角覆盖更均匀、最大缺口更小。
        """
        if encircle_hunter_ids is None or len(encircle_hunter_ids) <= 1:
            return 0.0

        target_pos = np.asarray(self.target.position, dtype=np.float32)
        angles = []
        for hid in encircle_hunter_ids:
            if hid < 0 or hid >= self.num_hunters:
                continue
            rel = np.asarray(self.hunters[int(hid)].position, dtype=np.float32) - target_pos
            if float(np.linalg.norm(rel)) <= 1e-8:
                continue
            angles.append(float(np.arctan2(rel[1], rel[0])))
        if len(angles) <= 1:
            return 0.0

        angles = np.sort(np.asarray(angles, dtype=np.float32))
        wrapped = np.concatenate([angles, angles[:1] + 2.0 * np.pi], axis=0)
        gaps = np.diff(wrapped)
        max_gap = float(np.max(gaps)) if gaps.size > 0 else float(2.0 * np.pi)
        quality = 1.0 - float(np.clip(max_gap / (2.0 * np.pi), 0.0, 1.0))
        return float(np.clip(quality, 0.0, 1.0))

    def _assign_capture_reward(self, capture_reward):
        """
        功能:
            按配置策略将捕获奖励分配给Hunter，并对Target施加捕获惩罚。
        输入:
            capture_reward (np.ndarray): 捕获奖励分量，shape=(agent_num,)。
        输出:
            无（原地写入capture_reward数组）。
        """
        captor_ids = self._get_capture_success_hunter_ids()
        if len(captor_ids) == 0:
            capture_reward[self.target_index] = -self.target_captured_penalty
            return

        mode = str(self.capture_reward_allocation)
        if mode == "team":
            for hid in range(self.num_hunters):
                if not bool(self.active_hunter_mask[hid]):
                    continue
                if not bool(self.hunters[hid].alive):
                    continue
                capture_reward[int(hid)] = float(self.hunter_capture_reward)
        elif mode == "alone":
            for hid in captor_ids:
                capture_reward[int(hid)] = float(self.hunter_capture_reward)
        else:
            for hid in captor_ids:
                capture_reward[int(hid)] = float(self.hunter_capture_reward)

            encircle_ids = self._get_encircle_hunter_ids_for_capture()
            captor_set = set(int(hid) for hid in captor_ids)
            support_ids = [int(hid) for hid in encircle_ids if int(hid) not in captor_set]
            if len(support_ids) > 0:
                quality = float(self._compute_capture_encircle_quality(encircle_ids))
                if quality > 0.0:
                    target_pos = np.asarray(self.target.position, dtype=np.float32)
                    inv_dist = []
                    for hid in support_ids:
                        dist = float(
                            np.linalg.norm(
                                np.asarray(self.hunters[int(hid)].position, dtype=np.float32) - target_pos
                            )
                        )
                        inv_dist.append(1.0 / max(dist, 1e-6))
                    inv_dist = np.asarray(inv_dist, dtype=np.float32)
                    weight_sum = float(np.max(inv_dist))
                    if weight_sum > 1e-8:
                        weights = inv_dist / weight_sum
                        support_pool = float(self.hunter_capture_reward) * float(quality)
                        for idx, hid in enumerate(support_ids):
                            capture_reward[int(hid)] = float(support_pool) * float(weights[idx])

        capture_reward[self.target_index] = -self.target_captured_penalty

    def _bounce_target_from_boundary(self):
        """
        功能:
            对learn策略Target执行边界反弹：位置拉回边界内并反射外向速度分量。
        输入:
            无。
        输出:
            无（内部更新target的位置、速度、朝向与速度标量）。
        """
        if not bool(self.target.alive):
            return

        ws = float(max(1e-6, self.world_size))
        bounce_margin = float(min(ws * 0.5, max(float(self.collision_dis) + 1e-3, 1e-3)))

        pos = np.asarray(self.target.position, dtype=np.float32).copy()
        vel = np.asarray(self.target.velocity, dtype=np.float32).copy()

        # Step 1: 反射撞墙方向上的外向速度分量。
        if pos[0] >= ws - bounce_margin and vel[0] > 0.0:
            vel[0] = -abs(float(vel[0]))
        elif pos[0] <= -ws + bounce_margin and vel[0] < 0.0:
            vel[0] = abs(float(vel[0]))

        if pos[1] >= ws - bounce_margin and vel[1] > 0.0:
            vel[1] = -abs(float(vel[1]))
        elif pos[1] <= -ws + bounce_margin and vel[1] < 0.0:
            vel[1] = abs(float(vel[1]))

        # Step 2: 将Target拉回到边界内侧，避免持续贴边触发硬碰撞。
        pos = np.clip(pos, -ws + bounce_margin, ws - bounce_margin)

        self.target.position = pos.astype(np.float32)
        self.target.velocity = vel.astype(np.float32)
        speed = float(np.linalg.norm(self.target.velocity))
        self.target.speed = speed
        if speed > 1e-8:
            self.target.heading = (self.target.velocity / speed).astype(np.float32)

    def _distance_to_nearest_boundary(self, position: np.ndarray) -> float:
        """
        功能:
            计算位置到正方形地图最近边界的距离（米）。
        输入:
            position (np.ndarray): 位置向量，shape=(2,)。
        输出:
            float: 到最近边界的距离（非负）。
        """
        # Step 1: 分别计算x/y方向到边界剩余距离
        ws = float(self.world_size)
        pos = np.asarray(position, dtype=np.float32)
        margin_x = ws - abs(float(pos[0]))
        margin_y = ws - abs(float(pos[1]))

        # Step 2: 返回最小非负边界距离
        return float(max(0.0, min(margin_x, margin_y)))

    def _compute_rewards(self, captured, collision_rewards):
        """
        功能:
            统一计算总奖励与各子项奖励分量。
        输入:
            captured (bool): 本步是否捕获成功。
            collision_rewards (np.ndarray): 碰撞奖励分量，shape=(agent_num,)。
        输出:
            tuple:
                - np.ndarray: 总奖励，shape=(agent_num,)。
                - dict[str, np.ndarray]: 奖励子项字典。
        """
        # Step 1: 计算基础奖励与捕获奖励（Hunter/Target共用一组系数）
        hunter_base_reward = np.zeros(self.agent_num, dtype=np.float32)
        target_base_reward = np.zeros(self.agent_num, dtype=np.float32)
        hunter_streak_reward = np.zeros(self.agent_num, dtype=np.float32)
        target_streak_reward = np.zeros(self.agent_num, dtype=np.float32)
        capture_reward = np.zeros(self.agent_num, dtype=np.float32)
        escape_gap_reward = np.zeros(self.agent_num, dtype=np.float32)
        escape_gap_hunter_reward = np.zeros(self.agent_num, dtype=np.float32)
        escape_gap_target_reward = np.zeros(self.agent_num, dtype=np.float32)
        escape_gap_encircle_reward = np.zeros(self.agent_num, dtype=np.float32)
        escape_gap_intercept_reward = np.zeros(self.agent_num, dtype=np.float32)
        escape_gap_encircle_hunter_reward = np.zeros(self.agent_num, dtype=np.float32)
        escape_gap_encircle_target_reward = np.zeros(self.agent_num, dtype=np.float32)
        escape_gap_intercept_hunter_reward = np.zeros(self.agent_num, dtype=np.float32)
        escape_gap_intercept_target_reward = np.zeros(self.agent_num, dtype=np.float32)

        dist_scale = max(float(self.world_size), 1e-6)
        capture_dis_safe = max(float(self.capture_dis), 1e-6)
        streak_cap = max(1, int(self.base_streak_cap))
        hunter_d = []
        streak_used = []

        if self.base_reward_mode == "delta_window":
            # Step 1.1: 新版base reward（每个hunter独立）
            # 用“该hunter前N步到Target的平均距离”与“当前距离”做对比，
            # 距离变小为正奖励，变大为负奖励；再按norm_scale归一化到[-1,1]。
            delta_norm_values = []
            norm_scale = max(float(self.base_delta_norm_scale), 1e-6)
            for hid, hunter in enumerate(self.hunters):
                if not bool(self.active_hunter_mask[hid]):
                    continue
                d_now = float(np.linalg.norm(hunter.position - self.target.position))
                hunter_d.append(d_now)
                hist = self.hunter_distance_histories[hid]
                # 仅在未进入capture范围时计算delta-window base reward。
                if d_now > capture_dis_safe:
                    d_prev_avg = float(np.mean(hist)) if len(hist) > 0 else d_now
                    delta_norm = float(np.clip((d_prev_avg - d_now) / norm_scale, -1.0, 1.0))
                    hunter_base_reward[hid] = float(self.base_delta_hunter_scale) * delta_norm
                    delta_norm_values.append(delta_norm)
                else:
                    hunter_base_reward[hid] = 0.0
                hist.append(d_now)

                streak_i = int(min(int(self.capture_counter[hid]), streak_cap))
                streak_used.append(streak_i)
                hunter_streak_reward[hid] = self.base_streak_scale * float(streak_i)

            # Step 1.2: Target base reward取Hunter改变量的反向均值（归一化后）。
            mean_delta_norm = float(np.mean(delta_norm_values)) if len(delta_norm_values) > 0 else 0.0
            target_base_reward[self.target_index] = -float(self.base_delta_target_scale) * mean_delta_norm
            avg_streak = float(np.mean(streak_used)) if streak_used else 0.0
            target_streak_reward[self.target_index] = -self.base_streak_scale * avg_streak
        else:
            # Step 1.1: 旧版base reward（距离分段 + streak）
            active_dist_pairs = []
            for hid, hunter in enumerate(self.hunters):
                if not bool(self.active_hunter_mask[hid]):
                    continue
                d_val = float(np.linalg.norm(hunter.position - self.target.position))
                active_dist_pairs.append((d_val, int(hid)))
            active_dist_pairs.sort(key=lambda x: x[0])
            if bool(self.base_reward_topk_enable):
                topk = int(min(max(1, self.base_reward_topk_k), len(active_dist_pairs)))
                topk_ids = {hid for _, hid in active_dist_pairs[:topk]}
            else:
                topk_ids = {hid for _, hid in active_dist_pairs}

            for i, h in enumerate(self.hunters):
                if not bool(self.active_hunter_mask[i]):
                    continue
                d = float(np.linalg.norm(h.position - self.target.position))
                hunter_d.append(d)
                is_topk_hunter = bool(int(i) in topk_ids)
                base_scale = 1.0 if is_topk_hunter else float(self.base_reward_non_topk_scale)

                if d <= capture_dis_safe:
                    near_ratio = 1.0 - d / capture_dis_safe
                    hunter_base_reward[i] = self.base_near_scale * near_ratio * base_scale
                else:
                    far_ratio = (d - capture_dis_safe) / dist_scale
                    hunter_base_reward[i] = -self.base_far_scale * far_ratio * base_scale

                streak_i = int(min(int(self.capture_counter[i]), streak_cap))
                streak_used.append(streak_i)
                hunter_streak_reward[i] = self.base_streak_scale * float(streak_i)

            min_d = min(hunter_d) if hunter_d else (2.0 * self.world_size)
            if min_d <= capture_dis_safe:
                near_ratio_t = 1.0 - min_d / capture_dis_safe
                target_base_reward[self.target_index] = -self.base_near_scale * near_ratio_t
            else:
                far_ratio_t = (min_d - capture_dis_safe) / dist_scale
                target_base_reward[self.target_index] = self.base_far_scale * far_ratio_t

            avg_streak = float(np.mean(streak_used)) if streak_used else 0.0
            target_streak_reward[self.target_index] = -self.base_streak_scale * avg_streak

        if captured:
            self._assign_capture_reward(capture_reward)

        # Step 2: escape_gap拆分奖励（包围质量reward + 拦截reward）
        (
            gap_hunter_encircle_reward_value,
            gap_target_encircle_reward_value,
            gap_hunter_intercept_reward_value,
            gap_target_intercept_reward_value,
            gap_hunter_ids,
            _,
        ) = self.target.compute_escape_gap_reward(
            hunters=self.hunters,
            active_hunter_mask=self.active_hunter_mask,
        )

        if len(gap_hunter_ids) > 0:
            for hid in gap_hunter_ids:
                if hid < 0 or hid >= self.num_hunters:
                    continue
                if (not bool(self.active_hunter_mask[hid])) or (not bool(self.hunters[hid].alive)):
                    continue
                escape_gap_encircle_hunter_reward[int(hid)] = float(gap_hunter_encircle_reward_value)
                escape_gap_intercept_hunter_reward[int(hid)] = float(gap_hunter_intercept_reward_value)
        if bool(self.target.alive):
            escape_gap_encircle_target_reward[self.target_index] = float(gap_target_encircle_reward_value)
            escape_gap_intercept_target_reward[self.target_index] = float(gap_target_intercept_reward_value)

        escape_gap_encircle_reward = (
            escape_gap_encircle_hunter_reward + escape_gap_encircle_target_reward
        ).astype(np.float32)
        escape_gap_intercept_reward = (
            escape_gap_intercept_hunter_reward + escape_gap_intercept_target_reward
        ).astype(np.float32)
        escape_gap_hunter_reward = (
            escape_gap_encircle_hunter_reward + escape_gap_intercept_hunter_reward
        ).astype(np.float32)
        escape_gap_target_reward = (
            escape_gap_encircle_target_reward + escape_gap_intercept_target_reward
        ).astype(np.float32)
        escape_gap_reward = (escape_gap_hunter_reward + escape_gap_target_reward).astype(np.float32)

        # Step 3: 归一化速度线性惩罚，避免数值爆炸
        speed_penalty_vals = []
        for a in self.agents:
            vmax = max(float(a.max_speed), 1e-6)
            speed_norm = float(np.linalg.norm(a.velocity)) / vmax
            speed_penalty_vals.append(speed_norm)
        speed_penalty_reward = -self.speed_penalty * np.asarray(speed_penalty_vals, dtype=np.float32)

        # Step 4: 聚合总奖励与子项
        total = (
            hunter_base_reward
            + target_base_reward
            + hunter_streak_reward
            + target_streak_reward
            + capture_reward
            + escape_gap_reward
            + collision_rewards
            + speed_penalty_reward
        ).astype(np.float32)
        reward_terms = {
            "total": total,
            "hunter_base_reward": hunter_base_reward.astype(np.float32),
            "target_base_reward": target_base_reward.astype(np.float32),
            "hunter_streak_reward": hunter_streak_reward.astype(np.float32),
            "target_streak_reward": target_streak_reward.astype(np.float32),
            "capture_reward": capture_reward.astype(np.float32),
            "escape_gap_reward": escape_gap_reward.astype(np.float32),
            "escape_gap_encircle_reward": escape_gap_encircle_reward.astype(np.float32),
            "escape_gap_intercept_reward": escape_gap_intercept_reward.astype(np.float32),
            "escape_gap_hunter_reward": escape_gap_hunter_reward.astype(np.float32),
            "escape_gap_target_reward": escape_gap_target_reward.astype(np.float32),
            "escape_gap_encircle_hunter_reward": escape_gap_encircle_hunter_reward.astype(np.float32),
            "escape_gap_encircle_target_reward": escape_gap_encircle_target_reward.astype(np.float32),
            "escape_gap_intercept_hunter_reward": escape_gap_intercept_hunter_reward.astype(np.float32),
            "escape_gap_intercept_target_reward": escape_gap_intercept_target_reward.astype(np.float32),
            "collision_reward": collision_rewards.astype(np.float32),
            "speed_penalty_reward": speed_penalty_reward.astype(np.float32),
        }
        return total, reward_terms

    def _base_rewards(self, captured):
        """
        功能:
            计算基础任务奖励（不含速度惩罚与碰撞惩罚）。
        输入:
            captured (bool): 本步是否捕获成功。
        输出:
            np.ndarray: shape=(agent_num,)。
        """
        r = np.zeros(self.agent_num, dtype=np.float32)
        dist_scale = max(float(self.world_size), 1e-6)
        capture_dis_safe = max(float(self.capture_dis), 1e-6)
        hunter_d = []
        for i, h in enumerate(self.hunters):
            if not bool(self.active_hunter_mask[i]):
                continue
            d = float(np.linalg.norm(h.position - self.target.position))
            hunter_d.append(d)
            if d <= capture_dis_safe:
                r[i] += self.base_near_scale * (1.0 - d / capture_dis_safe)
            else:
                r[i] += -self.base_far_scale * ((d - capture_dis_safe) / dist_scale)
            r[i] += self.hunter_capture_reward if captured else 0.0

        min_d = min(hunter_d) if hunter_d else (2.0 * self.world_size)
        if min_d <= capture_dis_safe:
            r[self.target_index] += -self.base_near_scale * (1.0 - min_d / capture_dis_safe)
        else:
            r[self.target_index] += self.base_far_scale * ((min_d - capture_dis_safe) / dist_scale)
        r[self.target_index] += -(self.target_captured_penalty if captured else 0.0)
        return r

    def _build_obs(self, team_sees_target):
        """
        功能:
            组装每个agent的观测向量（own + neighbor + target + memory + coord_summary）。
        输入:
            team_sees_target (bool): 追捕组是否共享可见target信息。
        输出:
            List[np.ndarray]: shape=(agent_num, obs_dim)。
        """
        scale = max(self.world_size, 1e-6)
        obs = []
        for i, ai in enumerate(self.agents):
            if i < self.num_hunters and not bool(self.active_hunter_mask[i]):
                obs.append(np.zeros(self.obs_dim, dtype=np.float32))
                continue
            own = np.concatenate([ai.position / scale, ai.velocity / scale]).astype(np.float32)
            neighbor_obs = self._neighbor_obs(i, scale)
            target_obs = self._target_obs(i, team_sees_target, scale)
            memory_obs = self._memory_obs(i, team_sees_target, scale)
            coord_summary_obs = self._coord_summary_obs(i)
            if bool(self.coord_summary_obs_enable):
                self.last_coord_summary_cache[int(i)] = coord_summary_obs.astype(np.float32)
            parts = [own, neighbor_obs, target_obs, memory_obs, coord_summary_obs]
            obs.append(np.concatenate(parts, axis=0).astype(np.float32))
        return obs

    def _neighbor_obs(self, obs_idx, scale):
        """
        功能:
            构造观测智能体的最近同阵营邻居观测（固定槽位 + valid mask）。
        输入:
            obs_idx (int): 观测者索引。
            scale (float): 归一化尺度（world_size）。
        输出:
            np.ndarray: shape=(neighbor_N * 6,)。
        """
        if self.neighbor_N <= 0:
            return np.zeros(0, dtype=np.float32)

        ai = self.agents[obs_idx]
        candidates = []
        for j, aj in enumerate(self.agents):
            if j == obs_idx:
                continue
            # 同阵营定义：hunter之间互为邻居；target无同阵营邻居（当前单target）
            if (
                ai.role == "hunter"
                and aj.role == "hunter"
                and aj.alive
                and bool(self.active_hunter_mask[j])
            ):
                dist = float(np.linalg.norm(aj.position - ai.position))
                candidates.append((dist, j))

        candidates.sort(key=lambda x: x[0])
        selected = candidates[: self.neighbor_N]

        slots = []
        for _, j in selected:
            aj = self.agents[j]
            rel_pos = (aj.position - ai.position) / scale
            rel_vel = (aj.velocity - ai.velocity) / scale
            dist = np.linalg.norm(aj.position - ai.position) / scale
            slots.append(
                np.array(
                    [rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], dist, 1.0],
                    dtype=np.float32,
                )
            )

        while len(slots) < self.neighbor_N:
            slots.append(np.zeros(self.neighbor_feat_dim, dtype=np.float32))

        return np.concatenate(slots, axis=0).astype(np.float32)

    def _target_obs(self, obs_idx, team_sees_target, scale):
        """
        功能:
            构造target字段观测（hunter看target；target看最近hunter）。
        输入:
            obs_idx (int): 观测者索引。
            team_sees_target (bool): 追捕组是否共享可见target信息。
            scale (float): 归一化尺度（world_size）。
        输出:
            np.ndarray: shape=(6,), [dx,dy,dvx,dvy,d,visible]。
        """
        ai = self.agents[obs_idx]

        # Hunter侧：观测target，是否可见由team共享可见性控制。
        if ai.role == "hunter":
            if (not self.target.alive) or (not team_sees_target):
                return np.zeros(self.target_feat_dim, dtype=np.float32)
            rel_pos = (self.target.position - ai.position) / scale
            rel_vel = (self.target.velocity - ai.velocity) / scale
            dist = np.linalg.norm(self.target.position - ai.position) / scale
            return np.array(
                [rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], dist, 1.0],
                dtype=np.float32,
            )

        # Target侧：观测最近且存活的hunter，visible表示是否存在有效hunter。
        alive_hunters = [
            h for idx, h in enumerate(self.hunters) if h.alive and bool(self.active_hunter_mask[idx])
        ]
        if len(alive_hunters) == 0:
            return np.zeros(self.target_feat_dim, dtype=np.float32)
        nearest_hunter = min(
            alive_hunters,
            key=lambda h: float(np.linalg.norm(h.position - ai.position)),
        )
        rel_pos = (nearest_hunter.position - ai.position) / scale
        rel_vel = (nearest_hunter.velocity - ai.velocity) / scale
        dist = np.linalg.norm(nearest_hunter.position - ai.position) / scale
        return np.array(
            [rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], dist, 1.0],
            dtype=np.float32,
        )

    def _pair_obs(self, i, j, team_sees_target, scale):
        """
        功能:
            构造agent i对agent j的相对观测特征。
        输入:
            i (int): 观察者索引。
            j (int): 被观察者索引。
            team_sees_target (bool): 追捕组是否可见target。
            scale (float): 归一化尺度（world_size）。
        输出:
            np.ndarray: shape=(5,), [dx,dy,dvx,dvy,d]。
        """
        ai = self.agents[i]
        aj = self.agents[j]
        if not aj.alive:
            return np.zeros(5, dtype=np.float32)

        visible = False
        # 同属于Hunter
        if i < self.num_hunters and j < self.num_hunters:
            visible = True

        # Hunter观测Target
        elif i < self.num_hunters and j == self.target_index:
            visible = team_sees_target
        
        # Target观测Hunter
        elif i == self.target_index and j < self.num_hunters:
            visible = (
                True
                if self.target_perception_radius < 0
                else (np.linalg.norm(ai.position - aj.position) <= self.target_perception_radius)
            )
        if not visible:
            return np.zeros(5, dtype=np.float32)

        rel_pos = (aj.position - ai.position) / scale
        rel_vel = (aj.velocity - ai.velocity) / scale
        dist = np.linalg.norm(aj.position - ai.position) / scale
        return np.array([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], dist], dtype=np.float32)

    def _memory_obs(self, obs_idx, team_sees_target, scale):
        """
        功能:
            构造共享target记忆槽特征。
        输入:
            obs_idx (int): 当前观测agent索引。
            team_sees_target (bool): 当前步追捕组是否真实观测到target。
            scale (float): 归一化尺度（world_size）。
        输出:
            np.ndarray: shape=(5,)。
        """
        # 对target自身，该槽位始终无效。
        if obs_idx == self.target_index:
            return np.zeros(5, dtype=np.float32)
        if obs_idx < self.num_hunters and not bool(self.active_hunter_mask[obs_idx]):
            return np.zeros(5, dtype=np.float32)

        # 去冗余策略：
        # 1) 真实可见时，target信息由pair_obs提供，memory置零；
        # 2) 不可见时，memory提供共享估计（含age）。
        if team_sees_target or (not self.shared_target_valid):
            return np.zeros(5, dtype=np.float32)

        agent = self.agents[obs_idx]
        rel_pos = (self.shared_target_pos - agent.position) / scale
        rel_vel = (self.shared_target_vel - agent.velocity) / scale
        age_norm = float(self.last_seen_age) / float(max(1, self.max_steps))
        return np.array([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], age_norm], dtype=np.float32)

    def _coord_summary_obs(self, obs_idx):
        """
        功能:
            构造协同摘要观测（self_is_topk_by_target_distance, hunters_in_escape_radius_count）。
        输入:
            obs_idx (int): 当前观测agent索引。
        输出:
            np.ndarray: shape=(2,)；未启用或无效时返回全零。
        """
        if not bool(self.coord_summary_obs_enable):
            return np.zeros(0, dtype=np.float32)
        if obs_idx == self.target_index:
            return np.zeros(2, dtype=np.float32)
        if obs_idx < self.num_hunters and not bool(self.active_hunter_mask[obs_idx]):
            return np.zeros(2, dtype=np.float32)
        if not bool(self.target.alive):
            return np.zeros(2, dtype=np.float32)

        # Step 1: 计算self是否位于“距离Target最近的Top-K Hunter”。
        alive_active_ids = [
            hid
            for hid in range(self.num_hunters)
            if bool(self.active_hunter_mask[hid]) and bool(self.hunters[hid].alive)
        ]
        if len(alive_active_ids) == 0:
            return np.zeros(2, dtype=np.float32)
        dist_pairs = []
        for hid in alive_active_ids:
            dist_val = float(np.linalg.norm(self.hunters[hid].position - self.target.position))
            dist_pairs.append((dist_val, int(hid)))
        dist_pairs.sort(key=lambda x: x[0])
        topk = int(min(max(1, self.coord_topk_hunters), len(dist_pairs)))
        topk_ids = {hid for _, hid in dist_pairs[:topk]}
        self_is_topk = 1.0 if int(obs_idx) in topk_ids else 0.0

        # Step 2: 统计与Target距离小于escape_radius的Hunter数量（潜在参与包围数量）。
        escape_radius = float(max(0.0, self.target.escape_dis))
        if escape_radius <= 0.0:
            hunters_in_escape_radius_count = 0.0
        else:
            in_radius_count = 0
            for _, hid in dist_pairs:
                dist_val = float(np.linalg.norm(self.hunters[hid].position - self.target.position))
                if dist_val < escape_radius:
                    in_radius_count += 1
            hunters_in_escape_radius_count = float(in_radius_count)

        return np.array([float(self_is_topk), float(hunters_in_escape_radius_count)], dtype=np.float32)

    def _load_patrol_routes(self, route_path, route_names):
        """
        功能:
            从JSON加载巡逻路线并转换到环境全局坐标系，同时过滤边界危险区航点。
        输入:
            route_path (str): 路线文件路径（绝对路径或项目相对路径）。
            route_names (list[str]): 指定路线名；空列表或包含\"all\"时加载全部。
        输出:
            list[list[np.ndarray]]: 每条路线由全局坐标航点组成。
        """
        self._last_loaded_patrol_route_names = []
        if not route_path:
            return []
        path = route_path
        if not os.path.isabs(path):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(root, path)
        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        route_names_list = [str(x).strip() for x in (route_names or []) if str(x).strip()]
        use_all_routes = (len(route_names_list) == 0) or any(n.lower() == "all" for n in route_names_list)
        selected = None if use_all_routes else set(route_names_list)
        coord_mode = data.get("meta", {}).get("coords", "")
        routes = []
        for item in data.get("routes", []):
            name = str(item.get("name", ""))
            if selected is not None and name not in selected:
                continue
            points = np.asarray(item.get("waypoints", []), dtype=np.float32)
            if points.size == 0:
                continue
            if coord_mode == "normalized_0_1":
                points = (points * 2.0 - 1.0) * self.world_size
            points = np.clip(points, -self.world_size, self.world_size).astype(np.float32)
            # Step 1: 过滤边界危险区（距离最近边界 <= target_safe_dis）的航点
            filtered_points = []
            for p in points:
                boundary_dist = float(max(0.0, self.world_size - max(abs(float(p[0])), abs(float(p[1])))))
                if boundary_dist <= float(self.target_safe_dis):
                    continue
                filtered_points.append(np.asarray(p, dtype=np.float32))

            # Step 2: 若过滤后航点为空，跳过该路线
            if len(filtered_points) == 0:
                continue
            routes.append(filtered_points)
            self._last_loaded_patrol_route_names.append(name)
        return routes
