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

import numpy as np
import matplotlib

# 仅在无显示环境下回退到Agg，避免影响交互式可视化脚本。
if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

        print(f"{role} Agent {agent_id}: Policy type: {policy_type}")

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
    ) -> np.ndarray:
        """
        输入:
            step_count (int): 当前环境步。
            action_from_policy (Optional[np.ndarray]):
                learn策略对应网络输出，shape=(2,), 归一化动作。
            rng (np.random.RandomState): 随机数发生器。
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
                and self.policy_type in ("random", "patrol")
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
    ):
        """
        功能:
            初始化Target智能体，支持learn/patrol/random三类策略。
        输入:
            agent_id (int): Target编号。
            max_speed (float): 最大速度（米/秒）。
            safe_dis (float): 安全距离阈值（米）。
            control_mode (str): 控制模式（velocity/acceleration）。
            max_acc (float): acceleration模式下最大加速度（米/秒²）。
            max_turn_angle (float): velocity模式下最大转角（度）。
            min_turn_limit_velo (float): 速度超过该阈值时才启用转角限制（米/秒）。
            policy_type (str): learn/patrol/random。
            policy_net (Any): learn模式下可选策略网络。
            action_update_interval (int): random策略重采样间隔（步）。
            patrol_waypoints (Optional[List[np.ndarray]]): 巡逻航点（全局坐标，米）。
            patrol_routes (Optional[List[List[np.ndarray]]]): 可选巡逻路线集合。
            switch_interval (int): 路线切换间隔（按episode计）。
            control_dt (float): 环境控制步长dt（秒），用于patrol速度缩放。
            world_size (float): 地图半边长（米），用于random策略边界修正。
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
        self.route_episode_count = 0
        self.route_index = 0
        self.patrol_index = 0

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
    ) -> np.ndarray:
        """
        功能:
            根据Target策略类型返回动作。
        输入:
            step_count (int): 当前环境步。
            action_from_policy (Optional[np.ndarray]): learn策略动作。
            rng (np.random.RandomState): 随机数生成器。
        输出:
            np.ndarray: shape=(2,), 归一化动作。
        """
        if self.policy_type == "patrol":
            return self._patrol_action()
        if self.policy_type == "random":
            raw_random_action = super().select_action(step_count, action_from_policy, rng)
            return self._random_action_with_boundary_avoidance(raw_random_action)
        return super().select_action(step_count, action_from_policy, rng)

    def _random_action_with_boundary_avoidance(self, random_action: np.ndarray) -> np.ndarray:
        """
        功能:
            random策略下对动作做边界安全修正；若预测下一时刻靠近边界，则叠加朝向中心的最大径向分量。
        输入:
            random_action (np.ndarray): shape=(2,), 原始随机动作（归一化）。
        输出:
            np.ndarray: shape=(2,), 修正后的随机动作（归一化）。
        """
        # Step 1: 预测下一时刻位置
        base_action = np.clip(np.asarray(random_action, dtype=np.float32), -1.0, 1.0)
        if self.control_mode == "acceleration":
            predicted_vel = self.velocity + base_action * float(self.max_acc) * float(self.control_dt)
            speed = float(np.linalg.norm(predicted_vel))
            if speed > float(self.max_speed) and speed > 1e-8:
                predicted_vel = predicted_vel / speed * float(self.max_speed)
        else:
            predicted_vel = base_action * float(self.max_speed)
        p_next = self.position + predicted_vel * float(self.control_dt)

        # Step 2: 下一时刻进入边界风险区时，叠加朝向中心的最大径向动作分量
        boundary_dist = float(max(0.0, self.world_size - max(abs(float(p_next[0])), abs(float(p_next[1])))))
        if boundary_dist >= float(self.safe_dis):
            return base_action

        center_vec = -np.asarray(p_next, dtype=np.float32)
        center_norm = float(np.linalg.norm(center_vec))
        if center_norm <= 1e-8:
            return base_action
        center_dir = center_vec / center_norm
        if self.control_mode == "acceleration":
            radial_action = np.clip(center_dir, -1.0, 1.0).astype(np.float32)
        else:
            radial_action = np.clip(center_dir, -1.0, 1.0).astype(np.float32)
        adjusted_action = base_action + radial_action

        # Step 3: 裁剪后返回有效动作
        return np.clip(adjusted_action, -1.0, 1.0).astype(np.float32)

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
        self.num_hunters = int(env_cfg.num_hunters)
        self.num_explorers = int(env_cfg.num_explorers)
        if self.num_explorers != 0:
            raise ValueError("当前阶段仅实现 hunter-only: num_explorers 必须为 0")

        self.target_index = self.num_hunters
        self.agent_num = self.num_hunters + 1
        self.neighbor_N = int(env_cfg.neighbor_N)
        self.neighbor_N = max(0, self.neighbor_N)
        self.neighbor_feat_dim = 6  # [dx,dy,dvx,dvy,d,valid]
        self.target_feat_dim = 6    # [dx,dy,dvx,dvy,d,visible]
        self.obs_dim = 4 + self.neighbor_N * self.neighbor_feat_dim + self.target_feat_dim + 5
        self.action_dim = 2

        self.capture_dis = float(env_cfg.capture_dis)
        self.capture_step = int(env_cfg.capture_step)
        self.collision_dis = float(env_cfg.collision_dis)
        self.target_pos_init_guidance_step = int(env_cfg.target_pos_init_guidance_step)
        self.target_pos_guidance = bool(env_cfg.target_pos_guidance)
        self.noisy_target_pos_std = float(env_cfg.noisy_target_pos_std)
        self.noisy_target_vel_std = float(env_cfg.noisy_target_vel_std)
        self.target_policy_source = str(env_cfg.target_policy_source).lower()
        self.target_switch_interval = int(env_cfg.target_switch_interval)

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
        self.hunter_capture_reward = float(reward_cfg.hunter_capture_reward)
        self.target_captured_penalty = float(reward_cfg.target_captured_penalty)
        self.target_collision_penalty = float(reward_cfg.target_collision_penalty)

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
        self.active_target_patrol_names = list(env_cfg.target_patrol_names)

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
        )
        self.patrol_routes = patrol_routes
        self.default_target_policy_source = str(self.target_policy_source)
        self.default_target_patrol_names = list(env_cfg.target_patrol_names)
        self.default_target_patrol_path = str(env_cfg.target_patrol_path)

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
        输入:
            无。
        输出:
            List[np.ndarray]: shape=(agent_num, obs_dim)。
        """
        init_positions = self._sample_initial_positions()
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
            应用任务规格（激活Hunter数量、地图尺寸、Target策略与巡逻路线、初始化seed）。
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
        return self.position_rng.uniform(
            -self.world_size, self.world_size, size=(self.agent_num, 2)
        ).astype(np.float32)

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
                "world_size": float(self.world_size),
                "target_policy_source": str(self.target.policy_type),
                "target_patrol_path": str(self.default_target_patrol_path),
                "target_patrol_names": list(self.default_target_patrol_names),
                "target_route_id": int(self.target_route_id),
                "seed": self.base_seed if self.task_seed is None else int(self.task_seed),
            }

        hunter_choices = list(split_cfg.hunter_count_choices)
        policy_choices = [str(x).lower() for x in list(split_cfg.target_policy_choices)]
        patrol_choices = list(split_cfg.patrol_name_choices)
        seed_range = list(split_cfg.seed_range)

        num_hunters = int(self.rng.choice(hunter_choices))
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

        # Step 6: 绘制Pursuit组接收到的Target位置（共享记忆）
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

        # Step 7: 绘制碰撞/抓捕事件提示
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

        # Step 8: 绘制图例
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

        # Step 9: 返回RGB数组或直接展示
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
            - Target边界碰撞后立即结束episode并追加较大惩罚。
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
                disable[i] = True
                boundary_collision_agents.append(int(i))
                if i == self.target_index:
                    target_collided = True
                    collision_rewards[i] -= float(self.target_collision_penalty)

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

        dist_scale = max(float(self.world_size), 1e-6)
        capture_dis_safe = max(float(self.capture_dis), 1e-6)
        streak_cap = max(1, int(self.base_streak_cap))
        hunter_d = []
        streak_used = []

        # 1. Base Reward计算： 主要目的是要求Hunter尽可能接近Target的捕捉半径范围内
        ## Hunter越接近Target(d <= capture_dis)，near_ratio越接近1, reward越大
        ## Hunter越远离Target(d >  capture_dis), far_ratio越接近1（用world_size进行归一化）
        ## Target的reward与Hunter完全相反

        # 2. streak reward: 要求hunter尽可能长期处于Target捕捉半径内
        for i, h in enumerate(self.hunters):
            if not bool(self.active_hunter_mask[i]):
                continue
            d = float(np.linalg.norm(h.position - self.target.position))
            hunter_d.append(d)

            if d <= capture_dis_safe:
                near_ratio = 1.0 - d / capture_dis_safe
                hunter_base_reward[i] = self.base_near_scale * near_ratio
            else:
                far_ratio = (d - capture_dis_safe) / dist_scale
                hunter_base_reward[i] = -self.base_far_scale * far_ratio

            streak_i = int(min(int(self.capture_counter[i]), streak_cap))
            streak_used.append(streak_i)
            hunter_streak_reward[i] = self.base_streak_scale * float(streak_i)

            if captured:
                capture_reward[i] = self.hunter_capture_reward

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
            capture_reward[self.target_index] = -self.target_captured_penalty

        # Step 2: 归一化速度线性惩罚，避免数值爆炸
        speed_penalty_vals = []
        for a in self.agents:
            vmax = max(float(a.max_speed), 1e-6)
            speed_norm = float(np.linalg.norm(a.velocity)) / vmax
            speed_penalty_vals.append(speed_norm)
        speed_penalty_reward = -self.speed_penalty * np.asarray(speed_penalty_vals, dtype=np.float32)

        # Step 3: 聚合总奖励与子项
        total = (
            hunter_base_reward
            + target_base_reward
            + hunter_streak_reward
            + target_streak_reward
            + capture_reward
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
            组装每个agent的观测向量（own + neighbor + target + memory）。
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
            parts = [own, neighbor_obs, target_obs, memory_obs]
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
