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
            # velocity模式: 动作表示目标速度，并施加单步转角上限约束。
            desired_velocity = u * self.max_speed
            self.velocity = self._apply_turn_limit(self.velocity, desired_velocity)

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
    ):
        """
        功能:
            初始化Target智能体，支持learn/patrol/random三类策略。
        输入:
            agent_id (int): Target编号。
            max_speed (float): 最大速度（米/秒）。
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
        输出:
            无。
        """
        super().__init__(
            agent_id=agent_id,
            role="target",
            max_speed=max_speed,
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
        self.route_episode_count = 0
        self.route_index = 0
        self.patrol_index = 0

    def reset(self, init_pos: np.ndarray):
        """
        功能:
            重置Target状态并将巡逻索引归零。
        输入:
            init_pos (np.ndarray): shape=(2,), 全局坐标（米）。
        输出:
            无。
        """
        super().reset(init_pos)
        self.route_episode_count += 1

        if self.policy_type == "patrol" and len(self.patrol_routes) > 0:
            # 每次reset后episode计数+1，与switch_interval取模为0时切换到下一条路线。
            if self.route_episode_count % self.switch_interval == 0:
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
        return super().select_action(step_count, action_from_policy, rng)

    def _patrol_action(self) -> np.ndarray:
        """
        功能:
            计算patrol模式下，指向当前航点的归一化动作。
        输入:
            无（内部使用self.position与self.patrol_waypoints）。
        输出:
            np.ndarray: shape=(2,), 归一化方向动作。
        """
        if not self.patrol_waypoints:
            return np.zeros(2, dtype=np.float32)
        waypoint = self.patrol_waypoints[self.patrol_index]
        vec = waypoint - self.position
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_waypoints)
            waypoint = self.patrol_waypoints[self.patrol_index]
            vec = waypoint - self.position
            dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (vec / dist).astype(np.float32)


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
        self.dt = float(env_cfg.dt)
        self.max_steps = int(env_cfg.episode_length)
        self.num_hunters = int(env_cfg.num_hunters)
        self.num_explorers = int(env_cfg.num_explorers)
        if self.num_explorers != 0:
            raise ValueError("当前阶段仅实现 hunter-only: num_explorers 必须为 0")

        self.target_index = self.num_hunters
        self.agent_num = self.num_hunters + 1
        self.obs_dim = 4 + (self.agent_num - 1) * 5 + 5
        self.action_dim = 2

        self.capture_dis = float(env_cfg.capture_dis)
        self.capture_step = int(env_cfg.capture_step)
        self.collision_dis = float(env_cfg.collision_dis)
        self.target_pos_init_guidance_step = int(env_cfg.target_pos_init_guidance_step)
        self.target_pos_guidance = bool(env_cfg.target_pos_guidance)
        self.noisy_target_pos_std = float(env_cfg.noisy_target_pos_std)
        self.noisy_target_vel_std = float(env_cfg.noisy_target_vel_std)
        self.target_policy_source = str(env_cfg.target_policy_source).lower()
        self.target_switch_interval = int(
            getattr(env_cfg, "target_switch_interval", getattr(env_cfg, "target_patrol_switch_interval", 1))
        )

        self.hunter_perception_radius = float(hunter_cfg.perception_radius)
        self.target_perception_radius = float(target_cfg.perception_radius)
        self.collision_penalty_k = float(reward_cfg.collision_penalty_k)
        self.speed_penalty = float(reward_cfg.speed_penalty)
        self.hunter_capture_reward = float(getattr(reward_cfg, "hunter_capture_reward", 10.0))
        self.target_captured_penalty = float(getattr(reward_cfg, "target_captured_penalty", 12.0))

        # 步骤3：初始化运行时状态缓存
        self.rng = np.random.RandomState()
        self.step_count = 0
        self.episode_count = 0
        self.capture_counter = np.zeros(self.num_hunters, dtype=np.int32)
        self.done = np.zeros(self.agent_num, dtype=bool)

        self.shared_target_pos = np.zeros(2, dtype=np.float32)
        self.shared_target_vel = np.zeros(2, dtype=np.float32)
        self.shared_target_valid = False
        self.last_seen_age = 0
        self.last_episode_captured = False
        self.last_capture_step = None
        self.last_target_collided = False
        self.last_collision_pairs = []

        # 步骤4：初始化Agent对象
        patrol_routes = self._load_patrol_routes(
            env_cfg.target_patrol_path, list(env_cfg.target_patrol_names)
        )
        self.hunters = [
            HunterAgent(
                i,
                max_speed=float(hunter_cfg.max_velo),
                control_mode=str(getattr(hunter_cfg, "control_mode", "velocity")).lower(),
                max_acc=float(getattr(hunter_cfg, "max_acc", 0.0)),
                max_turn_angle=float(getattr(hunter_cfg, "max_turn_angle", 180.0)),
                min_turn_limit_velo=float(getattr(hunter_cfg, "min_turn_limit_velo", 0.0)),
                policy_type="learn",
            )
            for i in range(self.num_hunters)
        ]
        self.target = TargetAgent(
            agent_id=self.target_index,
            max_speed=float(target_cfg.max_velo),
            control_mode=str(getattr(target_cfg, "control_mode", "velocity")).lower(),
            max_acc=float(getattr(target_cfg, "max_acc", 0.0)),
            max_turn_angle=float(getattr(target_cfg, "max_turn_angle", 180.0)),
            min_turn_limit_velo=float(getattr(target_cfg, "min_turn_limit_velo", 0.0)),
            policy_type=self.target_policy_source,
            action_update_interval=max(1, self.target_switch_interval),
            patrol_waypoints=patrol_routes[0] if patrol_routes else None,
            patrol_routes=patrol_routes,
            switch_interval=max(1, self.target_switch_interval),
        )
        self.patrol_routes = patrol_routes

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
        self.rng.seed(seed)

    def reset(self):
        """
        功能:
            重置环境到episode初始状态并返回初始观测。
        输入:
            无。
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

        # 步骤2：随机初始化所有agent
        for agent in self.agents:
            init_pos = self.rng.uniform(-self.world_size, self.world_size, size=2).astype(np.float32)
            agent.reset(init_pos)

        # 步骤3：返回初始观测
        return self._build_obs(team_sees_target=False)

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

        # 步骤5：终止条件判定
        self.step_count += 1
        timeout = self.step_count >= self.max_steps
        all_hunters_dead = not any(h.alive for h in self.hunters)
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
                "reward_capture": float(reward_terms["capture_reward"][i]),
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
            title (str | None): 可选标题文本；None时使用默认标题。
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

        # Step 3: 标题信息（含捕获/碰撞状态与步数）
        capture_text = "Success" if self.last_episode_captured else "NoCapture"
        capture_step_text = str(int(self.last_capture_step)) if self.last_capture_step is not None else "NA"
        collision_text = "TargetCollided" if self.last_target_collided else "NoCollision"
        if title is None:
            title = (
                f"Capture {capture_text} (step {capture_step_text}) | "
                f"Collision {collision_text} | EnvStep {int(self.step_count)}"
            )
        else:
            title = (
                f"{title} | Capture {capture_text} (step {capture_step_text}) | "
                f"Collision {collision_text}"
            )
        ax.set_title(title)

        # Step 4: 绘制轨迹渐隐、当前位置、速度向量、感知半径和碰撞半径
        hunter_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#17becf", "#8c564b"]
        hunter_idx = 0
        velocity_arrow_len = max(1e-6, float(self.world_size) * 0.08)
        for agent in self.agents:
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

        # Step 5: 绘制Pursuit组接收到的Target位置（共享记忆）
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
            for hunter in self.hunters:
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

        # Step 6: 绘制碰撞/抓捕事件提示
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

        # Step 7: 绘制图例
        legend_handles = []
        for hid in range(self.num_hunters):
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
        legend_handles.extend(
            [
                Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728", markeredgecolor="black", markersize=7, label="Target"),
                Line2D([0], [0], color="#1f77b4", lw=1.0, linestyle=":", label="Perception Radius"),
                Line2D([0], [0], color="black", lw=1.0, linestyle="--", label="Collision Radius"),
                Line2D([0], [0], color="#1f77b4", lw=1.8, marker=">", markersize=6, label="Velocity Vector"),
                Line2D([0], [0], color="#1f77b4", lw=1.0, linestyle="--", label="Turn Limit"),
                Line2D([0], [0], marker="*", color="w", markerfacecolor="#e377c2", markeredgecolor="black", markersize=9, label="Shared Target Pos"),
                Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markeredgecolor="black", markersize=9, label="Capture Event"),
                Line2D([0], [0], marker="X", color="w", markerfacecolor="black", markeredgecolor="black", markersize=8, label="Collision Event"),
            ]
        )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.85)

        # Step 8: 返回RGB数组或直接展示
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
        for h in self.hunters:
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
            if not h.alive:
                self.capture_counter[i] = 0
                continue
            d = float(np.linalg.norm(h.position - self.target.position))
            self.capture_counter[i] = self.capture_counter[i] + 1 if d <= self.capture_dis else 0
        return bool(np.any(self.capture_counter >= self.capture_step))

    def _handle_collision(self):
        """
        功能:
            扫描碰撞对并计算碰撞奖励分量，同时标记失活agent。
        输入:
            无。
        输出:
            tuple:
                - bool: Target是否发生碰撞。
                - np.ndarray: 碰撞奖励分量，shape=(agent_num,)。
        """
        target_collided = False
        collision_pairs = []
        agents = self.agents
        disable = [False] * self.agent_num
        collision_rewards = np.zeros(self.agent_num, dtype=np.float32)
        for i in range(self.agent_num):
            if not agents[i].alive:
                continue
            for j in range(i + 1, self.agent_num):
                if not agents[j].alive:
                    continue
                rel_pos = agents[i].position - agents[j].position
                dist = float(np.linalg.norm(rel_pos))
                if dist > self.collision_dis:
                    continue

                collision_pairs.append((int(i), int(j)))
                disable[i] = True
                disable[j] = True
                if i == self.target_index or j == self.target_index:
                    target_collided = True

                rel_vel = agents[i].velocity - agents[j].velocity
                if float(np.dot(rel_vel, rel_pos)) < 0.0:
                    collision_rewards[i] -= self.collision_penalty_k * float(np.linalg.norm(agents[i].velocity))
                    collision_rewards[j] -= self.collision_penalty_k * float(np.linalg.norm(agents[j].velocity))
                else:
                    vi = float(np.linalg.norm(agents[i].velocity))
                    vj = float(np.linalg.norm(agents[j].velocity))
                    if vi >= vj:
                        collision_rewards[i] -= self.collision_penalty_k * vi
                    else:
                        collision_rewards[j] -= self.collision_penalty_k * vj

        for idx, d in enumerate(disable):
            if d:
                agents[idx].alive = False
                agents[idx].velocity[:] = 0.0
        self.last_collision_pairs = collision_pairs
        return target_collided, collision_rewards

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
        # Step 1: 计算基础奖励与捕获奖励
        hunter_base_reward = np.zeros(self.agent_num, dtype=np.float32)
        target_base_reward = np.zeros(self.agent_num, dtype=np.float32)
        capture_reward = np.zeros(self.agent_num, dtype=np.float32)

        hunter_d = []
        for i, h in enumerate(self.hunters):
            d = float(np.linalg.norm(h.position - self.target.position))
            hunter_d.append(d)
            hunter_base_reward[i] = -d
            if captured:
                capture_reward[i] = self.hunter_capture_reward
        min_d = min(hunter_d) if hunter_d else (2.0 * self.world_size)
        target_base_reward[self.target_index] = min_d
        if captured:
            capture_reward[self.target_index] = -self.target_captured_penalty

        # Step 2: 计算速度惩罚分量:  所有Agent都要避免做无畏的高速运动   （考虑Hunter再增加step惩罚？）
        speed_penalty_reward = -self.speed_penalty * np.array(
            [agent.speed for agent in self.agents], dtype=np.float32
        )

        # Step 3: 聚合总奖励与子项
        total = (
            hunter_base_reward
            + target_base_reward
            + capture_reward
            + collision_rewards
            + speed_penalty_reward
        ).astype(np.float32)
        reward_terms = {
            "total": total,
            "hunter_base_reward": hunter_base_reward.astype(np.float32),
            "target_base_reward": target_base_reward.astype(np.float32),
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
        hunter_d = []
        for i, h in enumerate(self.hunters):
            d = float(np.linalg.norm(h.position - self.target.position))
            hunter_d.append(d)
            r[i] += -d + (self.hunter_capture_reward if captured else 0.0)    # 距离越小，惩罚越小  -- > Hunter: 缩小与Target的距离

        min_d = min(hunter_d) if hunter_d else (2.0 * self.world_size)
        r[self.target_index] += min_d - (self.target_captured_penalty if captured else 0.0)  # Target: 增加与其他Hunter的最小距离
        return r

    def _build_obs(self, team_sees_target):
        """
        功能:
            组装每个agent的观测向量（own + pair + memory）。
        输入:
            team_sees_target (bool): 追捕组是否共享可见target信息。
        输出:
            List[np.ndarray]: shape=(agent_num, obs_dim)。
        """
        scale = max(self.world_size, 1e-6)
        obs = []
        for i, ai in enumerate(self.agents):
            own = np.concatenate([ai.position / scale, ai.velocity / scale]).astype(np.float32)
            parts = [own]
            for j, aj in enumerate(self.agents):
                if i == j:
                    continue
                parts.append(self._pair_obs(i, j, team_sees_target, scale))
            parts.append(self._memory_obs(i, team_sees_target, scale))
            obs.append(np.concatenate(parts, axis=0).astype(np.float32))
        return obs

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
            从JSON加载巡逻路线并转换到环境全局坐标系。
        输入:
            route_path (str): 路线文件路径（绝对路径或项目相对路径）。
            route_names (list[str]): 指定路线名；空列表表示加载全部。
        输出:
            list[list[np.ndarray]]: 每条路线由全局坐标航点组成。
        """
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

        selected = set(route_names) if route_names else None
        coord_mode = data.get("meta", {}).get("coords", "")
        routes = []
        for item in data.get("routes", []):
            name = item.get("name")
            if selected is not None and name not in selected:
                continue
            points = np.asarray(item.get("waypoints", []), dtype=np.float32)
            if points.size == 0:
                continue
            if coord_mode == "normalized_0_1":
                points = (points * 2.0 - 1.0) * self.world_size
            points = np.clip(points, -self.world_size, self.world_size).astype(np.float32)
            routes.append([p for p in points])
        return routes
