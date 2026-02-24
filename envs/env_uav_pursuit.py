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
        policy_type: str = "learn",
        policy_net=None,
        action_update_interval: int = 1,
    ):
        self.agent_id = int(agent_id)
        self.role = role
        self.max_speed = float(max_speed)
        self.policy_type = str(policy_type).lower()
        self.policy_net = policy_net
        self.action_update_interval = max(1, int(action_update_interval))

        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.trajectory: List[np.ndarray] = []
        self.alive = True

        self._cached_random_action = np.zeros(2, dtype=np.float32)
        self._last_random_refresh_step = -1

    def reset(self, init_pos: np.ndarray):
        """
        输入:
            init_pos (np.ndarray): shape=(2,), 全局坐标(米)。
        输出:
            无。
        """
        self.position = np.asarray(init_pos, dtype=np.float32).copy()
        self.velocity = np.zeros(2, dtype=np.float32)
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
        self.velocity = u * self.max_speed
        self.position = self.position + self.velocity * float(dt)
        self.position = np.clip(self.position, -float(world_size), float(world_size))
        self.trajectory.append(self.position.copy())


class HunterAgent(BaseAgent):
    def __init__(self, agent_id: int, max_speed: float, policy_type: str = "learn", policy_net=None):
        """
        功能:
            初始化Hunter智能体。
        输入:
            agent_id (int): Hunter编号。
            max_speed (float): 最大速度（米/秒）。
            policy_type (str): 策略类型。
            policy_net (Any): 可选策略网络。
        输出:
            无。
        """
        super().__init__(agent_id, "hunter", max_speed, policy_type, policy_net)


class ExplorerAgent(BaseAgent):
    def __init__(self, agent_id: int, max_speed: float, policy_type: str = "learn", policy_net=None):
        """
        功能:
            初始化Explorer智能体（当前hunter-only任务中默认不启用）。
        输入:
            agent_id (int): Explorer编号。
            max_speed (float): 最大速度（米/秒）。
            policy_type (str): 策略类型。
            policy_net (Any): 可选策略网络。
        输出:
            无。
        """
        super().__init__(agent_id, "explorer", max_speed, policy_type, policy_net)


class TargetAgent(BaseAgent):
    def __init__(
        self,
        agent_id: int,
        max_speed: float,
        policy_type: str = "learn",
        policy_net=None,
        action_update_interval: int = 1,
        patrol_waypoints: Optional[List[np.ndarray]] = None,
    ):
        """
        功能:
            初始化Target智能体，支持learn/patrol/random三类策略。
        输入:
            agent_id (int): Target编号。
            max_speed (float): 最大速度（米/秒）。
            policy_type (str): learn/patrol/random。
            policy_net (Any): learn模式下可选策略网络。
            action_update_interval (int): random策略重采样间隔（步）。
            patrol_waypoints (Optional[List[np.ndarray]]): 巡逻航点（全局坐标，米）。
        输出:
            无。
        """
        super().__init__(
            agent_id=agent_id,
            role="target",
            max_speed=max_speed,
            policy_type=policy_type,
            policy_net=policy_net,
            action_update_interval=action_update_interval,
        )
        self.patrol_waypoints = patrol_waypoints or []
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
        self.target_patrol_switch_interval = int(env_cfg.target_patrol_switch_interval)

        self.hunter_perception_radius = float(hunter_cfg.perception_radius)
        self.target_perception_radius = float(target_cfg.perception_radius)
        self.collision_penalty_k = float(reward_cfg.collision_penalty_k)
        self.speed_penalty = float(reward_cfg.speed_penalty)

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

        # 步骤4：初始化Agent对象
        patrol_routes = self._load_patrol_routes(
            env_cfg.target_patrol_path, list(env_cfg.target_patrol_names)
        )
        self.hunters = [
            HunterAgent(i, max_speed=float(hunter_cfg.max_velo), policy_type="learn")
            for i in range(self.num_hunters)
        ]
        self.target = TargetAgent(
            agent_id=self.target_index,
            max_speed=float(target_cfg.max_velo),
            policy_type=self.target_policy_source,
            action_update_interval=max(1, self.target_patrol_switch_interval),
            patrol_waypoints=patrol_routes[0] if patrol_routes else None,
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
        self.last_seen_age = 0

        # 步骤2：随机初始化所有agent
        for agent in self.agents:
            init_pos = self.rng.uniform(-self.world_size, self.world_size, size=2).astype(np.float32)
            agent.reset(init_pos)

        # 步骤3：若target为patrol模式，重置巡逻路线状态
        if self.target.policy_type == "patrol" and self.patrol_routes:
            idx = (
                (self.episode_count - 1) // max(1, self.target_patrol_switch_interval)
            ) % len(self.patrol_routes)
            self.target.patrol_waypoints = self.patrol_routes[idx]
            self.target.patrol_index = 0
            self.target.position = self.target.patrol_waypoints[0].copy()
            self.target.trajectory = [self.target.position.copy()]

        # 步骤4：返回初始观测
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
        rewards = np.zeros(self.agent_num, dtype=np.float32)
        target_collided = self._handle_collision(rewards)
        team_sees_target = self._team_sees_target()
        self._update_shared_target_memory(team_sees_target)

        captured = False
        if self.target.alive and not target_collided:
            captured = self._update_capture_counter()
            if captured:
                self.target.alive = False
                self.target.velocity[:] = 0.0

        # 步骤4：奖励聚合（基础奖励 + 速度惩罚）
        rewards += self._base_rewards(captured)
        rewards -= self.speed_penalty * np.array(
            [np.sum(agent.velocity ** 2) for agent in self.agents], dtype=np.float32
        )

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
            }
            for a in self.agents
        ]
        return [obs, rews, dones, infos]

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

    def _handle_collision(self, rewards):
        """
        功能:
            扫描碰撞对并施加碰撞惩罚，同时标记失活agent。
        输入:
            rewards (np.ndarray): shape=(agent_num,), 原地叠加惩罚。
        输出:
            bool: Target是否发生碰撞。
        """
        target_collided = False
        agents = self.agents
        disable = [False] * self.agent_num
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

                disable[i] = True
                disable[j] = True
                if i == self.target_index or j == self.target_index:
                    target_collided = True

                rel_vel = agents[i].velocity - agents[j].velocity
                if float(np.dot(rel_vel, rel_pos)) < 0.0:
                    rewards[i] -= self.collision_penalty_k * float(np.linalg.norm(agents[i].velocity))
                    rewards[j] -= self.collision_penalty_k * float(np.linalg.norm(agents[j].velocity))
                else:
                    vi = float(np.linalg.norm(agents[i].velocity))
                    vj = float(np.linalg.norm(agents[j].velocity))
                    if vi >= vj:
                        rewards[i] -= self.collision_penalty_k * vi
                    else:
                        rewards[j] -= self.collision_penalty_k * vj

        for idx, d in enumerate(disable):
            if d:
                agents[idx].alive = False
                agents[idx].velocity[:] = 0.0
        return target_collided

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
            r[i] += -d + (10.0 if captured else 0.0)
        min_d = min(hunter_d) if hunter_d else (2.0 * self.world_size)
        r[self.target_index] += min_d - (12.0 if captured else 0.0)
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
            parts.append(self._memory_obs(i, scale))
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
        if i < self.num_hunters and j < self.num_hunters:
            visible = True
        elif i < self.num_hunters and j == self.target_index:
            visible = team_sees_target
        elif i == self.target_index and j < self.num_hunters:
            visible = np.linalg.norm(ai.position - aj.position) <= self.target_perception_radius
        if not visible:
            return np.zeros(5, dtype=np.float32)

        rel_pos = (aj.position - ai.position) / scale
        rel_vel = (aj.velocity - ai.velocity) / scale
        dist = np.linalg.norm(aj.position - ai.position) / scale
        return np.array([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], dist], dtype=np.float32)

    def _memory_obs(self, obs_idx, scale):
        """
        功能:
            构造共享target记忆槽特征。
        输入:
            obs_idx (int): 当前观测agent索引。
            scale (float): 归一化尺度（world_size）。
        输出:
            np.ndarray: shape=(5,)。
        """
        if obs_idx == self.target_index or not self.shared_target_valid:
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
