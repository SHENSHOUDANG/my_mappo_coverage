from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml


class MultiUavPursuitEnv:
    def __init__(
        self,
        num_hunters=1,
        num_blockers=0,
        world_size=1.0,
        dt=0.1,
        capture_radius=0.12,
        capture_steps=5,
        collision_radius=0.02,
        collision_penalty_k=5.0,
        noisy_target_info_when_unseen=False,
        noisy_target_pos_std=0.02,
        noisy_target_vel_std=0.02,
        lost_target_penalty=0.0,
        lost_target_penalty_age_scale=0.0,
        max_steps=300,
        seed=None,
        target_policy_source="train",
        target_patrol_path=None,
        max_speed_hunter=1.2,
        max_speed_blocker=0.9,
        max_speed_target=1.0,
        perception_hunter=0.8,
        perception_blocker=1.2,
        perception_target=0.8,
        speed_penalty=0.00,
        target_patrol_names=None,
    ):
        if num_hunters < 1:
            raise ValueError("num_hunters must be >= 1")
        if num_blockers < 0:
            raise ValueError("num_blockers must be >= 0")

        self.num_hunters = num_hunters
        self.num_blockers = num_blockers
        self.num_targets = 1
        self.agent_num = self.num_hunters + self.num_blockers + self.num_targets

        self.world_size = world_size
        self.dt = dt
        self.capture_radius = capture_radius
        self.capture_steps = capture_steps
        self.collision_radius = float(collision_radius)
        self.collision_penalty_k = float(collision_penalty_k)
        self.noisy_target_info_when_unseen = bool(noisy_target_info_when_unseen)
        self.noisy_target_pos_std = float(noisy_target_pos_std)
        self.noisy_target_vel_std = float(noisy_target_vel_std)
        self.lost_target_penalty = float(lost_target_penalty)
        self.lost_target_penalty_age_scale = float(lost_target_penalty_age_scale)
        self.max_steps = max_steps
        self.np_random = np.random.RandomState(seed)

        self.target_policy_source = target_policy_source
        self.target_patrol_names = self._normalize_name_list(target_patrol_names)
        self.patrol_routes, self.target_patrol_name, self.target_patrol_waypoints = self._load_patrol_routes(
            target_patrol_path,
            self.target_patrol_names,
        )
        self._target_patrol_idx = 0

        self.role_names = ["hunter"] * self.num_hunters + ["blocker"] * self.num_blockers + ["target"]
        self.role_groups = {
            "hunter": list(range(self.num_hunters)),
            "blocker": list(range(self.num_hunters, self.num_hunters + self.num_blockers)),
            "target": [self.agent_num - 1],
        }

        self.max_speeds = {
            "hunter": float(max_speed_hunter),
            "blocker": float(max_speed_blocker),
            "target": float(max_speed_target),
        }
        self.perception_ranges = {
            "hunter": float(perception_hunter),
            "blocker": float(perception_blocker),
            "target": float(perception_target),
        }
        self.speed_penalty = float(speed_penalty)

        self.positions = np.zeros((self.agent_num, 2), dtype=np.float32)
        self.velocities = np.zeros((self.agent_num, 2), dtype=np.float32)
        # 评估场景可指定固定初始位置，用于复现实验。
        self.fixed_initial_positions = None
        self.last_seen_target_pos = None
        self.last_seen_target_vel = None
        self.last_seen_target_age = 0
        self.target_visible = False
        self.pursuit_obs_target_pos = None
        self.pursuit_obs_target_vel = None
        self.agent_done = np.zeros(self.agent_num, dtype=bool)
        self._collision_pairs_seen = set()
        self._capture_counter = 0
        self._step_count = 0

        self.action_dim = 2
        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32) for _ in range(self.agent_num)]

        self.obs_dim = self._calc_obs_dim()
        self.observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32) for _ in range(self.agent_num)]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim * self.agent_num,), dtype=np.float32)
            for _ in range(self.agent_num)
        ]

    @staticmethod
    def _normalize_name_list(names):
        if names is None:
            return None
        if isinstance(names, (list, tuple)):
            cleaned = [str(n).strip() for n in names if str(n).strip()]
            return cleaned or None
        text = str(names).strip()
        if not text:
            return None
        return [n.strip() for n in text.replace(";", ",").split(",") if n.strip()]

    def _convert_waypoints(self, waypoints):
        arr = np.asarray(waypoints, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
            raise ValueError("巡逻路径至少需要2个二维路点")
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if 0.0 <= min_val and max_val <= 1.0:
            arr = (arr * 2.0 - 1.0) * self.world_size
        elif -1.0 <= min_val and max_val <= 1.0:
            arr = arr * self.world_size
        return arr

    def _load_patrol_routes(self, target_patrol_path, preferred_names):
        if self.target_policy_source != "patrol":
            return {}, None, None
        if target_patrol_path is None:
            return {}, None, None
        route_file = Path(target_patrol_path)
        data = yaml.safe_load(route_file.read_text(encoding="utf-8")) or {}
        routes = {}

        if isinstance(data, dict) and "routes" in data:
            routes_data = data["routes"]
            if isinstance(routes_data, dict):
                for name, payload in routes_data.items():
                    waypoints = payload.get("waypoints", payload)
                    routes[str(name)] = self._convert_waypoints(waypoints)
            elif isinstance(routes_data, list):
                for item in routes_data:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or item.get("alias")
                    waypoints = item.get("waypoints")
                    if name is None or waypoints is None:
                        continue
                    routes[str(name)] = self._convert_waypoints(waypoints)
        else:
            waypoints = data.get("waypoints", data) if isinstance(data, dict) else data
            routes["default"] = self._convert_waypoints(waypoints)

        if not routes:
            raise ValueError("未找到可用的巡逻路径")

        chosen_name = None
        if preferred_names:
            for name in preferred_names:
                if name in routes:
                    chosen_name = name
                    break
            if chosen_name is None:
                raise ValueError(f"指定的巡逻路线名称不存在: {preferred_names}")
        else:
            chosen_name = next(iter(routes.keys()))

        return routes, chosen_name, routes[chosen_name]

    def set_target_patrol_route(self, route_name):
        if self.target_policy_source != "patrol":
            return
        if route_name not in self.patrol_routes:
            raise ValueError(f"巡逻路线不存在: {route_name}")
        self.target_patrol_name = route_name
        self.target_patrol_waypoints = self.patrol_routes[route_name]
        self._target_patrol_idx = 0

    def set_target_patrol_waypoints(self, waypoints, route_name=None):
        if self.target_policy_source != "patrol":
            return
        converted = self._convert_waypoints(waypoints)
        name = route_name or self.target_patrol_name or "custom"
        self.patrol_routes[name] = converted
        self.target_patrol_name = name
        self.target_patrol_waypoints = converted
        self._target_patrol_idx = 0

    def get_patrol_route_names(self):
        return list(self.patrol_routes.keys())

    def _calc_obs_dim(self):
        # own pos (2) + own vel (2) + per-other: rel pos (2) + rel vel (2) + dist (1)
        # + shared target memory (pos2 + vel2 + age1)
        return 4 + (self.agent_num - 1) * 5 + 5


    def set_initial_positions(self, initial_positions=None):
        # 设置固定初始位置；若为空则回退到随机初始化。
        if initial_positions is None:
            self.fixed_initial_positions = None
            return
        arr = np.asarray(initial_positions, dtype=np.float32)
        if arr.shape != (self.agent_num, 2):
            raise ValueError(f"initial_positions shape must be ({self.agent_num}, 2), got {arr.shape}")
        self.fixed_initial_positions = np.clip(arr, -self.world_size, self.world_size)

    def apply_scenario_config(self, scenario):
        # 评估时按场景覆盖环境参数。
        if scenario is None:
            return
        self.world_size = float(scenario.get("world_size", self.world_size))
        self.dt = float(scenario.get("dt", self.dt))
        self.capture_radius = float(scenario.get("capture_radius", self.capture_radius))
        self.capture_steps = int(scenario.get("capture_steps", self.capture_steps))
        if "collision_radius" in scenario and scenario["collision_radius"] is not None:
            self.collision_radius = float(scenario["collision_radius"])
        if "collision_penalty_k" in scenario and scenario["collision_penalty_k"] is not None:
            self.collision_penalty_k = float(scenario["collision_penalty_k"])
        if "noisy_target_info_when_unseen" in scenario and scenario["noisy_target_info_when_unseen"] is not None:
            self.noisy_target_info_when_unseen = bool(scenario["noisy_target_info_when_unseen"])
        if "noisy_target_pos_std" in scenario and scenario["noisy_target_pos_std"] is not None:
            self.noisy_target_pos_std = float(scenario["noisy_target_pos_std"])
        if "noisy_target_vel_std" in scenario and scenario["noisy_target_vel_std"] is not None:
            self.noisy_target_vel_std = float(scenario["noisy_target_vel_std"])
        if "lost_target_penalty" in scenario and scenario["lost_target_penalty"] is not None:
            self.lost_target_penalty = float(scenario["lost_target_penalty"])
        if "lost_target_penalty_age_scale" in scenario and scenario["lost_target_penalty_age_scale"] is not None:
            self.lost_target_penalty_age_scale = float(scenario["lost_target_penalty_age_scale"])
        self.max_steps = int(scenario.get("episode_length", self.max_steps))
        # 允许场景覆盖速度与感知参数，便于多样化验证。
        for role in ("hunter", "blocker", "target"):
            speed_key = f"max_speed_{role}"
            if speed_key in scenario and scenario[speed_key] is not None:
                self.max_speeds[role] = float(scenario[speed_key])
            perception_key = f"perception_{role}"
            if perception_key in scenario and scenario[perception_key] is not None:
                self.perception_ranges[role] = float(scenario[perception_key])
        if "seed" in scenario and scenario["seed"] is not None:
            self.seed(int(scenario["seed"]))
        if "target_policy_source" in scenario and scenario["target_policy_source"] is not None:
            self.target_policy_source = str(scenario["target_policy_source"])
        self.set_initial_positions(scenario.get("initial_positions"))

    def reset(self):
        if self.fixed_initial_positions is None:
            self.positions = self.np_random.uniform(low=-self.world_size, high=self.world_size, size=(self.agent_num, 2)).astype(np.float32)
        else:
            self.positions = self.fixed_initial_positions.copy()
        self.velocities = np.zeros((self.agent_num, 2), dtype=np.float32)
        self.last_seen_target_pos = None
        self.last_seen_target_vel = None
        self.last_seen_target_age = 0
        self.target_visible = False
        self.pursuit_obs_target_pos = None
        self.pursuit_obs_target_vel = None
        self.agent_done = np.zeros(self.agent_num, dtype=bool)
        self._collision_pairs_seen = set()
        self._capture_counter = 0
        self._step_count = 0
        self._target_patrol_idx = 0
        self._update_target_memory()
        return self._get_obs()

    def _patrol_action(self, target_pos):
        waypoint = self.target_patrol_waypoints[self._target_patrol_idx]
        diff = waypoint - target_pos
        dist = np.linalg.norm(diff)
        if dist < self.max_speeds["target"] * self.dt:
            self._target_patrol_idx = (self._target_patrol_idx + 1) % len(self.target_patrol_waypoints)
            waypoint = self.target_patrol_waypoints[self._target_patrol_idx]
            diff = waypoint - target_pos
            dist = np.linalg.norm(diff)
        if dist < 1e-8:
            return np.zeros(2, dtype=np.float32)
        return (diff / dist).astype(np.float32)

    def step(self, actions):
        self._step_count += 1
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.agent_num, self.action_dim):
            raise ValueError(f"Expected actions shape ({self.agent_num}, {self.action_dim}), got {actions.shape}")

        target_idx = self.agent_num - 1
        if self.target_policy_source == "patrol":
            actions[target_idx] = self._patrol_action(self.positions[target_idx])

        prev_done = self.agent_done.copy()
        for idx in range(self.agent_num):
            if self.agent_done[idx]:
                self.velocities[idx] = 0.0
                continue
            role = self.role_names[idx]
            max_speed = self.max_speeds[role]
            action = np.clip(actions[idx], -1.0, 1.0)
            velocity = action * max_speed
            self.velocities[idx] = velocity
            self.positions[idx] += velocity * self.dt
            self.positions[idx] = np.clip(self.positions[idx], -self.world_size, self.world_size)

        self._update_target_memory()
        collision_agents, collision_penalties, target_collision = self._detect_collisions(target_idx)
        if collision_agents:
            for idx in collision_agents:
                self.agent_done[idx] = True
        if target_collision:
            self.agent_done[:] = True
            self._capture_counter = 0

        obs = self._get_obs()
        rewards, capture = self._get_rewards(target_collision=target_collision)
        rewards += collision_penalties
        rewards[prev_done] = 0.0
        dones = self.agent_done.copy()
        if target_collision or self._step_count >= self.max_steps:
            dones[:] = True
        elif capture:
            dones[:] = True
        infos = self._get_infos(capture, collision_agents, target_collision)
        return obs, rewards, dones, infos

    def _get_obs(self):
        obs = []
        target_idx = self.agent_num - 1
        target_pos = self.positions[target_idx]
        target_vel = self.velocities[target_idx]

        pursuer_ids = self.role_groups["hunter"] + self.role_groups["blocker"]
        target_spotted = False

        # Target已经被观察到
        for idx in pursuer_ids:
            perception = self.perception_ranges[self.role_names[idx]]
            if np.linalg.norm(target_pos - self.positions[idx]) <= perception:
                target_spotted = True
                break
            
        for idx in range(self.agent_num):
            role = self.role_names[idx]
            perception = self.perception_ranges[role]
            own_pos = self.positions[idx]
            own_vel = self.velocities[idx]
            other_features = []
            for jdx in range(self.agent_num):
                if jdx == idx:
                    continue
                rel_pos = self.positions[jdx] - own_pos
                rel_vel = self.velocities[jdx] - own_vel
                dist = np.linalg.norm(rel_pos)
                
                if role in ("hunter", "blocker"):
                    # Pursuit组共享位置速度信息
                    if self.role_names[jdx] in ("hunter", "blocker"):
                        in_range = True
                    else:
                        in_range = target_spotted
                else:
                    in_range = dist <= perception

                if in_range:
                    other_features.extend([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], dist])
                else:
                    other_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
            if role in ("hunter", "blocker"):
                if self.pursuit_obs_target_pos is None:
                    memory = np.zeros(5, dtype=np.float32)
                else:
                    rel_pos = self.pursuit_obs_target_pos - own_pos
                    rel_vel = self.pursuit_obs_target_vel - own_vel
                    memory = np.array([
                        rel_pos[0],
                        rel_pos[1],
                        rel_vel[0],
                        rel_vel[1],
                        float(self.last_seen_target_age),
                    ], dtype=np.float32)
            else:
                memory = np.zeros(5, dtype=np.float32)
            obs_vec = np.concatenate([own_pos, own_vel, np.array(other_features, dtype=np.float32), memory])
            obs.append(obs_vec.astype(np.float32))
        return np.stack(obs, axis=0)

    def _get_rewards(self, target_collision=False):
        target_idx = self.agent_num - 1
        target_pos = self.positions[target_idx]
        hunter_indices = self.role_groups["hunter"]
        distances_to_target = [np.linalg.norm(self.positions[idx] - target_pos) for idx in hunter_indices]
        min_distance = float(np.min(distances_to_target))

        if not target_collision and min_distance <= self.capture_radius:
            self._capture_counter += 1
        else:
            self._capture_counter = 0
        capture = self._capture_counter >= self.capture_steps
        if target_collision:
            capture = False

        rewards = np.zeros(self.agent_num, dtype=np.float32)
        for idx in range(self.agent_num - 1):
            role = self.role_names[idx]
            dist = np.linalg.norm(self.positions[idx] - target_pos)
            if role == "hunter":
                rewards[idx] = -dist + (10.0 if capture else 0.0)
            else:
                rewards[idx] = -0.7 * dist + (6.0 if capture else 0.0)

        rewards[target_idx] = min_distance - (12.0 if capture else 0.0)
        if self.speed_penalty > 0.0:
            speeds = np.linalg.norm(self.velocities, axis=1)
            rewards -= self.speed_penalty * (speeds ** 2)
        if self.num_blockers > 0 and not self.target_visible:
            age_penalty = self.lost_target_penalty_age_scale * float(self.last_seen_target_age)
            blocker_penalty = self.lost_target_penalty + age_penalty
            if blocker_penalty != 0.0:
                for idx in self.role_groups["blocker"]:
                    rewards[idx] -= blocker_penalty
        return rewards, capture

    def _get_infos(self, capture, collision_agents=None, target_collision=False):
        collision_agents = collision_agents or set()
        infos = []
        for idx in range(self.agent_num):
            infos.append({
                "role": self.role_names[idx],
                "capture": capture,
                "collision": idx in collision_agents,
                "target_collision": target_collision,
                "role_groups": self.role_groups,
                "target_policy_source": self.target_policy_source,
                "target_patrol_name": self.target_patrol_name,
            })
        return infos

    def _update_target_memory(self):
        target_idx = self.agent_num - 1
        target_pos = self.positions[target_idx]
        target_vel = self.velocities[target_idx]
        pursuer_ids = self.role_groups["hunter"] + self.role_groups["blocker"]
        visible = False
        for idx in pursuer_ids:
            perception = self.perception_ranges[self.role_names[idx]]
            if np.linalg.norm(target_pos - self.positions[idx]) <= perception:
                visible = True
                break

        if visible:
            self.last_seen_target_pos = target_pos.copy()
            self.last_seen_target_vel = target_vel.copy()
            self.last_seen_target_age = 0
            self.target_visible = True
            self.pursuit_obs_target_pos = target_pos.copy()
            self.pursuit_obs_target_vel = target_vel.copy()
            return
        
        # Unseen: still provide target info with noise; noise cleared once visible.
        noisy_pos = target_pos + self.np_random.normal(0.0, self.noisy_target_pos_std, size=2)
        noisy_vel = target_vel + self.np_random.normal(0.0, self.noisy_target_vel_std, size=2)
        self.pursuit_obs_target_pos = np.clip(noisy_pos, -self.world_size, self.world_size).astype(np.float32)
        self.pursuit_obs_target_vel = noisy_vel.astype(np.float32)
        if self.last_seen_target_pos is None:
            self.last_seen_target_pos = target_pos.copy()
            self.last_seen_target_vel = target_vel.copy()
            self.last_seen_target_age = 0
        else:
            self.last_seen_target_age += 1
        self.target_visible = False

    def _detect_collisions(self, target_idx):
        if self.collision_radius <= 0.0:
            return set(), np.zeros(self.agent_num, dtype=np.float32), False
        collision_agents = set()
        penalties = np.zeros(self.agent_num, dtype=np.float32)
        target_collision = False
        for i in range(self.agent_num - 1):
            for j in range(i + 1, self.agent_num):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist > self.collision_radius:
                    continue
                collision_agents.add(i)
                collision_agents.add(j)
                if i == target_idx or j == target_idx:
                    target_collision = True
                pair_key = (i, j)
                if pair_key in self._collision_pairs_seen:
                    continue
                self._collision_pairs_seen.add(pair_key)
                p_rel = self.positions[i] - self.positions[j]
                v_rel = self.velocities[i] - self.velocities[j]
                approaching = float(np.dot(v_rel, p_rel)) < 0.0
                speed_i = float(np.linalg.norm(self.velocities[i]))
                speed_j = float(np.linalg.norm(self.velocities[j]))
                if approaching:
                    penalties[i] -= self.collision_penalty_k * speed_i
                    penalties[j] -= self.collision_penalty_k * speed_j
                else:
                    if speed_i > speed_j:
                        penalties[i] -= self.collision_penalty_k * speed_i
                    elif speed_j > speed_i:
                        penalties[j] -= self.collision_penalty_k * speed_j
                    else:
                        penalties[i] -= self.collision_penalty_k * speed_i
                        penalties[j] -= self.collision_penalty_k * speed_j
        return collision_agents, penalties, target_collision

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return seed
