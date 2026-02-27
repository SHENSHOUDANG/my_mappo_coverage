"""
Multi-target inference-only environment for UAV pursuit.

设计目标:
1) 支持 N hunters + K explorers + M targets 的统一时空推进。
2) 支持 Explorer 搜索/跟踪状态机 + Hunter 分配。
3) 仅用于推理评估，不包含任何训练逻辑。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import json
import os

import numpy as np
import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from envs.env_uav_pursuit import HunterAgent, ExplorerAgent, TargetAgent


class MultiUAVInferenceEnv(object):
    """
    推理场景环境（多Explorer、多Target）。
    """

    def __init__(self, cfg):
        """
        功能:
            初始化多目标推理环境。
        输入:
            cfg (EasyDict): 合并后的配置对象。
        输出:
            无。
        """
        env_cfg = cfg.multi_infer.env
        hunter_cfg = cfg.Hunter
        target_cfg = cfg.Target
        explorer_cfg = cfg.multi_infer.explorer

        self.cfg = cfg
        self.world_size = float(env_cfg.world_size)
        self.dt = float(env_cfg.dt)
        self.max_steps = int(env_cfg.max_steps)

        self.num_hunters = int(env_cfg.num_hunters)
        self.num_explorers = int(env_cfg.num_explorers)
        self.num_targets = int(env_cfg.num_targets)
        self.assignment_hunters_per_target = int(max(1, int(env_cfg.assignment_hunters_per_target)))

        self.capture_dis = float(env_cfg.capture_dis)
        self.capture_step = int(env_cfg.capture_step)
        self.collision_dis = float(env_cfg.collision_dis)

        self.hunter_perception_radius = float(hunter_cfg.perception_radius)
        self.explorer_perception_radius = float(explorer_cfg.perception_radius)

        self.neighbor_N = int(env_cfg.neighbor_N)
        self.neighbor_feat_dim = 6
        self.target_feat_dim = 6
        self.obs_dim = 4 + self.neighbor_N * self.neighbor_feat_dim + self.target_feat_dim + 5

        self.target_policy_probs = dict(env_cfg.target_policy_probs)
        self.target_switch_interval = int(max(1, int(env_cfg.target_switch_interval)))
        self.route_overlap_rate = float(np.clip(float(explorer_cfg.route_overlap_rate), 0.0, 0.95))

        self.rng = np.random.RandomState()
        self.base_seed = 1

        self.hunters: List[HunterAgent] = []
        self.explorers: List[ExplorerAgent] = []
        self.targets: List[TargetAgent] = []

        self.explorer_state: List[str] = []
        self.explorer_track_target: List[int] = []
        self.explorer_paths: List[List[np.ndarray]] = []
        self.explorer_path_idx: List[int] = []

        self.hunter_assignment: List[int] = []  # hunter_id -> target_id, -1 means idle
        self.target_capture_counter = np.zeros((self.num_hunters, self.num_targets), dtype=np.int32)
        self.target_alive = np.ones(self.num_targets, dtype=bool)
        self.target_discovered = np.zeros(self.num_targets, dtype=bool)

        self.shared_target_info: List[Dict[str, np.ndarray]] = []

        self.step_count = 0
        self.done = False
        self._human_fig = None
        self._human_ax = None

        self._init_agents()

    def _init_agents(self):
        """
        功能:
            按配置构建 Hunter/Explorer/Target 智能体对象。
        输入:
            无。
        输出:
            无。
        """
        hunter_cfg = self.cfg.Hunter
        target_cfg = self.cfg.Target
        explorer_cfg = self.cfg.multi_infer.explorer

        self.hunters = [
            HunterAgent(
                agent_id=i,
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

        self.explorers = [
            ExplorerAgent(
                agent_id=i,
                max_speed=float(explorer_cfg.max_velo),
                safe_dis=float(max(self.collision_dis, float(explorer_cfg.safe_dis))),
                control_mode=str(explorer_cfg.control_mode).lower(),
                max_acc=float(explorer_cfg.max_acc),
                max_turn_angle=float(explorer_cfg.max_turn_angle),
                min_turn_limit_velo=float(explorer_cfg.min_turn_limit_velo),
                policy_type="random",
            )
            for i in range(self.num_explorers)
        ]

        patrol_routes = self._load_patrol_routes(
            route_path=str(self.cfg.env.target_patrol_path),
            route_names=list(self.cfg.env.target_patrol_names),
        )

        self.targets = []
        for i in range(self.num_targets):
            policy_type = self._sample_target_policy_type()
            t = TargetAgent(
                agent_id=i,
                max_speed=float(target_cfg.max_velo),
                safe_dis=float(max(self.collision_dis, float(target_cfg.safe_dis))),
                control_mode=str(target_cfg.control_mode).lower(),
                max_acc=float(target_cfg.max_acc),
                max_turn_angle=float(target_cfg.max_turn_angle),
                min_turn_limit_velo=float(target_cfg.min_turn_limit_velo),
                policy_type=policy_type,
                action_update_interval=max(1, self.target_switch_interval),
                patrol_waypoints=patrol_routes[0] if len(patrol_routes) > 0 else None,
                patrol_routes=patrol_routes,
                switch_interval=max(1, self.target_switch_interval),
                control_dt=float(self.dt),
                world_size=float(self.world_size),
            )
            self.targets.append(t)

    def _sample_target_policy_type(self) -> str:
        """
        功能:
            按配置概率采样 target 策略类型。
        输入:
            无。
        输出:
            str: 策略类型（random/patrol/learn/static）。
        """
        probs = {
            "random": float(self.target_policy_probs.get("random", 0.0)),
            "patrol": float(self.target_policy_probs.get("patrol", 0.0)),
            "learn": float(self.target_policy_probs.get("learn", 0.0)),
            "static": float(self.target_policy_probs.get("static", 0.0)),
        }
        total = float(sum(max(0.0, v) for v in probs.values()))
        if total <= 1e-8:
            return "random"
        keys = list(probs.keys())
        p = np.array([max(0.0, probs[k]) for k in keys], dtype=np.float32)
        p = p / np.sum(p)
        return str(self.rng.choice(keys, p=p))

    def seed(self, seed: int):
        """
        功能:
            设置环境随机种子。
        输入:
            seed (int): 随机种子。
        输出:
            无。
        """
        self.base_seed = int(seed)
        self.rng.seed(int(seed))

    def reset(self):
        """
        功能:
            重置全局环境状态。
        输入:
            无。
        输出:
            dict: 初始状态摘要。
        """
        self.step_count = 0
        self.done = False

        self.target_capture_counter[:] = 0
        self.target_alive[:] = True
        self.target_discovered[:] = False

        self.explorer_state = ["SEARCH" for _ in range(self.num_explorers)]
        self.explorer_track_target = [-1 for _ in range(self.num_explorers)]
        self.explorer_paths = self._build_explorer_paths()
        self.explorer_path_idx = [0 for _ in range(self.num_explorers)]

        self.hunter_assignment = [-1 for _ in range(self.num_hunters)]
        self.shared_target_info = [
            {
                "valid": False,
                "pos": np.zeros(2, dtype=np.float32),
                "vel": np.zeros(2, dtype=np.float32),
                "age": np.float32(0.0),
                "timestamp": np.int32(0),
            }
            for _ in range(self.num_targets)
        ]

        for h in self.hunters:
            h.reset(self._sample_position())
            h.alive = True

        for i, e in enumerate(self.explorers):
            init_pos = self.explorer_paths[i][0].copy() if len(self.explorer_paths[i]) > 0 else self._sample_position()
            e.reset(init_pos)
            e.alive = True

        for i, t in enumerate(self.targets):
            t.policy_type = self._sample_target_policy_type()
            t.reset(self._sample_position())
            t.alive = True
            if t.policy_type == "static":
                t.velocity[:] = 0.0

        return self.get_summary()

    def _sample_position(self) -> np.ndarray:
        """
        功能:
            从地图中均匀采样坐标。
        输入:
            无。
        输出:
            np.ndarray: shape=(2,) 的位置向量。
        """
        return self.rng.uniform(-self.world_size, self.world_size, size=(2,)).astype(np.float32)

    def _build_explorer_paths(self) -> List[List[np.ndarray]]:
        """
        功能:
            构建 K 条分片弓字形搜索航线（每条在自身x区间内往返）。
        输入:
            无。
        输出:
            list[list[np.ndarray]]: 每个Explorer对应航点序列。
        """
        paths: List[List[np.ndarray]] = []
        full = 2.0 * float(self.world_size)
        x_edges = np.linspace(-self.world_size, self.world_size, self.num_explorers + 1)

        spacing = 2.0 * float(self.explorer_perception_radius) * (1.0 - float(self.route_overlap_rate))
        spacing = float(max(1.0, spacing))
        stripes = int(max(2, np.ceil(full / spacing)))
        y_vals = np.linspace(-self.world_size, self.world_size, stripes)

        for k in range(self.num_explorers):
            x0 = float(x_edges[k])
            x1 = float(x_edges[k + 1])
            route = []
            for i, y in enumerate(y_vals):
                if i % 2 == 0:
                    route.append(np.array([x0, y], dtype=np.float32))
                    route.append(np.array([x1, y], dtype=np.float32))
                else:
                    route.append(np.array([x1, y], dtype=np.float32))
                    route.append(np.array([x0, y], dtype=np.float32))
            paths.append(route)
        return paths

    def _load_patrol_routes(self, route_path: str, route_names: List[str]) -> List[List[np.ndarray]]:
        """
        功能:
            读取目标巡逻路线数据。
        输入:
            route_path (str): 路径文件。
            route_names (list[str]): 路线名过滤。
        输出:
            list[list[np.ndarray]]: 全局坐标路线列表。
        """
        if len(str(route_path).strip()) == 0:
            return []
        path = route_path
        if not os.path.isabs(path):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(root, path)
        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        selected = set([str(x) for x in route_names if str(x).strip()])
        use_all = len(selected) == 0 or ("all" in [x.lower() for x in selected])
        coord_mode = str(data.get("meta", {}).get("coords", ""))
        out = []
        for item in data.get("routes", []):
            name = str(item.get("name", ""))
            if (not use_all) and (name not in selected):
                continue
            pts = np.asarray(item.get("waypoints", []), dtype=np.float32)
            if pts.size == 0:
                continue
            if coord_mode == "normalized_0_1":
                pts = (pts * 2.0 - 1.0) * self.world_size
            pts = np.clip(pts, -self.world_size, self.world_size)
            out.append([np.asarray(p, dtype=np.float32) for p in pts])
        return out

    def step(
        self,
        hunter_actions: Optional[np.ndarray],
        target_actions: Optional[np.ndarray],
    ) -> Tuple[bool, Dict[str, float]]:
        """
        功能:
            执行一步全局推理环境推进。
        输入:
            hunter_actions (Optional[np.ndarray]): Hunter动作，shape=(N,2)。
            target_actions (Optional[np.ndarray]): Target learn动作，shape=(M,2)。
        输出:
            tuple:
                - bool: 是否终止。
                - dict: 统计信息。
        """
        if self.done:
            return True, self.get_metrics()

        self._update_explorer_modes_and_assignment()

        self._step_explorers()
        self._step_targets(target_actions)
        self._step_hunters(hunter_actions)

        self._update_discovery_and_shared_memory()
        captured = self._update_capture_state()

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
        if int(np.sum(self.target_alive.astype(np.int32))) == 0:
            self.done = True

        info = self.get_metrics()
        info["new_captured"] = float(captured)
        return bool(self.done), info

    def _update_explorer_modes_and_assignment(self):
        """
        功能:
            更新 Explorer 的 SEARCH/TRACK 状态，并进行 hunter 分配。
        输入:
            无。
        输出:
            无。
        """
        for eid, explorer in enumerate(self.explorers):
            tracked_tid = int(self.explorer_track_target[eid])
            if tracked_tid >= 0 and (tracked_tid >= self.num_targets or not bool(self.target_alive[tracked_tid])):
                self.explorer_track_target[eid] = -1
                self.explorer_state[eid] = "SEARCH"

            if self.explorer_track_target[eid] >= 0:
                self.explorer_state[eid] = "TRACK"
                continue

            # SEARCH态尝试发现最近可见目标
            best_tid = -1
            best_dist = np.inf
            for tid, target in enumerate(self.targets):
                if not bool(self.target_alive[tid]):
                    continue
                d = float(np.linalg.norm(explorer.position - target.position))
                if d <= self.explorer_perception_radius and d < best_dist:
                    best_dist = d
                    best_tid = tid

            if best_tid >= 0:
                self.explorer_track_target[eid] = int(best_tid)
                self.explorer_state[eid] = "TRACK"
                self.target_discovered[best_tid] = True
                self._assign_hunters_to_target(best_tid)

        # 释放已死亡目标对应的hunter
        alive_targets = set([i for i in range(self.num_targets) if bool(self.target_alive[i])])
        for hid in range(self.num_hunters):
            t = int(self.hunter_assignment[hid])
            if t >= 0 and t not in alive_targets:
                self.hunter_assignment[hid] = -1

    def _assign_hunters_to_target(self, target_id: int):
        """
        功能:
            为指定目标分配固定数量 hunter（按距离优先）。
        输入:
            target_id (int): 目标索引。
        输出:
            无。
        """
        target_pos = self.targets[target_id].position

        assigned = [i for i, t in enumerate(self.hunter_assignment) if int(t) == int(target_id)]
        need = int(max(0, self.assignment_hunters_per_target - len(assigned)))
        if need <= 0:
            return

        idle_hunters = [i for i, t in enumerate(self.hunter_assignment) if int(t) < 0]
        if len(idle_hunters) == 0:
            return

        idle_hunters.sort(key=lambda hid: float(np.linalg.norm(self.hunters[hid].position - target_pos)))
        for hid in idle_hunters[:need]:
            self.hunter_assignment[hid] = int(target_id)

    def _step_explorers(self):
        """
        功能:
            按状态机推进 Explorer 位置。
        输入:
            无。
        输出:
            无。
        """
        for eid, explorer in enumerate(self.explorers):
            if not explorer.alive:
                continue
            if self.explorer_state[eid] == "TRACK" and self.explorer_track_target[eid] >= 0:
                tid = int(self.explorer_track_target[eid])
                target = self.targets[tid]
                if target.alive:
                    action = self._towards_action(explorer.position, target.position)
                else:
                    action = np.zeros(2, dtype=np.float32)
            else:
                path = self.explorer_paths[eid]
                if len(path) == 0:
                    action = np.zeros(2, dtype=np.float32)
                else:
                    wp = path[self.explorer_path_idx[eid]]
                    action = self._towards_action(explorer.position, wp)
                    if float(np.linalg.norm(explorer.position - wp)) <= max(1.0, explorer.max_speed * self.dt):
                        self.explorer_path_idx[eid] = (self.explorer_path_idx[eid] + 1) % len(path)
            explorer.step(action, self.dt, self.world_size)

    def _step_targets(self, target_actions: Optional[np.ndarray]):
        """
        功能:
            推进所有 target 状态。
        输入:
            target_actions (Optional[np.ndarray]): learn target 的动作输入。
        输出:
            无。
        """
        for tid, target in enumerate(self.targets):
            if not bool(self.target_alive[tid]):
                continue
            if str(target.policy_type).lower() == "static":
                action = np.zeros(2, dtype=np.float32)
            elif str(target.policy_type).lower() == "learn":
                if target_actions is None:
                    action = np.zeros(2, dtype=np.float32)
                else:
                    action = np.asarray(target_actions[tid], dtype=np.float32)
            else:
                action = target.select_action(
                    step_count=int(self.step_count),
                    action_from_policy=None,
                    rng=self.rng,
                )
            target.step(action, self.dt, self.world_size)

    def _step_hunters(self, hunter_actions: Optional[np.ndarray]):
        """
        功能:
            推进 hunter 状态（仅对已分配目标的hunter执行动作）。
        输入:
            hunter_actions (Optional[np.ndarray]): hunter动作输入。
        输出:
            无。
        """
        for hid, hunter in enumerate(self.hunters):
            if not hunter.alive:
                continue
            assigned_tid = int(self.hunter_assignment[hid])
            if assigned_tid < 0 or (not bool(self.target_alive[assigned_tid])):
                action = np.zeros(2, dtype=np.float32)
            else:
                if hunter_actions is None:
                    # 无模型动作时退化为追踪目标
                    action = self._towards_action(hunter.position, self.targets[assigned_tid].position)
                else:
                    action = np.asarray(hunter_actions[hid], dtype=np.float32)
            hunter.step(action, self.dt, self.world_size)

    def _towards_action(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        功能:
            计算从src指向dst的单位归一化动作。
        输入:
            src (np.ndarray): 起点坐标。
            dst (np.ndarray): 终点坐标。
        输出:
            np.ndarray: shape=(2,) 的归一化动作。
        """
        vec = np.asarray(dst, dtype=np.float32) - np.asarray(src, dtype=np.float32)
        d = float(np.linalg.norm(vec))
        if d <= 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (vec / d).astype(np.float32)

    def _update_discovery_and_shared_memory(self):
        """
        功能:
            根据 explorer 跟踪状态更新目标共享记忆。
        输入:
            无。
        输出:
            无。
        """
        for tid in range(self.num_targets):
            if not bool(self.target_alive[tid]):
                self.shared_target_info[tid]["valid"] = False
                continue

            tracker_found = False
            for eid in range(self.num_explorers):
                if int(self.explorer_track_target[eid]) == int(tid):
                    tracker_found = True
                    self.shared_target_info[tid]["valid"] = True
                    self.shared_target_info[tid]["pos"] = self.targets[tid].position.copy()
                    self.shared_target_info[tid]["vel"] = self.targets[tid].velocity.copy()
                    self.shared_target_info[tid]["age"] = np.float32(0.0)
                    self.shared_target_info[tid]["timestamp"] = np.int32(self.step_count)
                    break

            if not tracker_found:
                if bool(self.shared_target_info[tid]["valid"]):
                    self.shared_target_info[tid]["age"] = np.float32(self.shared_target_info[tid]["age"] + 1.0)

    def _update_capture_state(self) -> int:
        """
        功能:
            更新捕获计数并标记目标是否被捕获。
        输入:
            无。
        输出:
            int: 本步新捕获目标数量。
        """
        new_captured = 0
        for tid, target in enumerate(self.targets):
            if not bool(self.target_alive[tid]):
                continue

            captured = False
            for hid, hunter in enumerate(self.hunters):
                if not hunter.alive:
                    self.target_capture_counter[hid, tid] = 0
                    continue
                if int(self.hunter_assignment[hid]) != int(tid):
                    self.target_capture_counter[hid, tid] = 0
                    continue
                d = float(np.linalg.norm(hunter.position - target.position))
                if d <= self.capture_dis:
                    self.target_capture_counter[hid, tid] += 1
                else:
                    self.target_capture_counter[hid, tid] = 0
                if int(self.target_capture_counter[hid, tid]) >= int(self.capture_step):
                    captured = True

            if captured:
                self.target_alive[tid] = False
                target.alive = False
                target.velocity[:] = 0.0
                self.shared_target_info[tid]["valid"] = False
                new_captured += 1

        return int(new_captured)

    def get_hunter_obs(self, hunter_id: int) -> np.ndarray:
        """
        功能:
            构造单个 hunter 的推理观测（复用 N v 1 观测结构）。
        输入:
            hunter_id (int): hunter索引。
        输出:
            np.ndarray: shape=(obs_dim,)。
        """
        hunter = self.hunters[hunter_id]
        if not hunter.alive:
            return np.zeros(self.obs_dim, dtype=np.float32)

        scale = max(float(self.world_size), 1e-6)
        own = np.concatenate([hunter.position / scale, hunter.velocity / scale]).astype(np.float32)

        # 邻居使用同一目标分配组
        assigned_tid = int(self.hunter_assignment[hunter_id])
        neighbors = []
        if assigned_tid >= 0:
            for j, hj in enumerate(self.hunters):
                if j == hunter_id:
                    continue
                if not hj.alive:
                    continue
                if int(self.hunter_assignment[j]) != assigned_tid:
                    continue
                d = float(np.linalg.norm(hj.position - hunter.position))
                neighbors.append((d, j))
        neighbors.sort(key=lambda x: x[0])

        slots = []
        for _, j in neighbors[: self.neighbor_N]:
            hj = self.hunters[j]
            rel_pos = (hj.position - hunter.position) / scale
            rel_vel = (hj.velocity - hunter.velocity) / scale
            dist = float(np.linalg.norm(hj.position - hunter.position)) / scale
            slots.append(np.array([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], dist, 1.0], dtype=np.float32))
        while len(slots) < self.neighbor_N:
            slots.append(np.zeros(self.neighbor_feat_dim, dtype=np.float32))
        neighbor_obs = np.concatenate(slots, axis=0).astype(np.float32) if self.neighbor_N > 0 else np.zeros(0, dtype=np.float32)

        # target直接可见 or explorer共享记忆
        target_obs = np.zeros(self.target_feat_dim, dtype=np.float32)
        memory_obs = np.zeros(5, dtype=np.float32)
        if assigned_tid >= 0 and bool(self.target_alive[assigned_tid]):
            target = self.targets[assigned_tid]
            d = float(np.linalg.norm(target.position - hunter.position))
            visible = (self.hunter_perception_radius < 0) or (d <= float(self.hunter_perception_radius))
            if visible:
                rel_pos = (target.position - hunter.position) / scale
                rel_vel = (target.velocity - hunter.velocity) / scale
                target_obs = np.array([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], d / scale, 1.0], dtype=np.float32)
            else:
                info = self.shared_target_info[assigned_tid]
                if bool(info["valid"]):
                    rel_pos = (np.asarray(info["pos"], dtype=np.float32) - hunter.position) / scale
                    rel_vel = (np.asarray(info["vel"], dtype=np.float32) - hunter.velocity) / scale
                    age_norm = float(info["age"]) / float(max(1, self.max_steps))
                    memory_obs = np.array([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], age_norm], dtype=np.float32)

        return np.concatenate([own, neighbor_obs, target_obs, memory_obs], axis=0).astype(np.float32)

    def get_target_obs(self, target_id: int) -> np.ndarray:
        """
        功能:
            构造单个 target 的推理观测。
        输入:
            target_id (int): 目标索引。
        输出:
            np.ndarray: shape=(obs_dim,)。
        """
        target = self.targets[target_id]
        if not target.alive:
            return np.zeros(self.obs_dim, dtype=np.float32)

        scale = max(float(self.world_size), 1e-6)
        own = np.concatenate([target.position / scale, target.velocity / scale]).astype(np.float32)

        # target无同阵营邻居，neighbor全零
        neighbor_obs = np.zeros(self.neighbor_N * self.neighbor_feat_dim, dtype=np.float32)

        # 最近已分配且存活hunter
        assigned_hunters = [
            hid for hid in range(self.num_hunters)
            if self.hunters[hid].alive and int(self.hunter_assignment[hid]) == int(target_id)
        ]
        target_obs = np.zeros(self.target_feat_dim, dtype=np.float32)
        if len(assigned_hunters) > 0:
            nearest = min(assigned_hunters, key=lambda hid: float(np.linalg.norm(self.hunters[hid].position - target.position)))
            h = self.hunters[nearest]
            d = float(np.linalg.norm(h.position - target.position))
            rel_pos = (h.position - target.position) / scale
            rel_vel = (h.velocity - target.velocity) / scale
            target_obs = np.array([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], d / scale, 1.0], dtype=np.float32)

        memory_obs = np.zeros(5, dtype=np.float32)
        return np.concatenate([own, neighbor_obs, target_obs, memory_obs], axis=0).astype(np.float32)

    def get_summary(self) -> Dict[str, float]:
        """
        功能:
            返回当前环境摘要信息。
        输入:
            无。
        输出:
            dict: 摘要字典。
        """
        return {
            "step": float(self.step_count),
            "targets_alive": float(np.sum(self.target_alive.astype(np.int32))),
            "targets_captured": float(self.num_targets - int(np.sum(self.target_alive.astype(np.int32)))),
            "targets_discovered": float(np.sum(self.target_discovered.astype(np.int32))),
        }

    def get_metrics(self) -> Dict[str, float]:
        """
        功能:
            计算运行指标。
        输入:
            无。
        输出:
            dict: 指标字典。
        """
        alive_targets = int(np.sum(self.target_alive.astype(np.int32)))
        captured = int(self.num_targets - alive_targets)
        capture_rate = float(captured / max(1, self.num_targets))
        discovered = int(np.sum(self.target_discovered.astype(np.int32)))
        discover_rate = float(discovered / max(1, self.num_targets))
        return {
            "step": float(self.step_count),
            "targets_alive": float(alive_targets),
            "targets_captured": float(captured),
            "capture_rate": float(capture_rate),
            "discover_rate": float(discover_rate),
        }

    def _draw_scene(self, ax):
        """
        功能:
            在给定坐标轴上绘制当前场景。
        输入:
            ax (matplotlib.axes.Axes): 绘图坐标轴。
        输出:
            无。
        """
        ws = float(self.world_size)
        ax.set_xlim(-ws, ws)
        ax.set_ylim(-ws, ws)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.35)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(
            "Step {}/{} | Captured {}/{} | Discover {:.2f}".format(
                int(self.step_count),
                int(self.max_steps),
                int(self.num_targets - int(np.sum(self.target_alive.astype(np.int32)))),
                int(self.num_targets),
                float(np.sum(self.target_discovered.astype(np.int32)) / max(1, self.num_targets)),
            )
        )

        # 每个Explorer对应的搜索航线（按eid配色）
        explorer_route_colors = ["#9ecae1", "#c7e9c0", "#fdd0a2", "#d4b9da", "#fdae6b", "#a1d99b"]
        for eid, path in enumerate(self.explorer_paths):
            if len(path) <= 1:
                continue
            pts = np.asarray(path, dtype=np.float32)
            route_color = explorer_route_colors[eid % len(explorer_route_colors)]
            ax.plot(pts[:, 0], pts[:, 1], color=route_color, linewidth=1.0, alpha=0.5)

        # hunters
        hunter_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#17becf", "#8c564b", "#bcbd22"]
        for hid, h in enumerate(self.hunters):
            color = hunter_colors[hid % len(hunter_colors)]
            assigned = int(self.hunter_assignment[hid]) >= 0
            alpha = 1.0 if (h.alive and assigned) else (0.35 if h.alive else 0.2)
            ax.scatter([h.position[0]], [h.position[1]], c=[color], s=50, marker="o", alpha=alpha, edgecolors="black", linewidths=0.4)
            label = f"H{hid}" if assigned else f"H{hid}(idle)"
            ax.text(float(h.position[0]), float(h.position[1]), label, fontsize=7, color=color, alpha=alpha)

            tid = int(self.hunter_assignment[hid])
            if tid >= 0 and tid < self.num_targets and bool(self.target_alive[tid]):
                tp = self.targets[tid].position
                ax.plot(
                    [float(h.position[0]), float(tp[0])],
                    [float(h.position[1]), float(tp[1])],
                    color=color,
                    linestyle="--",
                    linewidth=0.6,
                    alpha=0.35,
                )

        # explorers
        for eid, e in enumerate(self.explorers):
            mode_txt = str(self.explorer_state[eid])
            color = "#9467bd" if mode_txt == "SEARCH" else "#e377c2"
            alpha = 0.45 if (e.alive and mode_txt == "SEARCH") else (1.0 if e.alive else 0.25)
            ax.scatter([e.position[0]], [e.position[1]], c=[color], s=65, marker="^", alpha=alpha, edgecolors="black", linewidths=0.5)
            ax.add_patch(
                plt.Circle(
                    (float(e.position[0]), float(e.position[1])),
                    float(self.explorer_perception_radius),
                    color=color,
                    fill=False,
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.2,
                )
            )
            ax.text(float(e.position[0]), float(e.position[1]), f"E{eid}:{mode_txt}", fontsize=7, color=color, alpha=alpha)

            tid = int(self.explorer_track_target[eid])
            if tid >= 0 and tid < self.num_targets and bool(self.target_alive[tid]):
                tp = self.targets[tid].position
                ax.plot(
                    [float(e.position[0]), float(tp[0])],
                    [float(e.position[1]), float(tp[1])],
                    color=color,
                    linestyle="-.",
                    linewidth=0.9,
                    alpha=0.45,
                )

        # targets
        for tid, t in enumerate(self.targets):
            alive = bool(self.target_alive[tid]) and bool(t.alive)
            discovered = bool(self.target_discovered[tid])
            color = "#d62728" if alive else "#7f7f7f"
            # 未被发现半透明，被发现不透明；死亡时低透明。
            alpha = 1.0 if (alive and discovered) else (0.45 if alive else 0.3)
            marker = "s" if alive else "X"
            ax.scatter([t.position[0]], [t.position[1]], c=[color], s=70, marker=marker, alpha=alpha, edgecolors="black", linewidths=0.5)
            if alive:
                status = "FOUND" if discovered else "HIDDEN"
            else:
                status = "DEAD"
            ax.text(
                float(t.position[0]),
                float(t.position[1]),
                f"T{tid}:{str(t.policy_type)[:2]}:{status}",
                fontsize=7,
                color=color,
                alpha=alpha,
            )

            if alive:
                ax.add_patch(
                    plt.Circle(
                        (float(t.position[0]), float(t.position[1])),
                        float(self.capture_dis),
                        color="#d62728",
                        fill=False,
                        linestyle="--",
                        linewidth=0.9,
                        alpha=0.5 if discovered else 0.18,
                    )
                )

        # shared memory marks
        for tid, info in enumerate(self.shared_target_info):
            if not bool(info["valid"]):
                continue
            p = np.asarray(info["pos"], dtype=np.float32)
            ax.scatter([p[0]], [p[1]], c=["#e377c2"], marker="*", s=130, alpha=0.9, edgecolors="black", linewidths=0.4)
            ax.text(float(p[0]), float(p[1]), f"S{tid}", fontsize=7, color="#e377c2")

        legend_handles = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728", markeredgecolor="black", markersize=8, label="Target (found, alive)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728", markeredgecolor="black", alpha=0.45, markersize=8, label="Target (hidden, alive)"),
            Line2D([0], [0], marker="X", color="w", markerfacecolor="#7f7f7f", markeredgecolor="black", markersize=8, label="Target dead"),
            Line2D([0], [0], color="#d62728", linestyle="--", linewidth=1.0, label="Capture radius"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markeredgecolor="black", markersize=7, label="Hunter assigned"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markeredgecolor="black", alpha=0.35, markersize=7, label="Hunter idle"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#9467bd", markeredgecolor="black", alpha=0.45, markersize=8, label="Explorer SEARCH"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#e377c2", markeredgecolor="black", markersize=8, label="Explorer TRACK"),
            Line2D([0], [0], color="#9ecae1", linewidth=1.0, label="Explorer route segment"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#e377c2", markeredgecolor="black", markersize=10, label="Shared target memory"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, framealpha=0.85, borderaxespad=0.0)

    def render(self, mode: str = "rgb_array"):
        """
        功能:
            渲染当前全局场景。
        输入:
            mode (str): 渲染模式，支持 \"rgb_array\" 和 \"human\"。
        输出:
            np.ndarray | None: rgb_array时返回图像，human时返回None。
        """
        if mode not in ("rgb_array", "human"):
            raise NotImplementedError(f"Unsupported render mode: {mode}")

        if mode == "human":
            if self._human_fig is None or self._human_ax is None:
                plt.ion()
                self._human_fig, self._human_ax = plt.subplots(figsize=(7.2, 7.2), dpi=100)
                self._human_fig.subplots_adjust(right=0.72)
            self._human_ax.clear()
            self._draw_scene(self._human_ax)
            self._human_fig.canvas.draw_idle()
            self._human_fig.canvas.flush_events()
            plt.pause(0.001)
            return None

        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=100)
        fig.subplots_adjust(right=0.72)
        self._draw_scene(ax)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
        plt.close(fig)
        return img

    def close(self):
        """
        功能:
            关闭绘图资源。
        输入:
            无。
        输出:
            无。
        """
        if self._human_fig is not None:
            plt.close(self._human_fig)
        self._human_fig = None
        self._human_ax = None
        plt.close("all")
