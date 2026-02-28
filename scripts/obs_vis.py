#!/usr/bin/env python3
"""Interactive observation visualizer driven by ContinuousActionEnv."""

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)

import argparse
import base64
import io
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import imageio.v2 as imageio
import numpy as np
import yaml
from easydict import EasyDict as edict

from envs.env_continuous import ContinuousActionEnv


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def to_edict(obj):
    if isinstance(obj, dict):
        return edict({k: to_edict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_edict(v) for v in obj]
    return obj


def load_config(config_path: Path) -> edict:
    root = Path(__file__).resolve().parents[1]
    default_path = root / "config" / "defaults.yaml"
    with default_path.open("r", encoding="utf-8") as f:
        defaults = yaml.safe_load(f) or {}

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    else:
        user_cfg = {}

    merged = deep_merge(defaults, user_cfg)
    return to_edict(merged)


class ObsVisApp:
    def __init__(self, root: tk.Tk, config_path: Path) -> None:
        self.root = root
        self.config_path = config_path
        try:
            self.initial_cfg = load_config(self.config_path)
        except Exception:
            self.initial_cfg = None

        self.env_wrapper: ContinuousActionEnv | None = None
        self.core_env = None

        self.obs_cache: list[np.ndarray] = []
        self.last_step_rewards: np.ndarray | None = None
        self.last_step_dones: np.ndarray | None = None
        self.reward_history: list[np.ndarray] = []
        self.reward_term_history: dict[str, list[np.ndarray]] = {}
        self.reward_agent_visible_vars: list[tk.BooleanVar] = []
        self.reward_term_color_cache: dict[str, str] = {}

        self.selected_index: int | None = None
        self.dragging = False
        self.render_photo = None

        self.canvas_size = 600
        self.canvas_pad = 30
        self.view_row_height_var = tk.IntVar(value=430)
        self.reward_panel_height_var = tk.IntVar(value=340)

        self._build_ui()
        self.init_env()

    def _build_ui(self) -> None:
        self.root.title("UAV Obs Visualizer (ContinuousActionEnv)")
        self.root.geometry("1280x760")

        top_obs = tk.LabelFrame(self.root, text="Obs Vector (Selected Agent)", padx=6, pady=6)
        top_obs.pack(fill=tk.BOTH, padx=8, pady=(8, 0))

        self.obs_two_line_var = tk.StringVar(value="")
        tk.Label(
            top_obs,
            textvariable=self.obs_two_line_var,
            justify=tk.LEFT,
            anchor="w",
            font=("Courier New", 10),
        ).pack(fill=tk.X)

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = tk.Frame(main, padx=8)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        layout_ctrl = tk.LabelFrame(left, text="Layout", padx=4, pady=4)
        layout_ctrl.pack(fill=tk.X, expand=False, pady=(0, 6))
        tk.Label(layout_ctrl, text="View Height", anchor="w").pack(side=tk.LEFT)
        tk.Scale(
            layout_ctrl,
            variable=self.view_row_height_var,
            from_=220,
            to=860,
            orient=tk.HORIZONTAL,
            resolution=10,
            showvalue=True,
            length=220,
            command=self._on_view_height_change,
        ).pack(side=tk.LEFT, padx=(8, 16))

        vis_row = tk.Frame(left)
        vis_row.pack(fill=tk.X, expand=False)
        vis_row.pack_propagate(False)
        vis_row.configure(height=int(self.view_row_height_var.get()))
        self.vis_row = vis_row

        drag_frame = tk.LabelFrame(vis_row, text="Drag/Edit View", padx=4, pady=4)
        drag_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.canvas = tk.Canvas(
            drag_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
            highlightthickness=1,
            highlightbackground="#808080",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Configure>", lambda _e: self._draw_scene())

        render_frame = tk.LabelFrame(
            vis_row,
            text='ContinuousActionEnv.render(mode="rgb_array")',
            padx=4,
            pady=4,
        )
        render_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        render_top = tk.Frame(render_frame)
        render_top.pack(fill=tk.BOTH, expand=True)
        self.render_label = tk.Label(render_top, bd=1, relief=tk.SOLID)
        self.render_label.pack(fill=tk.BOTH, expand=True)
        self.render_label.bind("<Configure>", lambda _e: self._update_render_preview())

        reward_panel = tk.LabelFrame(left, text="Agent Rewards per Step", padx=4, pady=4)
        reward_panel.pack(fill=tk.X, expand=False, pady=(6, 0))
        reward_panel.pack_propagate(False)
        reward_panel.configure(height=int(self.reward_panel_height_var.get()))

        reward_ctrl = tk.Frame(reward_panel)
        reward_ctrl.pack(fill=tk.X, pady=(0, 4))
        tk.Label(reward_ctrl, text="Reward Height", anchor="w").pack(side=tk.LEFT)
        tk.Scale(
            reward_ctrl,
            variable=self.reward_panel_height_var,
            from_=220,
            to=700,
            orient=tk.HORIZONTAL,
            resolution=10,
            showvalue=True,
            length=220,
            command=self._on_reward_height_change,
        ).pack(side=tk.LEFT, padx=(8, 0))

        reward_row = tk.Frame(reward_panel)
        reward_row.pack(fill=tk.BOTH, expand=True)
        self.reward_panel = reward_panel
        self.reward_row = reward_row

        self.reward_canvas_inst = tk.Canvas(
            reward_row,
            width=560,
            height=160,
            bg="white",
            highlightthickness=1,
            highlightbackground="#BBBBBB",
        )
        self.reward_canvas_inst.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 3))
        self.reward_canvas_inst.bind("<Configure>", lambda _e: self._draw_reward_plots())
        self.reward_canvas_cum = tk.Canvas(
            reward_row,
            width=560,
            height=160,
            bg="white",
            highlightthickness=1,
            highlightbackground="#BBBBBB",
        )
        self.reward_canvas_cum.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(3, 0))
        self.reward_canvas_cum.bind("<Configure>", lambda _e: self._draw_reward_plots())

        self.status_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.status_var, anchor="w").pack(fill=tk.X, pady=(8, 0))

        tabs = ttk.Notebook(right)
        tabs.pack(fill=tk.X, expand=False)
        tab_setup = tk.Frame(tabs)
        tab_agent = tk.Frame(tabs)
        tabs.add(tab_setup, text="Scene Setup")
        tabs.add(tab_agent, text="Agent+Control")

        cfg_frame = tk.LabelFrame(tab_setup, text="Scene Setup", padx=6, pady=6)
        cfg_frame.pack(fill=tk.X, pady=(6, 0))

        world_size = self._cfg_value(("env", "world_size"), 400.0)
        num_hunters = self._cfg_value(("env", "num_hunters"), 3)
        seed = self._cfg_value(("exp", "seed"), 1)
        hunter_max = self._cfg_value(("Hunter", "max_velo"), 20.0)
        hunter_perc = self._cfg_value(("Hunter", "perception_radius"), 25.0)
        target_max = self._cfg_value(("Target", "max_velo"), 12.0)
        target_perc = self._cfg_value(("Target", "perception_radius"), 40.0)
        target_policy = str(self._cfg_value(("env", "target_policy_source"), "learn")).lower()
        if target_policy not in ("learn", "random", "patrol"):
            target_policy = "learn"

        self.world_size_var = tk.DoubleVar(value=float(world_size))
        self.num_hunters_var = tk.IntVar(value=int(num_hunters))
        self.seed_var = tk.IntVar(value=int(seed))
        self.hunter_max_vel_var = tk.DoubleVar(value=float(hunter_max))
        self.hunter_perc_var = tk.DoubleVar(value=float(hunter_perc))
        self.target_max_vel_var = tk.DoubleVar(value=float(target_max))
        self.target_perc_var = tk.DoubleVar(value=float(target_perc))
        self.target_policy_var = tk.StringVar(value=target_policy)
        self.step_repeat_var = tk.IntVar(value=10)

        self._add_labeled_entry(cfg_frame, "World Size", self.world_size_var)
        self._add_labeled_entry(cfg_frame, "Num Hunters", self.num_hunters_var)
        self._add_labeled_entry(cfg_frame, "Seed", self.seed_var)
        self._add_labeled_entry(cfg_frame, "Hunter Max V", self.hunter_max_vel_var)
        self._add_labeled_entry(cfg_frame, "Hunter Perception", self.hunter_perc_var)
        self._add_labeled_entry(cfg_frame, "Target Max V", self.target_max_vel_var)
        self._add_labeled_entry(cfg_frame, "Target Perception", self.target_perc_var)

        row = tk.Frame(cfg_frame)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="Target Policy", width=16, anchor="w").pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.target_policy_var,
            values=["learn", "random", "patrol"],
            width=14,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(6, 0))

        tk.Button(cfg_frame, text="Initialize Scene", command=self.init_env).pack(fill=tk.X, pady=(6, 2))
        tk.Button(cfg_frame, text="Randomize Positions", command=self.randomize_positions).pack(fill=tk.X, pady=2)
        tk.Button(cfg_frame, text="Zero All Velocities", command=self.zero_velocities).pack(fill=tk.X, pady=2)

        tk.Button(cfg_frame, text="Step Once", command=self.step_once).pack(fill=tk.X, pady=(8, 2))
        step_row = tk.Frame(cfg_frame)
        step_row.pack(fill=tk.X, pady=2)
        tk.Label(step_row, text="Step N", width=10, anchor="w").pack(side=tk.LEFT)
        tk.Entry(step_row, textvariable=self.step_repeat_var, width=8).pack(side=tk.LEFT, padx=(6, 6))
        tk.Button(step_row, text="Run", command=self.step_n).pack(side=tk.LEFT)

        agent_frame = tk.LabelFrame(tab_agent, text="Agents", padx=6, pady=6)
        agent_frame.pack(fill=tk.X, pady=(6, 0))
        self.agent_listbox = tk.Listbox(agent_frame, width=42, height=8, exportselection=False)
        self.agent_listbox.pack(fill=tk.X)
        self.agent_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        reward_agent_frame = tk.LabelFrame(tab_agent, text="Reward Curves - Visible Agents", padx=6, pady=6)
        reward_agent_frame.pack(fill=tk.X, pady=(8, 0))
        reward_agent_btn_row = tk.Frame(reward_agent_frame)
        reward_agent_btn_row.pack(fill=tk.X)
        tk.Button(reward_agent_btn_row, text="All", command=self._set_all_reward_agents_visible).pack(
            side=tk.LEFT
        )
        tk.Button(reward_agent_btn_row, text="None", command=self._set_no_reward_agents_visible).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        self.reward_agent_checks_frame = tk.Frame(reward_agent_frame)
        self.reward_agent_checks_frame.pack(fill=tk.X, pady=(6, 0))

        edit_frame = tk.LabelFrame(tab_agent, text="Selected Agent Control", padx=6, pady=6)
        edit_frame.pack(fill=tk.X, pady=(8, 0))

        self.pos_x_var = tk.DoubleVar(value=0.0)
        self.pos_y_var = tk.DoubleVar(value=0.0)
        self.heading_var = tk.DoubleVar(value=0.0)
        self.speed_ratio_var = tk.DoubleVar(value=0.0)

        self._add_labeled_entry(edit_frame, "Pos X", self.pos_x_var)
        self._add_labeled_entry(edit_frame, "Pos Y", self.pos_y_var)
        tk.Button(edit_frame, text="Apply Position", command=self.apply_selected_position).pack(fill=tk.X, pady=2)

        tk.Label(edit_frame, text="Heading (deg)", anchor="w").pack(fill=tk.X, pady=(6, 0))
        tk.Scale(
            edit_frame,
            variable=self.heading_var,
            from_=-180,
            to=180,
            orient=tk.HORIZONTAL,
            resolution=1,
            showvalue=True,
            length=280,
        ).pack(fill=tk.X)

        tk.Label(edit_frame, text="Speed Ratio [0,1]", anchor="w").pack(fill=tk.X, pady=(6, 0))
        tk.Scale(
            edit_frame,
            variable=self.speed_ratio_var,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            resolution=0.01,
            showvalue=True,
            length=280,
        ).pack(fill=tk.X)

        tk.Button(edit_frame, text="Apply Velocity", command=self.apply_selected_velocity).pack(fill=tk.X, pady=2)
        tk.Button(edit_frame, text="Stop Selected", command=self.stop_selected).pack(fill=tk.X, pady=2)

        info_frame = tk.LabelFrame(right, text="Selected Agent Debug", padx=6, pady=6)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.info_var = tk.StringVar(value="")
        tk.Label(info_frame, textvariable=self.info_var, justify=tk.LEFT, anchor="w").pack(fill=tk.X)
        self.obs_text = tk.Text(info_frame, height=10, width=44, wrap=tk.WORD)
        self.obs_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self._on_view_height_change()
        self._on_reward_height_change()

    def _add_labeled_entry(self, parent: tk.Widget, label: str, var) -> None:
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, width=16, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=var, width=14).pack(side=tk.LEFT, padx=(6, 0))

    def _cfg_value(self, keys: tuple[str, ...], default):
        node = self.initial_cfg
        try:
            for k in keys:
                node = node[k] if isinstance(node, dict) else getattr(node, k)
            return node
        except Exception:
            return default

    def init_env(self) -> None:
        try:
            cfg = load_config(self.config_path)
            cfg.env.world_size = float(self.world_size_var.get())
            cfg.env.max_hunters_num = int(self.num_hunters_var.get())
            cfg.env.num_explorers = 0
            cfg.exp.seed = int(self.seed_var.get())
            cfg.Hunter.max_velo = float(self.hunter_max_vel_var.get())
            cfg.Hunter.perception_radius = float(self.hunter_perc_var.get())
            cfg.Target.max_velo = float(self.target_max_vel_var.get())
            cfg.Target.perception_radius = float(self.target_perc_var.get())
            cfg.env.target_policy_source = str(self.target_policy_var.get())
        except Exception as exc:
            messagebox.showerror("Config Error", f"Invalid setup values:\n{exc}")
            return

        try:
            self.env_wrapper = ContinuousActionEnv(cfg)
            self.env_wrapper.seed(int(self.seed_var.get()))
            obs = self.env_wrapper.reset()
            self.obs_cache = [obs[i].copy() for i in range(obs.shape[0])]
            self.core_env = self.env_wrapper.env
            self.last_step_rewards = None
            self.last_step_dones = None
            self.reward_history = []
            self.reward_term_history = {}
        except Exception as exc:
            messagebox.showerror("Env Error", f"Failed to initialize env:\n{exc}")
            return

        self.selected_index = 0 if self.core_env and self.core_env.agent_num > 0 else None
        self._refresh_agent_list()
        self.update_debug_views(recompute_obs=False)
        self.set_status("Scene initialized via ContinuousActionEnv.")

    def randomize_positions(self) -> None:
        if self.core_env is None:
            return
        ws = float(self.core_env.world_size)
        for agent in self.core_env.agents:
            agent.position = self.core_env.rng.uniform(-ws, ws, size=2).astype(np.float32)
            agent.trajectory = [agent.position.copy()]
        self.update_debug_views(recompute_obs=True)
        self.set_status("Randomized all agent positions.")

    def zero_velocities(self) -> None:
        if self.core_env is None:
            return
        for agent in self.core_env.agents:
            agent.velocity[:] = 0.0
        self.update_debug_views(recompute_obs=True)
        self.set_status("Set all velocities to zero.")

    def _build_actions_from_current_velocity(self) -> np.ndarray:
        if self.core_env is None:
            return np.zeros((0, 2), dtype=np.float32)
        actions = np.zeros((self.core_env.agent_num, self.core_env.action_dim), dtype=np.float32)
        for i, agent in enumerate(self.core_env.agents):
            max_speed = max(float(agent.max_speed), 1e-6)
            actions[i] = np.clip(np.asarray(agent.velocity, dtype=np.float32) / max_speed, -1.0, 1.0)
        return actions

    def _record_reward_terms(self, infos: list[dict]) -> None:
        n_agent = len(infos)
        term_keys = set()
        for info in infos:
            if isinstance(info, dict):
                for key in info.keys():
                    if str(key).startswith("reward_"):
                        term_keys.add(str(key))
        term_keys.add("reward_total")
        for key in sorted(term_keys):
            vals = np.zeros(n_agent, dtype=np.float32)
            for i, info in enumerate(infos):
                if isinstance(info, dict):
                    vals[i] = float(info.get(key, 0.0))
            self.reward_term_history.setdefault(key, []).append(vals)

    def step_once(self) -> None:
        if self.env_wrapper is None or self.core_env is None:
            return
        actions = self._build_actions_from_current_velocity()
        obs, rews, dones, infos = self.env_wrapper.step(actions)
        self.obs_cache = [obs[i].copy() for i in range(obs.shape[0])]
        self.last_step_rewards = rews.copy()
        self.last_step_dones = dones.copy()
        self.reward_history.append(np.asarray(rews, dtype=np.float32).reshape(-1))
        self._record_reward_terms(infos)
        self.update_debug_views(recompute_obs=False)
        self.set_status(f"Step done. step_count={self.core_env.step_count}")

    def step_n(self) -> None:
        if self.env_wrapper is None or self.core_env is None:
            return
        n = max(1, int(self.step_repeat_var.get()))
        executed = 0
        for _ in range(n):
            if np.all(self.core_env.done):
                break
            actions = self._build_actions_from_current_velocity()
            obs, rews, dones, infos = self.env_wrapper.step(actions)
            self.obs_cache = [obs[i].copy() for i in range(obs.shape[0])]
            self.last_step_rewards = rews.copy()
            self.last_step_dones = dones.copy()
            self.reward_history.append(np.asarray(rews, dtype=np.float32).reshape(-1))
            self._record_reward_terms(infos)
            executed += 1
        self.update_debug_views(recompute_obs=False)
        self.set_status(f"Stepped {executed}/{n}. step_count={self.core_env.step_count}")

    def _step_with_selected_action(self, selected_action: np.ndarray, status_prefix: str) -> None:
        if self.env_wrapper is None or self.core_env is None or self.selected_index is None:
            return
        actions = self._build_actions_from_current_velocity()
        actions[self.selected_index] = np.clip(selected_action.astype(np.float32), -1.0, 1.0)
        obs, rews, dones, infos = self.env_wrapper.step(actions)
        self.obs_cache = [obs[i].copy() for i in range(obs.shape[0])]
        self.last_step_rewards = rews.copy()
        self.last_step_dones = dones.copy()
        self.reward_history.append(np.asarray(rews, dtype=np.float32).reshape(-1))
        self._record_reward_terms(infos)
        self.update_debug_views(recompute_obs=False)
        self.set_status(f"{status_prefix} step_count={self.core_env.step_count}")

    def on_listbox_select(self, _event=None) -> None:
        if self.core_env is None or not self.agent_listbox.curselection():
            return
        self.selected_index = int(self.agent_listbox.curselection()[0])
        self.update_debug_views(recompute_obs=False)

    def apply_selected_position(self) -> None:
        if self.core_env is None or self.selected_index is None:
            return
        ws = float(self.core_env.world_size)
        x = float(self.pos_x_var.get())
        y = float(self.pos_y_var.get())
        agent = self.core_env.agents[self.selected_index]
        agent.position = np.array([np.clip(x, -ws, ws), np.clip(y, -ws, ws)], dtype=np.float32)
        agent.trajectory.append(agent.position.copy())
        self.update_debug_views(recompute_obs=True)
        self.set_status(f"Updated position of {agent.role}[{agent.agent_id}]")

    def apply_selected_velocity(self) -> None:
        if self.core_env is None or self.selected_index is None:
            return
        agent = self.core_env.agents[self.selected_index]
        theta = math.radians(float(self.heading_var.get()))
        ratio = float(np.clip(self.speed_ratio_var.get(), 0.0, 1.0))
        speed = ratio * float(agent.max_speed)
        agent.velocity = np.array([math.cos(theta) * speed, math.sin(theta) * speed], dtype=np.float32)
        self.update_debug_views(recompute_obs=True)
        self.set_status(f"Updated velocity of {agent.role}[{agent.agent_id}]")

    def stop_selected(self) -> None:
        if self.core_env is None or self.selected_index is None:
            return
        agent = self.core_env.agents[self.selected_index]
        agent.velocity[:] = 0.0
        self.update_debug_views(recompute_obs=True)
        self.set_status(f"Stopped {agent.role}[{agent.agent_id}]")

    def on_canvas_press(self, event: tk.Event) -> None:
        if self.core_env is None:
            return
        idx = self._find_nearest_agent(event.x, event.y)
        if idx is None:
            return
        self.selected_index = idx
        self.dragging = True
        self.agent_listbox.selection_clear(0, tk.END)
        self.agent_listbox.selection_set(idx)
        self.agent_listbox.activate(idx)
        self.update_debug_views(recompute_obs=False)

    def on_canvas_drag(self, event: tk.Event) -> None:
        if self.core_env is None or not self.dragging or self.selected_index is None:
            return
        agent = self.core_env.agents[self.selected_index]
        x, y = self._canvas_to_world(event.x, event.y)
        target = np.array([x, y], dtype=np.float32)
        delta = target - np.asarray(agent.position, dtype=np.float32)
        dt = max(float(self.core_env.dt), 1e-6)
        max_speed = max(float(agent.max_speed), 1e-6)
        # Convert drag displacement to the closest normalized action for one env step.
        action = delta / (dt * max_speed)
        self._step_with_selected_action(action, "Drag-step:")

    def on_canvas_release(self, _event: tk.Event) -> None:
        self.dragging = False

    def _refresh_agent_list(self) -> None:
        self.agent_listbox.delete(0, tk.END)
        if self.core_env is None:
            return
        for idx, agent in enumerate(self.core_env.agents):
            speed = float(np.linalg.norm(agent.velocity))
            self.agent_listbox.insert(
                tk.END,
                f"{idx:02d} | {agent.role}[{agent.agent_id}] alive={agent.alive} speed={speed:.2f}",
            )
        if self.selected_index is not None and 0 <= self.selected_index < self.core_env.agent_num:
            self.agent_listbox.selection_set(self.selected_index)
        self._rebuild_reward_agent_visibility_controls()

    def _rebuild_reward_agent_visibility_controls(self) -> None:
        if self.core_env is None or not hasattr(self, "reward_agent_checks_frame"):
            return
        count = int(self.core_env.agent_num)
        old_vals = [v.get() for v in self.reward_agent_visible_vars]
        recreate = len(self.reward_agent_visible_vars) != count
        if not recreate:
            return

        for child in self.reward_agent_checks_frame.winfo_children():
            child.destroy()

        self.reward_agent_visible_vars = []
        for idx, agent in enumerate(self.core_env.agents):
            if idx < len(old_vals):
                default_visible = old_vals[idx]
            else:
                default_visible = self.selected_index is None or idx == self.selected_index
            var = tk.BooleanVar(value=bool(default_visible))
            self.reward_agent_visible_vars.append(var)
            txt = f"{idx:02d}: {agent.role}[{agent.agent_id}]"
            tk.Checkbutton(
                self.reward_agent_checks_frame,
                text=txt,
                variable=var,
                command=self._draw_reward_plots,
                anchor="w",
            ).pack(fill=tk.X)

    def _set_all_reward_agents_visible(self) -> None:
        for var in self.reward_agent_visible_vars:
            var.set(True)
        self._draw_reward_plots()

    def _set_no_reward_agents_visible(self) -> None:
        for var in self.reward_agent_visible_vars:
            var.set(False)
        self._draw_reward_plots()

    def update_debug_views(self, recompute_obs: bool = True) -> None:
        if self.core_env is None:
            return
        team_sees_target = self.core_env._team_sees_target()
        if recompute_obs or not self.obs_cache:
            self.obs_cache = self.core_env._build_obs(team_sees_target)
        self._refresh_agent_list()
        self._draw_scene()
        self._update_render_preview()
        self._draw_reward_plots()
        self._update_selected_info(team_sees_target)
        self._refresh_obs_two_line()

    def _update_selected_info(self, team_sees_target: bool) -> None:
        self.obs_text.delete("1.0", tk.END)
        if self.core_env is None or self.selected_index is None:
            self.info_var.set("No selected agent.")
            return

        agent = self.core_env.agents[self.selected_index]
        pos = agent.position
        vel = agent.velocity
        speed = float(np.linalg.norm(vel))

        self.pos_x_var.set(round(float(pos[0]), 4))
        self.pos_y_var.set(round(float(pos[1]), 4))
        if speed > 1e-8:
            self.heading_var.set(math.degrees(math.atan2(float(vel[1]), float(vel[0]))))
            self.speed_ratio_var.set(np.clip(speed / max(agent.max_speed, 1e-6), 0.0, 1.0))

        reward_map = {}
        reward_keys = sorted(self.reward_term_history.keys())
        for key in reward_keys:
            hist = self.reward_term_history.get(key, [])
            if hist:
                reward_map[key] = float(hist[-1][self.selected_index])
            else:
                reward_map[key] = 0.0

        reward_lines = []
        for key in reward_keys:
            short_name = key.replace("reward_", "")
            reward_lines.append(f"Reward {short_name}: {reward_map[key]:.4f}")
        reward_text = "\n".join(reward_lines) if reward_lines else "Reward: no step data"

        info = (
            f"Role: {agent.role}[{agent.agent_id}]   Alive: {agent.alive}\n"
            f"Global Position: ({pos[0]:.4f}, {pos[1]:.4f})\n"
            f"Global Velocity: ({vel[0]:.4f}, {vel[1]:.4f})\n"
            f"Speed: {speed:.4f} / Max: {agent.max_speed:.4f}\n"
            f"{reward_text}\n"
            f"shared_target_pos: ({self.core_env.shared_target_pos[0]:.4f}, {self.core_env.shared_target_pos[1]:.4f})\n"
            f"shared_target_valid: {self.core_env.shared_target_valid}, last_seen_age: {self.core_env.last_seen_age}\n"
            f"team_sees_target: {team_sees_target}\n"
            f"obs_dim: {len(self.obs_cache[self.selected_index]) if self.obs_cache else 0}"
        )
        self.info_var.set(info)

        if self.obs_cache:
            obs = self.obs_cache[self.selected_index]
            self.obs_text.insert(tk.END, self._format_obs_vector(self.selected_index, obs))

    def _refresh_obs_two_line(self) -> None:
        if self.core_env is None or self.selected_index is None or not self.obs_cache:
            self.obs_two_line_var.set("")
            return

        obs = np.asarray(self.obs_cache[self.selected_index], dtype=np.float32).reshape(-1)
        segments = self._split_obs_segments(obs)
        headers = [name for name, _arr in segments]
        values = [" ".join(f"{v:.4f}" for v in arr) for _name, arr in segments]

        widths = [max(len(h), len(v), 18) for h, v in zip(headers, values)]

        def pack_row(items: list[str]) -> str:
            cells = [f" {txt.center(w)} " for txt, w in zip(items, widths)]
            return "|" + "|".join(cells) + "|"

        line1 = pack_row(headers)
        line2 = pack_row(values)
        self.obs_two_line_var.set(f"{line1}\n{line2}")

    def _format_obs_vector(self, obs_idx: int, obs: np.ndarray) -> str:
        if self.core_env is None:
            return np.array2string(obs, precision=4, separator=", ")

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        segments = self._split_obs_segments(obs)

        lines = []
        for name, arr in segments:
            lines.append(f"{name}:")
            lines.append("  " + np.array2string(arr, precision=4, separator=", "))
        return "\n".join(lines)

    def _split_obs_segments(self, obs: np.ndarray) -> list[tuple[str, np.ndarray]]:
        if self.core_env is None:
            return [("obs", np.asarray(obs, dtype=np.float32).reshape(-1))]

        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        own_dim = 4
        neighbor_n = int(getattr(self.core_env, "neighbor_N", 0))
        neighbor_feat_dim = int(getattr(self.core_env, "neighbor_feat_dim", 6))
        target_feat_dim = int(getattr(self.core_env, "target_feat_dim", 6))

        segments: list[tuple[str, np.ndarray]] = []
        start = 0

        own_end = min(start + own_dim, len(flat))
        segments.append(("own_obs", flat[start:own_end]))
        start += own_dim

        for i in range(neighbor_n):
            end = min(start + neighbor_feat_dim, len(flat))
            segments.append((f"neighbor_obs{i + 1}", flat[start:end]))
            start += neighbor_feat_dim

        target_end = min(start + target_feat_dim, len(flat))
        segments.append(("target_obs", flat[start:target_end]))
        start += target_feat_dim

        mem = flat[start:] if start < len(flat) else np.zeros(0, dtype=np.float32)
        segments.append(("share_mem_obs", mem))
        return segments

    def _draw_scene(self) -> None:
        if self.core_env is None:
            return

        self.canvas.delete("all")
        self._draw_grid()
        ws = float(self.core_env.world_size)
        _x0, _y0, _x1, _y1, span = self._canvas_metrics()

        for idx, agent in enumerate(self.core_env.agents):
            x, y = self._world_to_canvas(float(agent.position[0]), float(agent.position[1]))
            is_selected = idx == self.selected_index

            if agent.role == "hunter":
                color = "#1F77B4"
                radius = float(self.core_env.hunter_perception_radius)
            elif agent.role == "target":
                color = "#D62728"
                radius = float(self.core_env.target_perception_radius)
            else:
                color = "#2CA02C"
                radius = float(self.core_env.target_perception_radius)

            pr = (radius / ws) * span
            self.canvas.create_oval(x - pr, y - pr, x + pr, y + pr, outline=color, dash=(2, 2))

            r = 8 if is_selected else 6
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="black")
            self.canvas.create_text(x + 12, y - 12, text=f"{idx}:{agent.role[0].upper()}", fill=color)

            vx, vy = float(agent.velocity[0]), float(agent.velocity[1])
            self.canvas.create_line(
                x,
                y,
                x + vx,
                y - vy,
                arrow=tk.LAST,
                fill=color,
                width=2 if is_selected else 1,
            )

    def _update_render_preview(self) -> None:
        if self.env_wrapper is None:
            return
        img = self.env_wrapper.render(mode="rgb_array")
        if img is None:
            return

        arr = np.asarray(img, dtype=np.uint8)
        max_w = max(120, int(self.render_label.winfo_width()) - 4)
        max_h = max(120, int(self.render_label.winfo_height()) - 4)
        h, w = arr.shape[:2]
        factor_w = int(np.ceil(w / max_w))
        factor_h = int(np.ceil(h / max_h))
        factor = max(1, factor_w, factor_h)
        if factor > 1:
            arr = arr[::factor, ::factor, :]
        photo = self._rgb_to_photoimage(arr)

        self.render_photo = photo
        self.render_label.configure(image=self.render_photo)

    def _draw_reward_plots(self) -> None:
        if self.core_env is None:
            return
        if not self.reward_term_history:
            for cv, title in (
                (self.reward_canvas_inst, "Instant Reward"),
                (self.reward_canvas_cum, "Cumulative Reward"),
            ):
                cv.delete("all")
                w = max(int(cv.winfo_width()), 120)
                h = max(int(cv.winfo_height()), 90)
                cv.create_text(w // 2, 16, text=title, fill="#333333")
                cv.create_text(w // 2, h // 2, text="No step data yet")
            return

        visible_agents = [
            idx for idx, var in enumerate(self.reward_agent_visible_vars) if var.get()
        ]
        if not visible_agents and self.selected_index is not None:
            visible_agents = [self.selected_index]
        if not visible_agents and self.core_env.agent_num > 0:
            visible_agents = [0]

        term_keys = sorted(k for k in self.reward_term_history.keys() if k.startswith("reward_"))
        if "reward_total" in term_keys:
            term_keys.remove("reward_total")
            term_keys = ["reward_total"] + term_keys
        dash_styles = [None, (8, 3), (2, 2), (10, 3, 2, 3), (1, 3), (14, 4), (4, 4, 1, 4)]
        inst_series: list[dict] = []
        for key in term_keys:
            hist = self.reward_term_history.get(key, [])
            if not hist:
                continue
            label = key.replace("reward_", "")
            all_arr = np.stack(hist, axis=0)
            for agent_id in visible_agents:
                if agent_id >= all_arr.shape[1]:
                    continue
                inst_series.append(
                    {
                        "term": label,
                        "agent_id": agent_id,
                        "arr": all_arr[:, agent_id],
                        "color": self._get_reward_term_color(label),
                        "dash": dash_styles[agent_id % len(dash_styles)],
                    }
                )

        if not inst_series:
            return

        cum_series = []
        for item in inst_series:
            cum_series.append(
                {
                    "term": item["term"],
                    "agent_id": item["agent_id"],
                    "arr": np.cumsum(item["arr"]),
                    "color": item["color"],
                    "dash": item["dash"],
                }
            )
        self._draw_single_reward_plot(self.reward_canvas_inst, inst_series, "Instant Reward Terms")
        self._draw_single_reward_plot(self.reward_canvas_cum, cum_series, "Cumulative Reward Terms")

    def _on_reward_height_change(self, _event=None) -> None:
        panel_h = int(np.clip(self.reward_panel_height_var.get(), 220, 700))
        if hasattr(self, "reward_panel"):
            self.reward_panel.configure(height=panel_h)
        controls_h = 44
        inner_h = max(120, panel_h - controls_h)
        one_h = max(60, inner_h // 2 - 6)
        if hasattr(self, "reward_canvas_inst"):
            self.reward_canvas_inst.configure(height=one_h)
        if hasattr(self, "reward_canvas_cum"):
            self.reward_canvas_cum.configure(height=one_h)
        if self.core_env is not None:
            self._draw_reward_plots()

    def _on_view_height_change(self, _event=None) -> None:
        row_h = int(np.clip(self.view_row_height_var.get(), 220, 860))
        if hasattr(self, "vis_row"):
            self.vis_row.configure(height=row_h)
        if self.core_env is not None:
            self._draw_scene()
            self._update_render_preview()

    def _draw_single_reward_plot(self, canvas: tk.Canvas, series: list[dict], title: str) -> None:
        canvas.delete("all")
        w = max(int(canvas.winfo_width()), 140)
        h = max(int(canvas.winfo_height()), 100)
        pad_l, pad_r, pad_t, pad_b = 44, 8, 20, 22
        x0, y0 = pad_l, pad_t
        x1, y1 = w - pad_r, h - pad_b
        canvas.create_text((x0 + x1) // 2, 12, text=title, fill="#333333")
        canvas.create_rectangle(x0, y0, x1, y1, outline="#888888")

        first = series[0]["arr"]
        t_len = len(first)
        all_vals = np.concatenate([np.asarray(item["arr"], dtype=np.float32) for item in series])
        y_min = float(np.min(all_vals))
        y_max = float(np.max(all_vals))
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0
        y_margin = 0.08 * (y_max - y_min)
        y_min -= y_margin
        y_max += y_margin

        def map_x(ti: int) -> float:
            if t_len <= 1:
                return (x0 + x1) * 0.5
            return x0 + (x1 - x0) * (ti / (t_len - 1))

        def map_y(rv: float) -> float:
            return y1 - (y1 - y0) * ((rv - y_min) / (y_max - y_min))

        # Grid lines for readability.
        for k in range(1, 5):
            yy = y0 + (y1 - y0) * (k / 5.0)
            canvas.create_line(x0, yy, x1, yy, fill="#EFEFEF")
        x_div = min(8, max(1, t_len - 1))
        for k in range(1, x_div):
            xx = x0 + (x1 - x0) * (k / x_div)
            canvas.create_line(xx, y0, xx, y1, fill="#EFEFEF")

        canvas.create_text(x0 - 4, y0, text=f"{y_max:.2f}", anchor="e", fill="#666666")
        canvas.create_text(x0 - 4, y1, text=f"{y_min:.2f}", anchor="e", fill="#666666")
        canvas.create_text(x0, y1 + 10, text="0", anchor="w", fill="#666666")
        canvas.create_text(x1, y1 + 10, text=str(t_len - 1), anchor="e", fill="#666666")

        for item in series:
            arr = np.asarray(item["arr"], dtype=np.float32)
            pts = []
            for ti in range(t_len):
                pts.extend([map_x(ti), map_y(float(arr[ti]))])
            if len(pts) >= 4:
                kwargs = {"fill": item["color"], "width": 2}
                if item["dash"] is not None:
                    kwargs["dash"] = item["dash"]
                canvas.create_line(*pts, **kwargs)

        # Dual legends: reward-term color and agent line style.
        term_names = []
        for item in series:
            if item["term"] not in term_names:
                term_names.append(item["term"])
        agent_ids = []
        for item in series:
            if item["agent_id"] not in agent_ids:
                agent_ids.append(item["agent_id"])

        legend_y0 = y0 + 4
        canvas.create_text(x0 + 2, legend_y0, text="Color:", anchor="nw", fill="#444444")
        term_col_w = 110
        term_per_row = max(1, int((x1 - (x0 + 52)) // term_col_w))
        for i, term in enumerate(term_names):
            sample = next(s for s in series if s["term"] == term)
            row = i // term_per_row
            col = i % term_per_row
            lx = x0 + 52 + col * term_col_w
            ly = legend_y0 + 6 + row * 12
            canvas.create_line(lx, ly, lx + 12, ly, fill=sample["color"], width=2)
            canvas.create_text(lx + 16, ly, text=term, anchor="w", fill="#444444")

        style_row_y = legend_y0 + 14 + ((max(1, (len(term_names) + term_per_row - 1) // term_per_row) - 1) * 12)
        canvas.create_text(x0 + 2, style_row_y, text="Line:", anchor="nw", fill="#444444")
        agent_col_w = 74
        agent_per_row = max(1, int((x1 - (x0 + 52)) // agent_col_w))
        for i, aid in enumerate(agent_ids):
            sample = next(s for s in series if s["agent_id"] == aid)
            row = i // agent_per_row
            col = i % agent_per_row
            lx = x0 + 52 + col * agent_col_w
            ly = style_row_y + 6 + row * 12
            kwargs = {"fill": "#222222", "width": 2}
            if sample["dash"] is not None:
                kwargs["dash"] = sample["dash"]
            canvas.create_line(lx, ly, lx + 12, ly, **kwargs)
            canvas.create_text(lx + 16, ly, text=f"A{aid}", anchor="w", fill="#444444")

    def _get_reward_term_color(self, term: str) -> str:
        if term in self.reward_term_color_cache:
            return self.reward_term_color_cache[term]
        predefined = {
            "total": "#1F77B4",
            "hunter_base": "#2CA02C",
            "target_base": "#D62728",
            "capture": "#FF7F0E",
            "collision": "#9467BD",
            "speed_penalty": "#8C564B",
        }
        if term in predefined:
            color = predefined[term]
        else:
            palette = [
                "#17BECF",
                "#BCBD22",
                "#E377C2",
                "#7F7F7F",
                "#AEC7E8",
                "#FFBB78",
                "#98DF8A",
                "#FF9896",
                "#C5B0D5",
                "#C49C94",
            ]
            color = palette[len(self.reward_term_color_cache) % len(palette)]
        self.reward_term_color_cache[term] = color
        return color

    @staticmethod
    def _rgb_to_photoimage(rgb: np.ndarray) -> tk.PhotoImage:
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected RGB image with shape (H, W, 3)")
        buf = io.BytesIO()
        imageio.imwrite(buf, arr, format="png")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return tk.PhotoImage(data=b64, format="png")

    def _draw_grid(self) -> None:
        if self.core_env is None:
            return
        x0, y0, x1, y1, _ = self._canvas_metrics()
        ws = float(self.core_env.world_size)
        self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#666666",
            width=2,
        )
        for i in range(1, 10):
            t = i / 10.0
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            self.canvas.create_line(x, y0, x, y1, fill="#ECECEC")
            self.canvas.create_line(x0, y, x1, y, fill="#ECECEC")
        self.canvas.create_text(
            (x0 + x1) / 2.0,
            y1 + 10,
            text=f"world x,y in [-{ws:.1f}, {ws:.1f}]",
            fill="#666666",
        )

    def _canvas_metrics(self) -> tuple[float, float, float, float, float]:
        w = max(120, int(self.canvas.winfo_width()))
        h = max(120, int(self.canvas.winfo_height()))
        span = max(40.0, float(min(w, h) - 2 * self.canvas_pad))
        x0 = (w - span) / 2.0
        y0 = (h - span) / 2.0
        x1 = x0 + span
        y1 = y0 + span
        return x0, y0, x1, y1, span

    def _world_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        ws = max(float(self.core_env.world_size), 1e-6) if self.core_env is not None else 1.0
        x0, y0, _x1, _y1, span = self._canvas_metrics()
        cx = x0 + ((x + ws) / (2 * ws)) * span
        cy = y0 + span - ((y + ws) / (2 * ws)) * span
        return cx, cy

    def _canvas_to_world(self, cx: float, cy: float) -> tuple[float, float]:
        ws = max(float(self.core_env.world_size), 1e-6) if self.core_env is not None else 1.0
        x0, y0, _x1, _y1, span = self._canvas_metrics()
        x = ((cx - x0) / span) * (2 * ws) - ws
        y = ((y0 + span - cy) / span) * (2 * ws) - ws
        return float(np.clip(x, -ws, ws)), float(np.clip(y, -ws, ws))

    def _find_nearest_agent(self, cx: float, cy: float) -> int | None:
        if self.core_env is None:
            return None
        nearest = None
        best_d2 = float("inf")
        for idx, agent in enumerate(self.core_env.agents):
            ax, ay = self._world_to_canvas(float(agent.position[0]), float(agent.position[1]))
            d2 = (ax - cx) ** 2 + (ay - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                nearest = idx
        if nearest is not None and best_d2 <= 24 ** 2:
            return nearest
        return None

    def set_status(self, msg: str) -> None:
        self.status_var.set(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive observation visualizer for UAV pursuit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/hunter_only.yaml"),
        help="Config file path used as initialization base.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = ObsVisApp(root=root, config_path=args.config)
    app.set_status("Ready.")
    root.mainloop()


if __name__ == "__main__":
    main()
