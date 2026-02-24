#!/usr/bin/env python3
"""Visual patrol route editor for normalized [0, 1] waypoint JSON files."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


DEFAULT_META = {"version": 1, "coords": "normalized_0_1"}


class PatrolEditorApp:
    def __init__(self, root: tk.Tk, json_path: Path) -> None:
        self.root = root
        self.json_path = json_path
        self.data = {"meta": dict(DEFAULT_META), "routes": []}
        self.selected_index: int | None = None
        self.draw_mode: str | None = None
        self.pending_waypoints: list[list[float]] = []
        self.pending_name: str = ""
        self.pending_edit_index: int | None = None
        self.route_visible: list[bool] = []
        self.route_visible_vars: list[tk.BooleanVar] = []
        self.route_row_frames: list[tk.Frame] = []
        self.route_row_labels: list[tk.Label] = []
        self.select_all_var = tk.BooleanVar(value=True)
        self._updating_select_all = False
        self.dirty = False

        self.canvas_size = 680
        self.canvas_pad = 24

        self._build_ui()
        self.load_file(self.json_path, silent=False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        self.root.title("Patrol Route Editor")
        self.root.geometry("1120x760")

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main, padx=8, pady=8)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(main, padx=8, pady=8)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(
            left,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
            highlightthickness=1,
            highlightbackground="#888888",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.status_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.status_var, anchor="w").pack(fill=tk.X, pady=(8, 0))

        tk.Label(right, text="Routes", font=("Arial", 12, "bold")).pack(anchor="w")
        self.route_panel = tk.Frame(right, width=300, height=340, bd=1, relief=tk.SOLID)
        self.route_panel.pack(fill=tk.X, pady=(4, 8))
        self.route_panel.pack_propagate(False)

        self.select_all_checkbox = tk.Checkbutton(
            self.route_panel,
            text="Show All",
            variable=self.select_all_var,
            command=self.on_toggle_all_visible,
            anchor="w",
            padx=4,
            pady=2,
        )
        self.select_all_checkbox.pack(fill=tk.X)

        self.route_scroll_frame = tk.Frame(self.route_panel)
        self.route_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 2))

        self.route_list_canvas = tk.Canvas(
            self.route_scroll_frame,
            bg=self.root.cget("bg"),
            highlightthickness=0,
            bd=0,
        )
        self.route_scrollbar = tk.Scrollbar(
            self.route_scroll_frame,
            orient=tk.VERTICAL,
            command=self.route_list_canvas.yview,
        )
        self.route_list_canvas.configure(yscrollcommand=self.route_scrollbar.set)

        self.route_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.route_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.route_rows_container = tk.Frame(self.route_list_canvas, bg=self.root.cget("bg"))
        self.route_rows_window_id = self.route_list_canvas.create_window(
            (0, 0),
            window=self.route_rows_container,
            anchor="nw",
        )
        self.route_rows_container.bind("<Configure>", self._on_route_rows_configure)
        self.route_list_canvas.bind("<Configure>", self._on_route_canvas_configure)
        self._bind_route_scroll_events(self.route_list_canvas)
        self._bind_route_scroll_events(self.route_rows_container)

        btn_frame = tk.Frame(right)
        btn_frame.pack(fill=tk.X)

        self._add_button(btn_frame, "Load JSON", self.choose_file)
        self._add_button(btn_frame, "Save", self.save_file)
        self._add_button(btn_frame, "Add (Manual Draw)", self.start_add_manual)
        self._add_button(btn_frame, "Add (Random)", self.add_random_route)
        self._add_button(btn_frame, "Edit (Redraw Selected)", self.start_edit_selected)
        self._add_button(btn_frame, "Rename", self.rename_selected)
        self._add_button(btn_frame, "Delete", self.delete_selected)
        self._add_button(btn_frame, "Undo Last Point", self.undo_last_point)
        self._add_button(btn_frame, "Clear Current Draw", self.clear_pending_draw)
        self._add_button(btn_frame, "Finish Draw", self.finish_draw)
        self._add_button(btn_frame, "Cancel Draw", self.cancel_draw)

        self.file_var = tk.StringVar(value=f"File: {self.json_path}")
        tk.Label(right, textvariable=self.file_var, justify=tk.LEFT, wraplength=280).pack(
            anchor="w", pady=(10, 0)
        )

    def _add_button(self, parent: tk.Widget, text: str, command) -> None:
        tk.Button(parent, text=text, command=command, width=30).pack(fill=tk.X, pady=2)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _x_to_canvas(self, x: float) -> float:
        span = self.canvas_size - 2 * self.canvas_pad
        return self.canvas_pad + self._clamp01(x) * span

    def _y_to_canvas(self, y: float) -> float:
        span = self.canvas_size - 2 * self.canvas_pad
        return self.canvas_size - self.canvas_pad - self._clamp01(y) * span

    def _canvas_to_norm(self, px: int, py: int) -> tuple[float, float]:
        span = self.canvas_size - 2 * self.canvas_pad
        nx = (px - self.canvas_pad) / span
        ny = (self.canvas_size - self.canvas_pad - py) / span
        return self._clamp01(nx), self._clamp01(ny)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def mark_dirty(self) -> None:
        self.dirty = True
        self.root.title("Patrol Route Editor *")

    def clear_dirty(self) -> None:
        self.dirty = False
        self.root.title("Patrol Route Editor")

    def normalize_loaded_data(self, raw: dict) -> dict:
        meta = raw.get("meta")
        if not isinstance(meta, dict):
            meta = dict(DEFAULT_META)
        routes = raw.get("routes")
        if not isinstance(routes, list):
            routes = []

        normalized_routes = []
        for i, route in enumerate(routes):
            if not isinstance(route, dict):
                continue
            name = route.get("name")
            if not isinstance(name, str) or not name.strip():
                name = f"route_{i+1:03d}"

            points = []
            for pt in route.get("waypoints", []):
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    try:
                        x = float(pt[0])
                        y = float(pt[1])
                    except (TypeError, ValueError):
                        continue
                    points.append([self._clamp01(x), self._clamp01(y)])

            normalized_routes.append({"name": name, "waypoints": points})

        return {"meta": meta, "routes": normalized_routes}

    def load_file(self, path: Path, silent: bool = True) -> None:
        if self.dirty and not silent:
            if not messagebox.askyesno("Unsaved Changes", "You have unsaved changes. Continue loading?"):
                return

        if not path.exists():
            self.data = {"meta": dict(DEFAULT_META), "routes": []}
            self.json_path = path
            self.route_visible = []
            self.file_var.set(f"File: {self.json_path}")
            self.refresh_route_list()
            self.refresh_canvas()
            self.clear_dirty()
            self.set_status("File does not exist yet. Started with empty routes.")
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load JSON:\n{exc}")
            return

        self.data = self.normalize_loaded_data(raw if isinstance(raw, dict) else {})
        self.json_path = path
        self.file_var.set(f"File: {self.json_path}")
        self.selected_index = None
        self.route_visible = [True for _ in self.data["routes"]]
        self.pending_waypoints = []
        self.draw_mode = None
        self.pending_edit_index = None
        self.pending_name = ""
        self.refresh_route_list()
        self.refresh_canvas()
        self.clear_dirty()
        self.set_status(f"Loaded {len(self.data['routes'])} routes.")

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select patrol route JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.load_file(Path(path), silent=False)

    def save_file(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.json_path.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            messagebox.showerror("Save Error", f"Failed to save JSON:\n{exc}")
            return
        self.clear_dirty()
        self.set_status(f"Saved to {self.json_path}")

    def refresh_route_list(self) -> None:
        self._ensure_visibility_length()
        for child in self.route_rows_container.winfo_children():
            child.destroy()

        self.route_visible_vars = []
        self.route_row_frames = []
        self.route_row_labels = []

        if self.selected_index is not None and not (0 <= self.selected_index < len(self.data["routes"])):
            self.selected_index = None

        for idx, route in enumerate(self.data["routes"]):
            row = tk.Frame(self.route_rows_container, bg=self.root.cget("bg"))
            row.pack(fill=tk.X, pady=1)

            visible_var = tk.BooleanVar(value=self.route_visible[idx])
            chk = tk.Checkbutton(
                row,
                variable=visible_var,
                command=lambda i=idx: self._on_toggle_visible(i),
                padx=2,
                pady=0,
            )
            chk.pack(side=tk.LEFT)

            npts = len(route.get("waypoints", []))
            lbl = tk.Label(row, text=f"{idx + 1:02d}. {route['name']} ({npts} pts)", anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))
            lbl.bind("<Button-1>", lambda _e, i=idx: self.select_route_by_index(i, ensure_visible=True))
            row.bind("<Button-1>", lambda _e, i=idx: self.select_route_by_index(i, ensure_visible=True))
            self._bind_route_scroll_events(row)
            self._bind_route_scroll_events(chk)
            self._bind_route_scroll_events(lbl)

            self.route_visible_vars.append(visible_var)
            self.route_row_frames.append(row)
            self.route_row_labels.append(lbl)

        self._sync_select_all_checkbox()
        self._style_route_rows()
        self.root.after_idle(self._update_route_scrollregion)

    def _ensure_visibility_length(self) -> None:
        route_count = len(self.data["routes"])
        if len(self.route_visible) < route_count:
            self.route_visible.extend([True] * (route_count - len(self.route_visible)))
        elif len(self.route_visible) > route_count:
            self.route_visible = self.route_visible[:route_count]

    def _style_route_rows(self) -> None:
        for idx, row in enumerate(self.route_row_frames):
            is_selected = self.selected_index == idx
            bg = "#D9ECFF" if is_selected else self.root.cget("bg")
            row.configure(bg=bg)
            self.route_row_labels[idx].configure(bg=bg, fg="#1F2937" if is_selected else "#222222")

    def _on_toggle_visible(self, idx: int) -> None:
        if not (0 <= idx < len(self.route_visible_vars)):
            return
        self.route_visible[idx] = bool(self.route_visible_vars[idx].get())
        self._sync_select_all_checkbox()
        self.select_route_by_index(idx, ensure_visible=False)
        self.refresh_canvas()

    def _sync_select_all_checkbox(self) -> None:
        all_visible = bool(self.route_visible) and all(self.route_visible)
        self._updating_select_all = True
        self.select_all_var.set(all_visible)
        self._updating_select_all = False

    def on_toggle_all_visible(self) -> None:
        if self._updating_select_all:
            return
        visible = bool(self.select_all_var.get())
        self.route_visible = [visible for _ in self.data["routes"]]
        for idx, var in enumerate(self.route_visible_vars):
            if idx < len(self.route_visible):
                var.set(self.route_visible[idx])
        self.refresh_canvas()

    def select_route_by_index(self, idx: int, ensure_visible: bool = False) -> None:
        if self.draw_mode:
            return
        if not (0 <= idx < len(self.data["routes"])):
            return
        if ensure_visible and idx < len(self.route_visible) and not self.route_visible[idx]:
            self.route_visible[idx] = True
            if idx < len(self.route_visible_vars):
                self.route_visible_vars[idx].set(True)
        self.selected_index = idx
        route = self.data["routes"][idx]
        self._style_route_rows()
        self.refresh_canvas()
        self.set_status(f"Selected route '{route['name']}' with {len(route['waypoints'])} points.")

    def _on_route_rows_configure(self, _event=None) -> None:
        self._update_route_scrollregion()

    def _on_route_canvas_configure(self, event: tk.Event) -> None:
        self.route_list_canvas.itemconfigure(self.route_rows_window_id, width=event.width)
        self._update_route_scrollregion()

    def _update_route_scrollregion(self) -> None:
        bbox = self.route_list_canvas.bbox("all")
        if bbox is not None:
            self.route_list_canvas.configure(scrollregion=bbox)

    def _bind_route_scroll_events(self, widget: tk.Widget) -> None:
        widget.bind("<MouseWheel>", self._on_route_mousewheel)
        widget.bind("<Button-4>", self._on_route_mousewheel)
        widget.bind("<Button-5>", self._on_route_mousewheel)

    def _on_route_mousewheel(self, event: tk.Event):
        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            delta = int(getattr(event, "delta", 0))
            if delta == 0:
                return "break"
            step = -1 if delta > 0 else 1
        self.route_list_canvas.yview_scroll(step, "units")
        return "break"

    def refresh_canvas(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()

        selected = self.selected_index
        for idx, route in enumerate(self.data["routes"]):
            if idx >= len(self.route_visible) or not self.route_visible[idx]:
                continue
            waypoints = route.get("waypoints", [])
            if not waypoints:
                continue
            is_selected = selected is not None and idx == selected
            color = "#2E86DE" if is_selected else "#B0B0B0"
            width = 3 if is_selected else 1
            start_color = "#F59E0B" if is_selected else "#E67E22"
            self._draw_waypoints(
                waypoints,
                color=color,
                width=width,
                show_index=is_selected,
                start_color=start_color,
            )

        if self.draw_mode:
            self._draw_waypoints(
                self.pending_waypoints,
                color="#E74C3C",
                width=2,
                show_index=True,
                start_color="#2ECC71",
            )
            self.canvas.create_text(
                12,
                12,
                anchor="nw",
                text=f"DRAW MODE: {self.draw_mode}. Left-click to add points.",
                fill="#C0392B",
                font=("Arial", 11, "bold"),
            )

    def _draw_grid(self) -> None:
        for i in range(11):
            t = i / 10.0
            x = self._x_to_canvas(t)
            y = self._y_to_canvas(t)
            color = "#DADADA" if i not in (0, 10) else "#888888"
            self.canvas.create_line(x, self.canvas_pad, x, self.canvas_size - self.canvas_pad, fill=color)
            self.canvas.create_line(self.canvas_pad, y, self.canvas_size - self.canvas_pad, y, fill=color)

        self.canvas.create_text(
            self.canvas_size // 2,
            self.canvas_size - 6,
            text="x in [0, 1]",
            fill="#666666",
            font=("Arial", 9),
        )
        self.canvas.create_text(
            8,
            self.canvas_size // 2,
            text="y",
            fill="#666666",
            angle=90,
            font=("Arial", 9),
        )

    def _draw_waypoints(
        self,
        waypoints: list[list[float]],
        color: str,
        width: int,
        show_index: bool,
        start_color: str | None = None,
    ) -> None:
        if not waypoints:
            return
        coords = []
        for x, y in waypoints:
            coords.extend([self._x_to_canvas(x), self._y_to_canvas(y)])

        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill=color, width=width)

        for i, (x, y) in enumerate(waypoints):
            cx = self._x_to_canvas(x)
            cy = self._y_to_canvas(y)
            is_start = i == 0 and start_color is not None
            r = 5 if is_start else (4 if show_index else 3)
            point_color = start_color if is_start else color
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=point_color, outline="")
            if show_index:
                self.canvas.create_text(cx + 8, cy - 8, text=str(i + 1), fill=point_color, font=("Arial", 8))

    def on_canvas_click(self, event: tk.Event) -> None:
        if not self.draw_mode:
            return
        x, y = self._canvas_to_norm(event.x, event.y)
        self.pending_waypoints.append([x, y])
        self.set_status(f"Added point #{len(self.pending_waypoints)}: ({x:.3f}, {y:.3f})")
        self.refresh_canvas()

    def ensure_unique_name(self, name: str, ignore_index: int | None = None) -> bool:
        name = name.strip()
        if not name:
            return False
        for idx, route in enumerate(self.data["routes"]):
            if ignore_index is not None and idx == ignore_index:
                continue
            if route.get("name") == name:
                return False
        return True

    def ask_new_name(self, title: str, initial: str = "") -> str | None:
        name = simpledialog.askstring(title, "Route name:", initialvalue=initial, parent=self.root)
        if name is None:
            return None
        name = name.strip()
        if not name:
            messagebox.showwarning("Invalid Name", "Route name cannot be empty.")
            return None
        return name

    def start_add_manual(self) -> None:
        if self.draw_mode:
            messagebox.showinfo("Draw In Progress", "Finish or cancel current drawing first.")
            return

        name = self.ask_new_name("Add Route")
        if name is None:
            return
        if not self.ensure_unique_name(name):
            messagebox.showwarning("Duplicate Name", f"Route '{name}' already exists.")
            return

        self.draw_mode = "add"
        self.pending_name = name
        self.pending_edit_index = None
        self.pending_waypoints = []
        self.set_status(f"Manual draw started for '{name}'. Left-click on canvas to add points.")
        self.refresh_canvas()

    def _generate_random_route(self, n_points: int) -> list[list[float]]:
        cx = random.uniform(0.25, 0.75)
        cy = random.uniform(0.25, 0.75)
        base_r = random.uniform(0.12, 0.32)
        angles = sorted(random.uniform(0.0, 2 * math.pi) for _ in range(n_points))
        points = []
        for a in angles:
            r = base_r * random.uniform(0.65, 1.35)
            x = self._clamp01(cx + r * math.cos(a))
            y = self._clamp01(cy + r * math.sin(a))
            points.append([x, y])
        if random.random() < 0.5:
            points.reverse()
        return points

    def add_random_route(self) -> None:
        if self.draw_mode:
            messagebox.showinfo("Draw In Progress", "Finish or cancel current drawing first.")
            return

        name = self.ask_new_name("Add Random Route")
        if name is None:
            return
        if not self.ensure_unique_name(name):
            messagebox.showwarning("Duplicate Name", f"Route '{name}' already exists.")
            return

        n_points = simpledialog.askinteger(
            "Random Route",
            "Number of waypoints:",
            initialvalue=12,
            minvalue=3,
            maxvalue=128,
            parent=self.root,
        )
        if n_points is None:
            return

        route = {"name": name, "waypoints": self._generate_random_route(int(n_points))}
        self.data["routes"].append(route)
        self.selected_index = len(self.data["routes"]) - 1
        self.refresh_route_list()
        self.refresh_canvas()
        self.mark_dirty()
        self.set_status(f"Added random route '{name}' with {n_points} points.")

    def start_edit_selected(self) -> None:
        if self.draw_mode:
            messagebox.showinfo("Draw In Progress", "Finish or cancel current drawing first.")
            return
        if self.selected_index is None:
            messagebox.showwarning("No Selection", "Please select a route to edit.")
            return

        route = self.data["routes"][self.selected_index]
        self.draw_mode = "edit"
        self.pending_edit_index = self.selected_index
        self.pending_name = route["name"]
        self.pending_waypoints = []
        self.set_status(
            f"Redrawing route '{route['name']}'. Left-click to add new points, then click Finish Draw."
        )
        self.refresh_canvas()

    def delete_selected(self) -> None:
        if self.draw_mode:
            messagebox.showinfo("Draw In Progress", "Finish or cancel current drawing first.")
            return
        if self.selected_index is None:
            messagebox.showwarning("No Selection", "Please select a route to delete.")
            return

        route = self.data["routes"][self.selected_index]
        if not messagebox.askyesno("Confirm Delete", f"Delete route '{route['name']}'?"):
            return

        self.data["routes"].pop(self.selected_index)
        if self.selected_index >= len(self.data["routes"]):
            self.selected_index = len(self.data["routes"]) - 1 if self.data["routes"] else None
        self.refresh_route_list()
        self.refresh_canvas()
        self.mark_dirty()
        self.set_status(f"Deleted route '{route['name']}'.")

    def rename_selected(self) -> None:
        if self.draw_mode:
            messagebox.showinfo("Draw In Progress", "Finish or cancel current drawing first.")
            return
        if self.selected_index is None:
            messagebox.showwarning("No Selection", "Please select a route to rename.")
            return

        route = self.data["routes"][self.selected_index]
        new_name = self.ask_new_name("Rename Route", initial=route["name"])
        if new_name is None:
            return
        if not self.ensure_unique_name(new_name, ignore_index=self.selected_index):
            messagebox.showwarning("Duplicate Name", f"Route '{new_name}' already exists.")
            return

        old_name = route["name"]
        route["name"] = new_name
        self.refresh_route_list()
        self.refresh_canvas()
        self.mark_dirty()
        self.set_status(f"Renamed '{old_name}' to '{new_name}'.")

    def undo_last_point(self) -> None:
        if not self.draw_mode:
            return
        if not self.pending_waypoints:
            return
        self.pending_waypoints.pop()
        self.refresh_canvas()
        self.set_status(f"Removed last point. Remaining: {len(self.pending_waypoints)}")

    def clear_pending_draw(self) -> None:
        if not self.draw_mode:
            return
        self.pending_waypoints = []
        self.refresh_canvas()
        self.set_status("Cleared current drawing points.")

    def finish_draw(self) -> None:
        if not self.draw_mode:
            messagebox.showinfo("No Draw Session", "No active drawing to finish.")
            return
        if len(self.pending_waypoints) < 2:
            messagebox.showwarning("Too Few Points", "Please draw at least 2 points.")
            return

        if self.draw_mode == "add":
            self.data["routes"].append(
                {"name": self.pending_name, "waypoints": [list(pt) for pt in self.pending_waypoints]}
            )
            self.selected_index = len(self.data["routes"]) - 1
            self.set_status(
                f"Added route '{self.pending_name}' with {len(self.pending_waypoints)} waypoints."
            )
        elif self.draw_mode == "edit":
            if self.pending_edit_index is None or not (0 <= self.pending_edit_index < len(self.data["routes"])):
                messagebox.showerror("Edit Error", "Selected route is no longer valid.")
                self.cancel_draw()
                return
            self.data["routes"][self.pending_edit_index]["waypoints"] = [
                list(pt) for pt in self.pending_waypoints
            ]
            self.selected_index = self.pending_edit_index
            self.set_status(
                f"Updated route '{self.pending_name}' with {len(self.pending_waypoints)} waypoints."
            )

        self.draw_mode = None
        self.pending_name = ""
        self.pending_edit_index = None
        self.pending_waypoints = []
        self.refresh_route_list()
        self.refresh_canvas()
        self.mark_dirty()

    def cancel_draw(self) -> None:
        if not self.draw_mode:
            return
        mode = self.draw_mode
        self.draw_mode = None
        self.pending_name = ""
        self.pending_edit_index = None
        self.pending_waypoints = []
        self.refresh_canvas()
        self.set_status(f"Canceled {mode} drawing.")

    def on_close(self) -> None:
        if self.dirty:
            keep = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Save before exit?",
            )
            if keep is None:
                return
            if keep:
                self.save_file()
                if self.dirty:
                    return
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual editor for patrol route JSON files.")
    parser.add_argument(
        "json",
        default="datasets/patrol_routes.json",
        help="Path to patrol route JSON file (default: datasets/patrol_routes.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = PatrolEditorApp(root=root, json_path=Path(args.json))
    app.set_status("Ready.")
    root.mainloop()


if __name__ == "__main__":
    main()
