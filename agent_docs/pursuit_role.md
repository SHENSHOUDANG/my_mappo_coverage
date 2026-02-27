# Multi-UAV Pursuit Environment (Current Implementation)

本文档描述当前代码真实实现（重点对应 `train/train.py` 与 `envs/env_uav_pursuit.py`）。

## 1. 任务定义
- 环境为二维正方形区域：`[-world_size, world_size] x [-world_size, world_size]`。
- 当前版本为 **hunter-only**：
  - Pursuit 组仅包含 `num_hunters` 个 Hunter。
  - 仅 1 个 Target。
  - `num_explorers` 必须为 0（否则环境直接报错）。
- 目标：Hunter 在满足连续步条件下捕获 Target。

## 2. Agent 与控制
- Hunter: `policy_type=learn`，动作来自 MAPPO/RMAPPO。
- Target: `policy_type` 支持 `learn` / `random` / `patrol`。
- 控制模式：
  - `velocity`: 动作为目标速度方向/幅值（归一化到 `[-1,1]`），映射到最大速度。
  - `acceleration`: 动作为归一化加速度，积分更新速度。
- 转向限制：`velocity` 模式下可用 `max_turn_angle` + `min_turn_limit_velo` 限制单步转角。

## 3. 动作与观测
### 3.1 动作空间
- 每个 agent 动作：`Box(low=-1, high=1, shape=(2,))`。
- 实际物理速度或加速度由各角色 `max_velo/max_acc` 决定。

### 3.2 观测空间
单 agent 观测：
- `own_obs`: `[x, y, vx, vy] / world_size`。
- `neighbor_obs`: 最近 `neighbor_N` 同阵营邻居槽位，每槽 6 维：
  `[dx, dy, dvx, dvy, d, valid]`。
- `target_obs`: 6 维 `[dx, dy, dvx, dvy, d, visible]`。
- `memory_obs`: 5 维共享目标记忆 `[dx, dy, dvx, dvy, age_norm]`。

总维度：
`obs_dim = 4 + neighbor_N * 6 + 6 + 5`

说明：
- Hunter 在“本步团队不可见 target 且共享记忆有效”时，`memory_obs` 才非零。
- Target 的 `memory_obs` 永远为零。
- 不活跃 hunter（由任务规格裁剪）观测为全零，并在 info 中标记 `active_agent=false`。

## 4. 可见性与共享记忆
- `team_sees_target=True` 条件：任一 active 且存活 Hunter 在感知半径内（或感知半径 < 0）。
- 在前 `target_pos_init_guidance_step` 步，即使不可见，也会更新共享 target 真值。
- 若之后不可见：
  - `target_pos_guidance=false`：共享记忆失效。
  - `target_pos_guidance=true`：共享记忆使用带噪声估计（位置与速度噪声）。

## 5. 捕获、碰撞与终止
### 5.1 捕获
- 对每个 Hunter 维护 `capture_counter[i]`：当 `dist(h_i, target) <= capture_dis` 时计数加一，否则清零。
- 任一 Hunter 计数达到 `capture_step` 即捕获成功。

### 5.2 碰撞
- 边界风险：所有 active agent 都计算到最近边界距离，并施加安全区惩罚。
- 硬边界碰撞：到边界距离 `<= collision_dis` 时该 agent 失活。
- 两两碰撞：仅在非 target agent 间判定（当前即 hunter-hunter）。
- Target 不参与两两碰撞判定，但会参与边界碰撞判定。
- Target 发生边界碰撞时追加 `target_collision_penalty`。

### 5.3 Episode 终止
满足任一条件终止：
- 达到 `episode_length`。
- 成功捕获 target。
- target 边界碰撞。
- 全部 active hunter 死亡。

## 6. 奖励实现
总奖励由以下项线性相加：
- `hunter_base_reward`
- `target_base_reward`
- `hunter_streak_reward`
- `target_streak_reward`
- `capture_reward`
- `collision_reward`
- `speed_penalty_reward`

关键机制：
- Hunter 距离目标越近，基础奖励越大；超出捕获半径时给负向远距惩罚。
- Target 使用与 Hunter 镜像的基础奖励。
- Hunter 连续处于捕获半径内可获得 streak 奖励（有上限 `base_streak_cap`）。
- 捕获瞬间 Hunter 得到 `hunter_capture_reward`，Target 受到 `target_captured_penalty`。
- 碰撞惩罚包含安全区线性惩罚与硬碰撞惩罚，并受 `collision_penalty_cap` 截断。
- 所有 agent 有归一化速度惩罚：`-speed_penalty * (||v|| / max_speed)`。

## 7. reset 模式与任务规格
环境 `reset(mode, task_spec)` 支持：
- `initial`: 训练常规 reset。
- `recover`: 用当前任务 seed 复原（评估固定任务时保证可重复）。
- `regen`: 生成或应用新任务规格。

任务规格字段（`task_spec`）支持：
- `num_hunters`
- `world_size`
- `target_policy_source`
- `target_patrol_path`
- `target_patrol_names`
- `target_route_id`
- `seed`

## 8. train/train.py 训练与评估流程
`train/train.py` 的核心实现：
1. 读取并合并配置（`defaults.yaml` + 用户 yaml）。
2. 校验 `algorithm_name` 与 RNN 开关一致性（`mappo` vs `rmappo`）。
3. 创建训练环境：`DummyVecEnv + ContinuousActionEnv`，`auto_reset=initial`。
4. 创建评估环境：固定任务 + `auto_reset=recover`。
5. 若 `eval.fixed_tasks_file` 提供任务，自动将 `n_eval_rollout_threads` 对齐任务数。
6. 推断评估任务最大 `num_hunters`，防止评估配置上限不足导致 active hunter 被截断。
7. 创建 `RoleBasedRunner` 执行训练或 `--time_stat`。

## 9. Domain Randomization 与 Fixed Eval
- 训练域随机化来源：`domain_randomization.train_split`。
  - 当 `enable=true` 时，`initial reset` 按 `regen_interval_episode` 与 `regen_prob` 触发任务重采样。
- 评估固定任务来源：
  - `eval.fixed_tasks`（内联）或
  - `eval.fixed_tasks_file`（yaml/json，推荐）。
- 当 `target_policy_source=learn` 时，会额外创建 `target_learn` 评估桶并记录独立指标。

## 10. 训练产物
默认输出目录：
`results/<env_name>/<algorithm_name>/<experiment_name>/run*/`

主要文件：
- `models/actor_*.pt`, `models/critic_*.pt`
- `models/best_eval_reward|capture_rate|capture_steps/`
- `logs/`（TensorBoard）与 `logs/summary.json`
- `gifs/`（训练/评估 GIF）
- `log.csv`, `eval.csv`, `time_stat.csv(可选)`

## 11. 多目标推理扩展（Inference-only MVP）
新增 `envs/env_uav_multi_infer.py` + `train/infer_multi.py`，用于非训练场景的全流程推理：
- 支持 `N hunters + K explorers + M targets`。
- Explorer 执行分片弓字航线搜索；发现目标后切换 TRACK，并触发 hunter 分配。
- Hunter 接收 explorer 共享的目标状态（位置/速度/age）并执行追捕。
- Target 支持 `random/patrol/learn/static` 混合采样。
- 终止条件：达到 `max_steps` 或全部 target 被捕获。

说明：
- 该扩展当前为推理MVP，不替代 `env_uav_pursuit.py` 训练主环境。
- `--hunter_actor/--target_actor` 为可选；不提供时会使用启发式/零动作回退逻辑。
