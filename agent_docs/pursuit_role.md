# Multi-UAV Pursuit Environment
该环境为多无人机目标追捕任务，其中无人机被分为以下三类：
- Target: 被追捕目标，其可能在环境中随机游走/巡逻，或使用学习的Policy，在指定环境区域内进行逃脱
- Hunter: 运动速度极快，但是感知范围受限，当Target处于其一定距离范围内一定步长后，可以成功抓捕目标。
- Explorer: 运动速度中等，有较大的感知范围，需要在任务中辅助Hunter捕获环境中的被追捕目标，一方面探查到其位置，另一方面尽可能缩小Target的逃脱空间


其中Hunter和Explorer被归为Pursuit组，而Target则属于Target组。
Pursuit组在任务初始的`target_pos_init_guidance_step`个step内可以获取到Target的当前初始位置与速度，随后则该位置不可知。当Target出现在Pursuit组的任意一个无人机的感知范围内后，其位置和速度将会被感知到，且共享给Pursuit组内的每个无人机。

在每个任务中，Pursuit组都有1架及以上的无人机（至少一个Hunter），Target组中则只有1架target无人机

## Environment
- **World**: 2D square with bounds `[-world_size, world_size]` and time step `dt`.
- **Agents**: `num_hunters` + `num_explorers` + 1 target (evasion). Roles are fixed by index.
- **Target patrol mode**: if `target_policy_source=patrol`, the target action is overridden to follow waypoints loaded from `target_patrol_path` (with optional named routes).
- **Pursuit sharing**: pursuers fully share their own positions/velocities at all times; target observations are shared across the pursuit team only when any pursuer detects the target.
- **Capture**: When the distance between target and hunter is lower than `capture_dis` in `capture_step` steps, the target will be captured.
- **Collision**: When distance between any two agents is lower than `collision_dis`, the two agents will be destroied (Cannot execute any other actions). If the target is destroied, or all hunters are destroied, the episode end. 
- **Episode end**: capture or `max_steps` or collision. 

## State / Observation Space
Per-agent observation vector:
- `own_obs`: own position `(x, y)` and velocity `(vx, vy)`, normalized by `world_size`.
- `neighbor_obs`: nearest `neighbor_N` teammates (fixed slots) with each slot
  `(dx, dy, dvx, dvy, d, valid)`.
  - `valid=1` means this slot is occupied by a real alive teammate; otherwise all zeros with `valid=0`.
  - If teammate count is less than `neighbor_N`, remaining slots are zero padded.
- `target_obs`:
  - For hunter: `(dx, dy, dvx, dvy, d, visible)` of target.
  - For target: `(dx, dy, dvx, dvy, d, visible)` of nearest alive hunter.
- Pursuit shared target memory (hunter/blocker only): pursuit-observed target position/velocity **in relative coordinates** `(dx, dy, dvx, dvy)`, plus `last_seen_age` (steps since last true sighting).
  - If any pursuer sees the target this step, all pursuers receive the current target position/velocity with `last_seen_age = 0`.
  - If no pursuer sees the target, pursuers receive the last seen position/velocity and `last_seen_age` increments.
  - When the target is not visible, pursuers still receive a noisy relative target position/velocity if `target_pos_guidance` is True, or will be zeroed; once visible, the noise is cleared.
  - The target agent itself receives zeros in this memory slot.

Dimension: `obs_dim = 4 + neighbor_N * 6 + 6 + 5`.

Centralized observation (for shared critics): concatenation of all agents’ observations with dimension `obs_dim * agent_num`.

## Action Space
Continuous 2D action for each agent: `Box(low=-1, high=1, shape=(2,))`.
- Actions are scaled by role-specific max speed and applied as velocity updates.
- Positions are updated with `dt` and clipped to world bounds.

## Rewards
Let `d_i` be hunter `i`'s distance to target, `d_min = min_i d_i`, `R = capture_dis`.

- **Shared base coefficients** (hunter/target use the same set): `base_far_scale`, `base_near_scale`, `base_streak_scale`, `base_streak_cap`.
- **Hunter base reward**:
  - If `d_i <= R`: `+ base_near_scale * (1 - d_i / R)` (inside capture range, closer is better).
  - If `d_i > R`: `- base_far_scale * ((d_i - R) / world_size)` (outside capture range, farther is worse).
- **Hunter in-range streak reward**:
  - Reuse each hunter's capture counter (continuous steps in capture range),
  - `+ base_streak_scale * min(capture_counter_i, base_streak_cap)`.
- **Target base reward**:
  - If `d_min <= R`: `- base_near_scale * (1 - d_min / R)`.
  - If `d_min > R`: `+ base_far_scale * ((d_min - R) / world_size)`.
- **Target streak penalty**:
  - Based on hunters' in-range counters (mean over hunters),
  - `- base_streak_scale * mean(min(capture_counter_i, base_streak_cap))`.
- **Capture event reward** (single-step event):
  - Hunters: `+ hunter_capture_reward`.
  - Target: `- target_captured_penalty`.
- **Speed penalty** (normalized linear form):
  - For every agent `a`: `- k_speed * (||v_a|| / max_speed_a)`.
  - (Backward compatibility: if `k_speed` is absent, fallback to `speed_penalty`.)

## Collision Condition
- If any two agents are within `collision_radius`, they are marked as collided.
- If the target collides, the episode ends immediately and capture does not count.
- Non-target collisions do not end the episode, but collided agents are marked done (stop moving).
- Collision penalties are applied once per pair:
  - If the pair is approaching (`dot(v_rel, p_rel) < 0`), both are penalized.
  - Otherwise, the faster agent is penalized.
  - Penalty magnitude: `collision_penalty_k * speed`, then clipped by `collision_penalty_cap` to avoid extreme values.

## TODO List
- Multi-Agent Pursuit-Evasion: 
    - [ ] Datasets in Train/Val/Test: 用预先定义好的配置文件来生成各种Pursuit-Evasion环境，用于模型训练/验证/评估。从而确保不同配置参数下的模型训练结果可以进行公平对比。其配置文件需要包含以下配置
        - seed: 随机种子数，用于确保环境状态随机初始化的结果一致
        - map_size: 整个环境的长 x 宽 （单位为米）
        - Target: 被追捕目标相关配置
          - Target Max Velo: 目标最大运动速度
          - Target Route Type: 
            - Patrol: 固定路线巡逻 （环境中的Target从初始位置开始，沿巡逻路线和最大速度运动）
            - Random: 一定step后，在0～MAX Velo中随机采样速度和方向进行运动（不越过地图边界）
            - Policy: 加载训练好的Policy模型进行运动
            
    - [ ] Env generation from config: 可以通过配置文件来生成对应的任务环境，确保不同的策略模型在相同的状态下开始，以确保结果一致
    - [ ] Metrics: 数据集需要有配套的性能指标用于对比。对于Pursuit-evasion任务，使用
        - Pursuit Rate: 捕获目标的成功率
        - Avg Pursuit Steps: 成功捕获目标需要的平均步数
        - Max Escape Interval: 被追捕目标的最大潜在逃生区间夹角

- Pursuit Agent Configuration
  - [ ] Policy Sharing: `policy_share`为True时，相同Role的Agent共享一个Policy网络；为False时，所有Agent独立使用一个Policy网络
  - [ ] Greedy Policy: 采用Greedy Policy的Agent不需要训练policy，直接向着当前的到的target位置以最大速度运动（由于难以控制速度进行精确捕获，碰撞也视作捕获成功）
- Target Agent Configuration
  - [ ] Route Loading: Target本身有多种Policy形式，一种是与Pursuit组无人机一起参与训练，用学习到的Policy进行运动；一种是Patrol，直接加载预定义的巡逻轨迹，沿线运动；一种是random, 在环境中随机游走；

- Training Settings
  - [ ] Config file all in on: 只用一个.yaml文件来管理所有训练过程中涉及的参数，通过加载该配置文件来进行训练
  - [ ] Validation & Evaluation: 训练一定episodes后，执行Validation，保存Validation过程的GIF图（标注性能指标、Agent运动轨迹、感知范围以及捕获结果），计算对应性能指标，将性能指标最优的模型进行保存；训练完成后，进行Evaluation；
  - [ ] Immediate Results Saving: 在训练过程中保存必要的中间结果（Reward、Validation Metrics等），通过Tensorboard进行可视化观察
