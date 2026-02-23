# UAV Pursuit Environment (Roles, Spaces, Rewards)

## Environment
- **World**: 2D square with bounds `[-world_size, world_size]` and time step `dt`.
- **Agents**: `num_hunters` + `num_blockers` + 1 target (evasion). Roles are fixed by index.
- **Episode end**: capture or `max_steps`.
- **Target patrol mode**: if `target_policy_source=patrol`, the target action is overridden to follow waypoints loaded from `target_patrol_path` (with optional named routes).
- **Pursuit sharing**: pursuers fully share their own positions/velocities at all times; target observations are shared across the pursuit team only when any pursuer detects the target.

## Roles
- **Hunter**: pursuer with highest max speed, smaller perception range.
- **Blocker**: pursuer with larger perception range, lower max speed.
- **Target**: evader (single agent), may be learned or patrol-driven.

## State / Observation Space
Per-agent observation vector:
- Own position `(x, y)` and velocity `(vx, vy)`.
- For every other agent: relative position `(dx, dy)`, relative velocity `(dvx, dvy)`, and distance `d`.
  - If the other agent is outside the observer’s perception range, `(dx, dy, dvx, dvy, d)` is zeroed.
  - Pursuer-to-pursuer observations are always shared (full visibility within the pursuit team).
  - The pursuers' perception range only affects target observations; target observations are shared within the team when any pursuer detects the target.
  - The target observes pursuers only within its own perception range (asymmetric visibility).
- Pursuit shared target memory (hunter/blocker only): pursuit-observed target position/velocity **in relative coordinates** `(dx, dy, dvx, dvy)`, plus `last_seen_age` (steps since last true sighting).
  - If any pursuer sees the target this step, all pursuers receive the current target position/velocity with `last_seen_age = 0`.
  - If no pursuer sees the target, pursuers receive the last seen position/velocity and `last_seen_age` increments.
  - When the target is not visible, pursuers still receive a noisy relative target position/velocity (never zeroed); once visible, the noise is cleared.
  - The target agent itself receives zeros in this memory slot.

Dimension: `obs_dim = 4 + (agent_num - 1) * 5 + 5`.

Centralized observation (for shared critics): concatenation of all agents’ observations with dimension `obs_dim * agent_num`.

## Action Space
Continuous 2D action for each agent: `Box(low=-1, high=1, shape=(2,))`.
- Actions are scaled by role-specific max speed and applied as velocity updates.
- Positions are updated with `dt` and clipped to world bounds.

## Rewards
Let `d` be the distance from a pursuer to the target, and `min_distance` be the closest hunter distance to the target.
- **Hunter**: `-d + 10` on capture, otherwise `-d`.
- **Blocker**: `-0.7 * d + 6` on capture, otherwise `-0.7 * d`.
- **Target**: `min_distance - 12` on capture, otherwise `min_distance`.
- **Speed penalty**: all agents receive `-speed_penalty * speed^2` to discourage meaningless high-speed motion.
- **Target lost penalty (blocker only)**: if the target is not visible to any pursuer, each blocker receives
  `-(lost_target_penalty + lost_target_penalty_age_scale * last_seen_age)` to encourage active search.

## Capture Condition
Capture occurs when any hunter stays within `capture_radius` of the target for `capture_steps` consecutive steps.

## Collision Condition
- If any two agents are within `collision_radius`, they are marked as collided.
- If the target collides, the episode ends immediately and capture does not count.
- Non-target collisions do not end the episode, but collided agents are marked done (stop moving).
- Collision penalties are applied once per pair:
  - If the pair is approaching (`dot(v_rel, p_rel) < 0`), both are penalized.
  - Otherwise, the faster agent is penalized.
  - Penalty magnitude: `collision_penalty_k * speed`.
