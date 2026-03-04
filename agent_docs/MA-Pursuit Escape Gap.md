# 摘要
项目提出并实现了一个面向多无人机协同追逃任务的连续控制训练框架。将多 Hunter 与单 Target 的博弈过程建模为多智能体马尔可夫决策过程，并基于 MAPPO/RMAPPO 完成策略优化。针对多 Hunter 在接近目标时容易因碰撞与路径冲突导致协同退化的问题，项目引入了基于最大潜在逃脱夹角的围捕几何奖励：Hunter 被激励形成更高包围质量并压制目标沿缺口逃逸，Target 被激励朝缺口方向运动。工程层面，框架支持固定任务可复现评估、分桶统计、过程可视化，以及近实战初始化（Hunter 区域编队随机投放与 Target 区域避让）。实验接口与日志体系完整，具备可复现实验与后续算法扩展能力。

---

# 1. 研究背景与意义
## 研究背景
当前的许多无人机集群应用中的群体决策过程都可以抽象为“搜索——分配——执行”三个阶段的任务流程：
* 搜索阶段: 对指定任务区域开展覆盖式搜索，在指定时间内尽可能多地找到任务目标；
* 分配阶段: 每当发现新的任务目标 / 完成一个区域搜索后，对目前发现的任务目标进行任务分配，派遣指定数量无人机处理任务目标；
* 执行阶段: 根据任务分配结果，组成任务小组执行具体任务。

而其中的执行阶段中，最常见的场景便是多个无人机与任务目标之间的相互博弈，需要多个无人机互相配合，尽快到达任务目标周边完成抓捕任务，而任务目标则需要尽可能地远离追捕无人机避免被抓捕。

本方法主要针对以下几个问题进行针对性优化：
 - 追捕任务无人机数量弹性可变：本文的追捕无人机只利用自身的局部观测信息独立开展追捕任务，且在训练阶段所有追捕无人机共用一个policy网络，从而实现追捕无人机数量的弹性扩展；
 - 训练/评估场景泛化：现有的许多RL追逃方法仅仅随机初始化追捕无人机位置来进行实验评估，覆盖的场景case过少。本文为Target设置了三种行动策略（learn，patrol，random），模拟任务目标 智能躲避追捕无人机/沿固定路线巡逻/随机游走 三类行为模式，同时在训练阶段可以随机变更hunters数量和Target策略模式进行训练；同时在评估阶段生成了100个不同设置的固定场景用于进行性能评估；
 - 追捕升级为围捕动作：相比于单纯的追捕，本文希望在追捕目标的同时，无人机主动形成有效包围圈压缩任务目标的潜在逃脱空间，具体可参考escape-gap reward的实现；


### 1.1 核心目标
本项目聚焦 **多无人机协同追逃（Multi-UAV Pursuit-Evasion）** 场景，构建了一个可训练、可评估、可视化的连续控制仿真平台。当前主任务为 **Hunter-only**：多个 Hunter 协同追捕单个 Target，并在碰撞与边界约束下学习高效策略。

### 1.2 项目意义
- 为多机协同控制、安防巡捕、动态围捕等任务提供可复现实验平台；
- 提供从环境建模、奖励设计到训练评估闭环的工程化基线；
- 为后续真实系统迁移（策略验证、控制约束映射）提供算法与仿真基础。

### 1.3 主要亮点
- **角色化建模**：清晰区分 Hunter/Target，支持 `learn/random/patrol` 多目标策略源；
- **连续动力学控制**：支持 `velocity/acceleration` 控制模式与转向限制；
- **协同奖励增强**：在基础追捕奖励上引入围捕结构奖励（escape-gap），鼓励“包围压缩逃脱空间”；
- **可复现评估机制**：训练 `initial` 与评估 `recover` 双模式 reset，支持固定任务集；
- **可解释可视化**：渲染包含轨迹、感知/碰撞半径、围捕几何（escape radius / 拦截段 / 最大逃脱扇区）；
- **近实战化配置**：支持 Hunter 区域编队起飞（zone 随机分布），并可约束 Target 避开该区域，

---

## 2. 技术原理

### 2.1 整体框架

项目采用“环境层 + 算法层 + runner层 + 训练/评估入口”的分层结构：

1) **环境层（Env）**
- `UAVPursuitEnv` 定义任务状态转移、碰撞/捕获判定、奖励计算、渲染；
- `ContinuousActionEnv` 负责动作空间/接口封装，与 MAPPO 训练接口对齐；
- 向量环境封装支持并行 rollout。

2) **算法层（MAPPO/RMAPPO）**
- 基于共享/角色策略进行多智能体 PPO 优化；
- 支持集中式价值函数与循环策略（RMAPPO）。

3) **执行层（RoleBasedRunner）**
- 负责采样、存储、更新、日志、评估与模型保存；
- 支持按固定任务评估，并输出 `log.csv`、`eval.csv`、GIF、best模型快照。

4) **入口层（train/eval）**
- `train/train.py`：训练主入口（配置合并、环境构造、训练循环）；
- `train/eval.py`：离线重载模型评估入口（可指定 model_glob）。

---

### 2.2 MDP 定义

将追逃问题建模为多智能体马尔可夫决策过程（Markov Game）：

#### 2.2.1 状态空间（隐状态）
环境全局状态由以下变量构成：
- 所有 agent 的位置、速度、朝向、存活状态；
- active hunter 掩码（可变 hunter 数）；
- target 可见性与共享记忆状态；
- 碰撞、捕获计数器与 episode 时间步。

#### 2.2.2 观测空间（局部观测）
单智能体观测由四部分拼接：
- `own_obs`：自身归一化位置/速度；
- `neighbor_obs`：最近同阵营邻居槽位（含相对位置/速度/距离/有效位）；
- `target_obs`：相对目标特征（含可见性位）；
- `memory_obs`：团队共享目标记忆（仅 Hunter 可用）。
- `coord_summary_obs`（可选，默认开启，2维）：
  - `self_is_topk_by_target_distance`：自身是否处于“距离Target最近Top-K Hunter”；
  - `hunters_in_escape_radius_count`：当前与Target距离小于 `escape_radius` 的Hunter数量（潜在参与包围数量）。

#### 2.2.3 动作空间
每个 agent 动作为二维连续向量 `a ∈ [-1,1]^2`：
- `velocity` 模式：动作映射为目标速度；
- `acceleration` 模式：动作映射为加速度并积分更新速度；
- 可选转向角限制（高速时启用）。

协同摘要观测由配置控制：
- `env.coord_summary_obs_enable`：是否启用该2维观测槽位；
- `env.coord_topk_hunters`：Top-K阈值。

#### 2.2.4 状态转移与终止
单步流程：
1. 依据策略与目标行为源得到动作；
2. 更新动力学状态；
3. 计算碰撞/边界惩罚并标记失活；
4. 更新可见性与共享记忆；
5. 更新捕获计数并判断是否捕获；
6. 计算奖励并返回观测。

终止条件：
- 达到最大步长；
- Target 被捕获；
- Target 边界碰撞；
- 所有 active Hunter 失活。

---

### 2.3 Reward 设置

总奖励由多项线性叠加：

\[
R = R_{\text{hunter\_base}} + R_{\text{target\_base}} + R_{\text{streak}} + R_{\text{capture}} + R_{\text{collision}} + R_{\text{speed}} + R_{\text{escape\_gap}}
\]

#### 2.3.1 基础追捕项
- Hunter：越接近 Target（特别是进入 `capture_dis`）奖励越高；
- Target：与 Hunter 基础项镜像（鼓励拉开最小距离）。

#### 2.3.2 连续压制项（streak）
- Hunter 在捕获半径内的连续步计数形成额外奖励（有上限）；
- Target 获得对应负向项。

#### 2.3.3 捕获与碰撞项
- 捕获成功时 Hunter 获得正奖励，Target 受惩罚；
- 碰撞项包含安全区风险惩罚与硬碰撞惩罚，边界碰撞单独处理；
- Target 边界碰撞附加较大惩罚。

#### 2.3.4 速度正则项
- 对全体 agent 施加归一化速度惩罚，抑制不必要高速震荡。

#### 2.3.5 围捕结构项（escape-gap reward）
为缓解多 Hunter 追捕时互扰问题，引入“最大潜在逃脱夹角”建模：

1. 在 Target 周围 `escape_radius` 内筛选围捕 Hunter；
2. 每个 Hunter 依据 `block_length = capture_dis * escape_block_scale` 形成拦截角覆盖；
3. 对 360°离散 bins 计算 blocked mask，求最大未阻塞扇区（max escape gap）；
4. 构造两类方向激励（速度低于阈值不计算）：
   - **Hunter**：鼓励 Target 运动方向与逃脱缺口中心相反；
   - **Target**：鼓励其朝逃脱缺口中心运动。

当前实现将围捕项分解为：
- `reward_escape_gap_hunter`
- `reward_escape_gap_target`
- `reward_escape_gap`（二者和）

并在 info 中提供诊断量：
- `max_escape_gap_angle / center / valid`
- `escape_gap_encircle_score`
- `escape_gap_hunter_direction_score`
- `escape_gap_target_direction_score`

---

### 2.4 训练/评估环境设置

#### 2.4.1 配置组织
- 默认配置：`config/defaults.yaml`
- 训练时先加载 defaults，再与用户 YAML 深度合并；
- 推荐通过 `--config_file` 管理实验。

#### 2.4.2 训练环境（Train）
- 使用 `reset(mode=initial)`；
- 可启用域随机化 `domain_randomization.train_split`；
- 支持变动 active hunter 数、目标策略源、随机种子等。

#### 2.4.3 评估环境（Eval）
- 使用 `reset(mode=recover)` 保证同任务可复现；
- 支持 `eval.fixed_tasks` 或 `eval.fixed_tasks_file` 固定任务评估；
- 若使用外部 fixed task 文件，评估线程数自动与任务数对齐；
- 输出总体指标与按 hunter 数分桶指标（含新增 `max_escape_gap_angle`）。

#### 2.4.4 初始位置策略（关键）
1. **通用随机初始化**：全图均匀采样。  
2. **初始捕获保护**：若 Target 初始 `capture_dis` 内已有 active Hunter，则重采样（最多10次）。  
3. **Zone 编队初始化**（`hunters_in_zone=true`）：
   - 先按 `m=ceil(sqrt(max_hunters_num))` 生成 `m*m` 方阵槽位；
   - 槽位间距为 `hunter_zone_spacing = max(collision_dis*3, Hunter.safe_dis)`；
   - 阵列中心不是固定地图中心，而是每次 reset 随机采样；
   - 每次 reset 都随机重排 Hunter 与槽位映射。
4. **Target 避让 Zone（可选）**：
   - `target_avoid_hunter_zone=true` 时，Target 初始位置与 zone 中心至少保持 `target_hunter_zone_min_dis`（重采样上限10次）。
5. **recover 可复现性**：
   - `recover` 模式下会先重置随机种子，再执行随机槽位映射；
   - 因此映射过程本身是随机策略，但同任务 recover 结果保持一致可复现。

#### 2.4.5 运行命令
- 训练：  
`python train/train.py --config_file <your_config.yaml>`

- 离线评估：  
`python train/eval.py --config_file <your_config.yaml> --run_dir <results/.../runX>`

---

## 3. 小结

本项目已形成较完整的多UAV追逃技术链路：  
**任务建模 → 多项奖励塑形（含围捕几何）→ MAPPO/RMAPPO 训练 → 固定任务评估与可视化分析**。  

在工程上，项目兼顾了可复现性、可解释性和可扩展性；在算法上，围捕结构奖励与编队初始策略为后续提升协同捕获效率提供了明确抓手。

---

## 4. 术语表

- **Hunter-only**：仅训练/评估 Hunter 与单 Target 的任务设定（`num_explorers=0`）。
- **active hunter**：当前任务规格中参与 episode 的 Hunter 槽位。
- **capture_dis**：捕获判定半径；Hunter 进入该半径并满足连续步条件可触发捕获。
- **capture_step**：连续处于捕获半径内的最小步数阈值。
- **escape gap**：基于围捕几何估计的最大潜在可逃脱扇区。
- **encircle score**：包围质量分，最大逃脱夹角越小分数越高。
- **zone 编队初始化**：Hunter 先按固定阵列偏移排布，再整体随机投放到地图某区域中心。
- **recover reset**：评估模式 reset，保证固定任务可复现对比。

## 5. 符号表

- \(N_h\)：active Hunter 数量。
- \(p_i, v_i\)：第 \(i\) 个 Hunter 的位置与速度。
- \(p_t, v_t\)：Target 的位置与速度。
- \(d_i=\|p_i-p_t\|\)：Hunter 到 Target 的距离。
- \(d_c\)：`capture_dis`。
- \(L_b=d_c \cdot \text{escape\_block\_scale}\)：Hunter 拦截长度参数。
- \(\theta_i\)：Target 指向 Hunter 的方位角。
- \(\alpha_i=\arctan(L_b/d_i)\)：对应拦截角半宽。
- \(\Delta\theta_{\max}\)：最大潜在逃脱夹角。
- \(s_{\text{encircle}}\)：包围质量分（与 \(\Delta\theta_{\max}\) 负相关）。
- \(s_{\text{dir,h}}\)：Hunter 方向分（鼓励 Target 运动方向与缺口中心相反）。
- \(s_{\text{dir,t}}\)：Target 方向分（鼓励朝缺口中心方向）。
- \(R_{\text{escape,h}}\)：Hunter 侧 escape-gap 奖励。
- \(R_{\text{escape,t}}\)：Target 侧 escape-gap 奖励。
