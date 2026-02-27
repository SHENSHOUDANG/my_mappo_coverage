# light_mappo (Multi-UAV Pursuit)

本仓库是一个面向 **Multi-UAV Pursuit** 的轻量级 MAPPO/RMAPPO 训练实现，核心入口为 `train/train.py`，核心环境为 `envs/env_uav_pursuit.py`。

## 1. 当前可实现功能
- 多 Hunter + 单 Target 的连续动作追逃训练与评估。
- Target 策略可切换：`learn` / `random` / `patrol`。
- 训练阶段 domain randomization（按 episode 周期重采样任务）。
- 固定任务评估（支持 YAML/JSON 任务文件）。
- 角色化训练器：Hunter 共享策略；Target 可选是否参与训练。
- 自动保存模型、最优模型快照、CSV 指标、TensorBoard、GIF 可视化。
- 可选 `--time_stat` 输出训练各阶段耗时统计。

## 2. 快速开始
### 2.1 环境
```bash
conda activate mappo
```

如需 CUDA 训练且本机环境冲突，可先执行：
```bash
unset LD_LIBRARY_PATH
```

### 2.2 训练
```bash
python train/train.py --config_file config/acc_hunter_only_seen.yaml
```

### 2.3 训练耗时统计
```bash
python train/train.py --config_file config/acc_time_stat.yaml --time_stat
```

### 2.4 评估
```bash
python train/eval.py --config_file config/acc_hunter_only_seen.yaml
```

## 3. 代码结构
- `train/train.py`: 训练入口（配置加载、env 构建、runner 调度）。
- `envs/env_uav_pursuit.py`: 追逃环境主逻辑（动力学、观测、奖励、碰撞、渲染）。
- `envs/env_continuous.py`: 与 MAPPO 接口对齐的连续动作封装。
- `envs/env_wrappers.py`: `DummyVecEnv` 同步向量环境与自动 reset。
- `runner/uav/role_runner.py`: 角色化训练循环、评估、日志、GIF、best 模型管理。
- `config/defaults.yaml`: 默认配置基线。

## 4. train/train.py 实现方式（简要）
训练主流程：
1. 读取并深度合并配置：`defaults.yaml + 用户yaml`。
2. 检查算法模式：
   - `rmappo` 要求启用 recurrent policy。
   - `mappo` 要求关闭 recurrent policy。
3. 创建结果目录：
   `results/<env_name>/<algorithm_name>/<experiment_name>/run*/`
4. 创建训练环境：`make_train_env()`（`auto_reset=initial`）。
5. 创建评估环境：`make_eval_env()`（固定任务 + `auto_reset=recover`）。
6. 若使用外部 fixed task 文件，自动覆盖评估线程数为任务数。
7. 构建 `RoleBasedRunner` 并执行 `run()` 或 `run_time_stat()`。

## 5. env_uav_pursuit.py 实现方式（简要）
- Agent 体系：`HunterAgent`, `TargetAgent`（Explorer 类保留但当前任务不启用）。
- 动力学：支持 `velocity` 与 `acceleration` 两种控制。
- 观测：`own + nearest neighbors + target + shared memory`。
- 捕获：任一 Hunter 连续 `capture_step` 步进入 `capture_dis` 视为捕获。
- 碰撞：
  - 边界风险与边界硬碰撞。
  - Hunter-Hunter 两两碰撞（Target 不参与两两碰撞）。
- 奖励：基础追逃 + streak + capture + collision + speed penalty。
- reset 模式：`initial` / `recover` / `regen`。
- patrol 路线：从 JSON 读取，可按 route name 选择，并过滤边界危险航点。

更完整说明见：`agent_docs/pursuit_role.md`。

## 6. 配置说明
默认配置在 `config/defaults.yaml`，常用覆盖项：
- `exp`: 算法名、线程数、总步数、seed。
- `env`: world_size、episode_length、hunter数量、target策略。
- `Hunter` / `Target`: 控制模式、速度/加速度、感知半径、转向参数。
- `reward`: 捕获/碰撞/速度等奖励系数。
- `eval`: 固定评估任务配置。
- `domain_randomization.train_split`: 训练任务重采样策略。

## 7. 输出产物
每次训练会生成一个新 `run*` 目录，典型内容：
- `models/`: 当前模型与 `best_eval_*` 快照。
- `logs/`: TensorBoard 日志和 `summary.json`。
- `gifs/`: 训练/评估可视化。
- `log.csv`: 训练过程指标。
- `eval.csv`: 评估指标（reward/capture_rate/capture_steps/alive_rate）。
- `time_stat.csv`: 开启 `--time_stat` 时生成。

## 8. 注意事项
- 当前环境强制 hunter-only：`env.num_explorers` 必须为 `0`。
- 评估建议使用固定任务文件，保证不同实验可复现、可对比。
- 若 `target_policy_source=learn`，训练中会额外评估 `target_learn` 桶。

## 9. 多目标推理场景（不训练）
仓库新增了 inference-only MVP，用于你描述的 `N hunters + K explorers + M targets` 全流程推理：
- 环境文件：`envs/env_uav_multi_infer.py`
- 入口脚本：`train/infer_multi.py`
- 示例配置：`config/multi_infer_demo.yaml`

运行示例（无模型，启发式hunter + random/patrol/static target）：
```bash
python train/infer_multi.py --config_file config/multi_infer_demo.yaml --episodes 2 --deterministic
```

实时窗口可视化（有图形界面时）：
```bash
python train/infer_multi.py --config_file config/multi_infer_demo.yaml --episodes 1 --render
```

保存 GIF：
```bash
python train/infer_multi.py \
  --config_file config/multi_infer_demo.yaml \
  --episodes 1 --save_gif --gif_frame_interval 20
```

运行示例（加载训练好的 hunter actor）：
```bash
python train/infer_multi.py \
  --config_file config/multi_infer_demo.yaml \
  --hunter_actor results/<env>/<algo>/<exp>/run*/models/actor_hunter.pt \
  --episodes 3 --deterministic
```

可选加载 `--target_actor`，仅对 `policy_type=learn` 的 target 生效。脚本会输出每回合捕获率/发现率，并在 `results/multi_infer/` 写入汇总 JSON。
