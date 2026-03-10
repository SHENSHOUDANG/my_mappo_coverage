# Repository Guidelines

## Project Structure & Module Organization
本仓库为 `light_mappo` 的 Multi-UAV Pursuit 训练实现，当前主任务是 **hunter-only** 追捕（`num_explorers` 必须为 0）。

关键目录：
- `train/`: 训练与评估入口（`train.py`, `eval.py`）。
- `envs/`: 环境实现与向量封装（`env_uav_pursuit.py`, `env_continuous.py`, `env_wrappers.py`）。
- `runner/uav/`: 任务专用 Runner（`role_runner.py`）。
- `algorithms/`: MAPPO/RMAPPO 算法主体。
- `config/`: `defaults.yaml` 与任务配置文件。
- `agent_docs/`: 环境与任务说明文档。
- `results/`: 训练输出（模型、日志、GIF、评估结果）。

## Multi-UAV Pursuit Task
任务和环境细节请以 `agent_docs/pursuit_role.md` 为准。当前实现包含：
- 多 Hunter + 单 Target 的连续控制追逃。
- Target 三种策略：`learn` / `random` / `patrol`。
- 训练域随机化（`domain_randomization.train_split`）与固定评估任务（`eval.fixed_tasks(_file)`）。

## Build, Test, and Development Commands
建议优先使用 YAML 配置运行。

开启 Python 环境：
`conda activate mappo`

CUDA 训练前（按需）：
`unset LD_LIBRARY_PATH`

训练：
`python train/train.py --config_file <config_file_path>`

训练 + 性能统计：
`python train/train.py --config_file <config_file_path> --time_stat`

离线评估：
`python train/eval.py --config_file <config_file_path>`

生成固定评估任务（基础环境 × hunter数量，用于横向对比）：
`python scripts/gen_fixed_eval_task.py --output <eval_tasks_yaml_or_json> --num_base_envs <N> --hunter_count_choices <h1,h2,...> --world_size <min_ws> <max_ws> --target_policy_choices <random,patrol> --target_patrol_paths <path1,path2,...> --hunters_in_zone_choices <false,true> --seed_start <seed0> --seed_step <step> --rand_seed <rand_seed>`

## Implementation Notes (train/env)
- `train/train.py` 会先加载并深度合并 `config/defaults.yaml` 与用户 YAML。
- 训练环境使用 `mode=initial` 自动 reset；评估环境使用固定任务并 `mode=recover` 自动 reset，保证同任务可重复比较。
- 若 `eval.fixed_tasks_file` 存在，评估线程数会自动对齐任务数。
- `RoleBasedRunner` 使用角色共享策略：hunter 共享一个 policy；当 `target_policy_source=learn` 时 target 也参与训练。
- `domain_randomization.train_split.hunters_in_zone_choices` 可配置训练阶段初始化模式采样（zone / 非zone）。

## Coding Style & Naming Conventions
- 遵循 PEP 8，4 空格缩进。
- 函数/变量用 `snake_case`，类名用 `CamelCase`。
- import 顺序：stdlib -> third-party -> local。
- 避免无关的大规模格式化改动。
- YAML key 使用 `lower_snake_case`。

注释要求：
- 每个函数头需写明功能、输入/输出参数（名称、类型、含义）。
- 函数内部关键步骤建议用简洁注释说明逻辑。

## Testing Guidelines
当前无独立单元测试框架。提交前至少应完成：
- 用目标 YAML 成功启动训练（或评估）。
- 核对 `results/.../run*/` 下日志与模型文件是否按预期生成。

## Commit & Pull Request Guidelines
- Commit message 简短祈使句，建议 < 60 字符。
- PR 需包含：
  - 修改摘要。
  - 使用的配置文件与命令。
  - 关键结果（日志片段、CSV、GIF 路径）。
  - 兼容性影响（如有）。

## Configuration & Outputs
推荐仅通过 `--config_file` 管理实验参数。

输出路径（由 `train/train.py` 构造）：
`results/<env_name>/<algorithm_name>/<experiment_name>/run*/`

`run*` 下主要内容：
- `models/`: 常规模型与 `best_eval_*` 最优快照。
- `logs/`: TensorBoard 事件与 `summary.json`。
- `gifs/`: 训练/评估过程可视化 GIF。
- `log.csv`, `eval.csv`: 训练与评估结构化指标。
- `time_stat.csv`: `--time_stat` 开启时的耗时统计。
