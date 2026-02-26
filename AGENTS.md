# Repository Guidelines

## Project Structure & Module Organization
This repository hosts `light_mappo`, a lightweight MAPPO training stack centered on a Multi‑UAV pursuit‑evasion environment. Key directories:
- `algorithms/`: MAPPO core implementations (policy, actor‑critic, RNN/CNN/MLP helpers).
- `config/`: Training .yaml config files.
- `envs/`: environment wrappers and examples, including `uav_pursuit_env.py`.
- `runner/`: training loop orchestration (shared vs separated policies).
- `train/`: entry script `train.py`.
- `config/`: YAML configs (including patrol routes).
- `scripts/`: utilities (patrol route editor, MPE rendering).
- `utils/`: shared utilities.
- `results/`: training outputs (logs, artifacts).

## Multi-UAV Pursuit Task
The MAPPO in Multi-UAV Pursuit Task. 其具体环境配置可以参考文件：
@./agent_docs/pursuit_role.md

## Build, Test, and Development Commands
This project runs directly with Python and prefers YAML configs over long CLI argument lists.

开启Python环境：
`conda activate mappo`

支持CUDA训练：
`unset LD_LIBRARY_PATH`

开始训练
`python train/train.py --config_file <config_file_path>`

## Coding Style & Naming Conventions
Follow PEP 8 with 4‑space indentation. Use `snake_case` for functions/variables and `CamelCase` for classes. Keep imports grouped as stdlib, third‑party, then local. No enforced formatter; keep edits consistent with surrounding code and avoid large reformat‑only diffs. YAML keys should be clear `lower_snake_case`.

每个函数头都需要注释函数功能、输入输出参数名称、数据类型以及描述；函数内部需要对其大致步骤和逻辑进行说明

## Testing Guidelines
There is no dedicated unit test suite. 

## Commit & Pull Request Guidelines
Commit messages should be short, imperative, and under ~60 characters (e.g., “Update config”). Pull requests should include:
- A concise summary of changes.
- Configs and CLI args used.
- Any artifacts (log snippets, GIFs in `results/`).
- Linked issues and notes on breaking changes or new dependencies.

## Configuration & Outputs
Prefer `--config` YAML files for reproducibility. Outputs are written to:
`results/<env>/<scenario>/<algorithm>/<experiment_name>/run*/`.
Keep large generated files out of PRs unless explicitly requested.
