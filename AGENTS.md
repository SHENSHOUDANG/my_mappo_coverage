# Repository Guidelines

## Project Structure & Module Organization
This repository hosts `light_mappo`, a lightweight MAPPO training stack centered on a Multi‑UAV pursuit‑evasion environment. Key directories:
- `algorithms/`: MAPPO core implementations (policy, actor‑critic, RNN/CNN/MLP helpers).
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
其需要借助config中的.yaml配置文件来开启一段训练

## Coding Style & Naming Conventions
Follow PEP 8 with 4‑space indentation. Use `snake_case` for functions/variables and `CamelCase` for classes. Keep imports grouped as stdlib, third‑party, then local. No enforced formatter; keep edits consistent with surrounding code and avoid large reformat‑only diffs. YAML keys should be clear `lower_snake_case`.

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
