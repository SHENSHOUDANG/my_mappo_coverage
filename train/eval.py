"""
Standalone evaluation entry for saved models under one run_dir.

设计目标:
1) 指定一次训练实验的run_dir后，可重载models中各类模型进行评估。
2) 评估结果GIF固定输出到各模型目录下的res/目录。
3) 复用训练入口的配置与环境构建逻辑，保持行为一致。
"""

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from utils.util import load_config
from train.train import (
    _build_target_learn_eval_specs,
    _load_eval_task_specs,
    _print_domain_randomization_settings,
    _resolve_eval_max_hunters_num,
    _resolve_train_max_hunters_num,
    make_eval_env,
    make_train_env,
)


def _infer_last_episode(run_dir: Path) -> int:
    """
    功能:
        从run目录日志中推断最后一个episode编号。
    输入:
        run_dir (Path): 单次实验目录。
    输出:
        int: 推断出的最后episode编号；若无法推断则返回0。
    """
    # Step 1: 优先从eval.csv读取最后episode。
    eval_csv = run_dir / "eval.csv"
    if eval_csv.exists():
        try:
            with open(eval_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if len(rows) > 0:
                return int(rows[-1].get("episode", 0))
        except Exception:
            pass

    # Step 2: 回退到log.csv读取最后episode。
    log_csv = run_dir / "log.csv"
    if log_csv.exists():
        try:
            with open(log_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if len(rows) > 0:
                return int(rows[-1].get("episode", 0))
        except Exception:
            pass

    # Step 3: 无日志时返回0。
    return 0


def main(args):
    """
    功能:
        独立评估入口：重载指定run_dir下保存模型并输出GIF评估结果。
    输入:
        args (argparse.Namespace):
            - config_file (str): 配置文件路径。
            - run_dir (str): 单次训练run目录路径。
            - cuda (bool): 是否启用GPU。
            - total_num_steps (int): 评估记录步数（可选）。
            - episode (int): 评估标注episode（可选）。
            - model_glob (str | None): 仅评估匹配该glob的模型目录（相对models目录）。
    输出:
        无。
    """
    # Step 1: 加载配置并准备设备
    merged_cfg = load_config(args.config_file)
    use_cuda = (bool(args.cuda) or bool(merged_cfg.exp.cuda)) and torch.cuda.is_available()
    if use_cuda:
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(int(merged_cfg.exp.n_training_threads))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(int(merged_cfg.exp.n_training_threads))

    # Step 2: 校验run_dir
    run_dir = Path(str(args.run_dir))
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    print(f"[EvalOnly] run_dir={run_dir}")

    # Step 3: 固定随机种子，构建训练/评估环境
    torch.manual_seed(int(merged_cfg.exp.seed))
    torch.cuda.manual_seed_all(int(merged_cfg.exp.seed))
    np.random.seed(int(merged_cfg.exp.seed))

    if not bool(merged_cfg.eval.use_eval):
        merged_cfg.eval.use_eval = True
    eval_task_specs, from_external_file = _load_eval_task_specs(merged_cfg)
    if bool(from_external_file) and eval_task_specs is not None:
        merged_cfg.exp.n_eval_rollout_threads = int(len(eval_task_specs))
        print(
            "[EvalConfig] override n_eval_rollout_threads={} (from external fixed task file)".format(
                int(merged_cfg.exp.n_eval_rollout_threads)
            )
        )
    train_max_hunters_num = _resolve_train_max_hunters_num(merged_cfg)
    eval_max_hunters_num = _resolve_eval_max_hunters_num(merged_cfg, eval_task_specs)
    _print_domain_randomization_settings(
        merged_cfg,
        eval_task_specs,
        train_max_hunters_num=train_max_hunters_num,
        eval_max_hunters_num=eval_max_hunters_num,
    )

    envs = make_train_env(merged_cfg, train_max_hunters_num=train_max_hunters_num)
    eval_envs = make_eval_env(
        merged_cfg,
        eval_task_specs,
        eval_max_hunters_num=eval_max_hunters_num,
    )
    eval_envs_target_learn = None
    if str(merged_cfg.env.target_policy_source).lower() == "learn":
        eval_task_specs_target_learn = _build_target_learn_eval_specs(eval_task_specs)
        eval_envs_target_learn = make_eval_env(
            merged_cfg,
            eval_task_specs_target_learn,
            eval_max_hunters_num=eval_max_hunters_num,
        )

    # Step 4: 构建Runner并执行“重载模型目录评估”
    from runner.uav.role_runner import RoleBasedRunner as Runner

    num_agents = int(train_max_hunters_num) + int(merged_cfg.env.num_explorers) + 1
    runner_cfg = {
        "envs": envs,
        "eval_envs": eval_envs,
        "eval_envs_target_learn": eval_envs_target_learn,
        "device": device,
        "run_dir": run_dir,
        "num_agents": num_agents,
        "train_max_hunters_num": int(train_max_hunters_num),
        "eval_max_hunters_num": int(eval_max_hunters_num),
        "init_csv": False,
    }
    runner = Runner(runner_cfg, merged_cfg)

    total_num_steps = (
        int(args.total_num_steps)
        if args.total_num_steps is not None
        else int(merged_cfg.exp.num_env_steps)
    )
    episode = int(args.episode) if args.episode is not None else _infer_last_episode(run_dir)
    print("[EvalOnly] use total_num_steps={}, episode={}".format(int(total_num_steps), int(episode)))
    runner._final_eval_saved_best_models(
        total_num_steps=total_num_steps,
        episode=episode,
        model_glob=args.model_glob,
    )

    # Step 5: 资源收尾
    envs.close()
    if eval_envs is not envs:
        eval_envs.close()
    if eval_envs_target_learn is not None and eval_envs_target_learn is not envs:
        eval_envs_target_learn.close()
    runner.writter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone evaluation for saved model dirs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config_file", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to one experiment run directory")
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--total_num_steps", type=int, default=None, help="Override total_num_steps for eval logging")
    parser.add_argument("--episode", type=int, default=None, help="Override episode id used in eval/GIF naming")
    parser.add_argument(
        "--model_glob",
        type=str,
        default=None,
        help="Only evaluate model dirs matching this glob under run_dir/models, e.g. best_eval_reward",
    )
    cli_args = parser.parse_args()
    main(cli_args)
