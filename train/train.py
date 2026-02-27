"""
MAPPO training entry for Multi-UAV Pursuit.

设计原则:
1) 外部配置始终使用分层结构（merged_cfg）。
2) 环境创建直接读取分层参数，不依赖扁平化参数。
3) 仅在Runner内部需要初始化算法组件时，才进行扁平化参数映射。
"""

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)

import argparse
import json
from pathlib import Path

import numpy as np
import setproctitle
import torch
import yaml

from utils.util import load_config
from envs.env_wrappers import DummyVecEnv


def _load_eval_task_specs(merged_cfg):
    """
    构建评估线程对应的固定任务规格列表。

    输入:
        merged_cfg (EasyDict): 分层配置对象。
    输出:
        list[dict] | None: 长度为n_eval_rollout_threads的任务规格列表；若未配置则返回None。
    """
    fixed_tasks = list(merged_cfg.eval.fixed_tasks)
    fixed_tasks_file = merged_cfg.eval.fixed_tasks_file
    from_external_file = fixed_tasks_file is not None
    if fixed_tasks_file is not None:
        task_path = Path(str(fixed_tasks_file))
        if not task_path.is_absolute():
            root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            task_path = root / task_path
        if not task_path.exists():
            raise FileNotFoundError(f"eval.fixed_tasks_file not found: {task_path}")
        if task_path.suffix.lower() in [".yaml", ".yml"]:
            with open(task_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        elif task_path.suffix.lower() == ".json":
            with open(task_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported eval task file format: {task_path.suffix}")

        if isinstance(data, dict):
            if "tasks" not in data:
                raise ValueError("eval task file dict must contain key 'tasks'")
            fixed_tasks = list(data["tasks"])
        elif isinstance(data, list):
            fixed_tasks = list(data)
        else:
            raise ValueError("eval task file must be a list or a dict with key 'tasks'")
    if len(fixed_tasks) == 0:
        return None, bool(from_external_file)

    # 兼容策略:
    # 1) 外部任务文件：eval线程数自动对齐为任务数；
    # 2) 非外部任务文件：沿用配置线程数，并将fixed_tasks按线程数展开。
    if from_external_file:
        return [dict(x) for x in fixed_tasks], True

    n_env = int(merged_cfg.exp.n_eval_rollout_threads)
    if n_env <= 0:
        return None, False
    out = []
    for i in range(n_env):
        out.append(dict(fixed_tasks[i % len(fixed_tasks)]))
    return out, False


def _build_target_learn_eval_specs(eval_task_specs):
    """
    基于固定评估任务构建Target=learn版本任务列表。

    输入:
        eval_task_specs (list[dict]): 固定评估任务列表。
    输出:
        list[dict]: 将target_policy_source强制为learn后的任务列表。
    """
    out = []
    for spec in eval_task_specs:
        s = dict(spec)
        s["target_policy_source"] = "learn"
        out.append(s)
    return out


def _print_domain_randomization_settings(merged_cfg, eval_task_specs):
    """
    打印domain randomization配置，便于训练启动时快速确认。

    输入:
        merged_cfg (EasyDict): 分层配置对象。
        eval_task_specs (list[dict] | None): 展开后的评估固定任务列表。
    输出:
        无。
    """
    train_split = merged_cfg.domain_randomization.train_split
    print(
        "[DomainRandConfig] train.enable={}, interval={}, prob={}, hunter_choices={}, seed_range={}, target_policies={}, patrol_pool={}".format(
            bool(train_split.enable),
            int(train_split.regen_interval_episode),
            float(train_split.regen_prob),
            list(train_split.hunter_count_choices),
            list(train_split.seed_range),
            list(train_split.target_policy_choices),
            list(train_split.patrol_name_choices),
        )
    )
    eval_source = "inline"
    if merged_cfg.eval.fixed_tasks_file is not None:
        eval_source = str(merged_cfg.eval.fixed_tasks_file)
    print(
        "[EvalConfig] fixed_task_source={}, fixed_tasks={}, dual_eval_target_learn={}".format(
            eval_source,
            0 if eval_task_specs is None else int(len(eval_task_specs)),
            str(merged_cfg.env.target_policy_source).lower() == "learn",
        )
    )


def make_train_env(merged_cfg):
    """
    创建训练环境向量封装。

    输入:
        merged_cfg (EasyDict): 分层配置对象。
            - merged_cfg.exp.n_rollout_threads: 训练并行环境数（int）。
            - merged_cfg.exp.seed: 基础随机种子（int）。
    输出:
        DummyVecEnv: 向量化环境，接口满足现有MAPPO训练循环。
    """

    def get_env_fn(rank):
        """
        输入:
            rank (int): 当前环境线程编号。
        输出:
            callable: 延迟创建单个环境实例的函数。
        """

        def init_env():
            """
            输入:
                无。
            输出:
                ContinuousActionEnv: 单个连续动作环境实例。
            """
            from envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv(merged_cfg)
            env.set_regen_scope("train")
            env.seed(int(merged_cfg.exp.seed) + rank * 1000)
            return env

        return init_env

    vec_env = DummyVecEnv([get_env_fn(i) for i in range(int(merged_cfg.exp.n_rollout_threads))])
    vec_env.set_auto_reset(mode="initial")
    return vec_env


def make_eval_env(merged_cfg, eval_task_specs):
    """
    创建评估环境向量封装。

    输入:
        merged_cfg (EasyDict): 分层配置对象。
            - merged_cfg.exp.n_eval_rollout_threads: 评估并行环境数（int）。
            - merged_cfg.exp.seed: 基础随机种子（int）。
    输出:
        DummyVecEnv: 评估向量环境。
    """

    def get_env_fn(rank):
        """
        输入:
            rank (int): 当前环境线程编号。
        输出:
            callable: 延迟创建单个环境实例的函数。
        """

        def init_env():
            """
            输入:
                无。
            输出:
                ContinuousActionEnv: 单个连续动作环境实例。
            """
            from envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv(merged_cfg)
            env.set_regen_scope("eval")
            env.seed(int(merged_cfg.exp.seed) * 10 + rank * 1000)
            return env

        return init_env

    eval_env = DummyVecEnv(
        [get_env_fn(i) for i in range(int(merged_cfg.exp.n_eval_rollout_threads))]
    )
    if eval_task_specs is None:
        raise ValueError("Eval requires fixed tasks. Please set eval.fixed_tasks or eval.fixed_tasks_file.")
    eval_env.reset_task(mode="regen", task_specs=eval_task_specs)
    eval_env.set_auto_reset(mode="recover", task_specs=eval_task_specs)
    return eval_env


parser = argparse.ArgumentParser(
    description="mappo-pursuit", formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--config_file",
    type=str,
    required=True,
    help="Path to YAML config file",
)
parser.add_argument(
    "--cuda",
    action="store_true",
    help="Use GPU or not (CLI flag has higher priority than yaml false)",
)
parser.add_argument(
    "--time_stat",
    action="store_true",
    help="Enable detailed per-episode runtime statistics and run runner.run_time_stat()",
)


def main(args):
    """
    训练入口。

    输入:
        args (argparse.Namespace):
            - config_file (str): 配置文件路径。
            - cuda (bool): 命令行是否强制启用GPU。
    输出:
        无（执行训练并写入日志/模型文件）。
    """
    merged_cfg = load_config(args.config_file)

    # Step 1: 校验算法与RNN策略开关一致性
    algo_name = str(merged_cfg.exp.algorithm_name)
    if algo_name == "rmappo":
        assert merged_cfg.model.use_recurrent_policy or merged_cfg.model.use_naive_recurrent_policy, \
            "rmappo requires recurrent policy."
    elif algo_name == "mappo":
        assert (not merged_cfg.model.use_recurrent_policy) and \
            (not merged_cfg.model.use_naive_recurrent_policy), \
            "mappo should disable recurrent policy."
    else:
        raise NotImplementedError(f"Unsupported algorithm: {algo_name}")

    # Step 2: 设备与线程设置
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

    # Step 3: 结果目录构建
    run_root = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / str(merged_cfg.env.env_name)
        / str(merged_cfg.exp.algorithm_name)
        / str(merged_cfg.exp.experiment_name)
    )
    os.makedirs(str(run_root), exist_ok=True)

    exst_run_nums = [
        int(str(folder.name).split("run")[1])
        for folder in run_root.iterdir()
        if str(folder.name).startswith("run")
    ] if run_root.exists() else []
    curr_run = "run1" if len(exst_run_nums) == 0 else f"run{max(exst_run_nums) + 1}"
    run_dir = run_root / curr_run
    os.makedirs(str(run_dir), exist_ok=True)

    # Step 4: 进程名与随机种子
    setproctitle.setproctitle(
        f"{merged_cfg.exp.algorithm_name}-{merged_cfg.env.env_name}-{merged_cfg.exp.experiment_name}"
    )
    torch.manual_seed(int(merged_cfg.exp.seed))
    torch.cuda.manual_seed_all(int(merged_cfg.exp.seed))
    np.random.seed(int(merged_cfg.exp.seed))

    # Step 5: 构建环境
    eval_task_specs, from_external_file = _load_eval_task_specs(merged_cfg) if bool(merged_cfg.eval.use_eval) else (None, False)
    if bool(merged_cfg.eval.use_eval) and bool(from_external_file) and eval_task_specs is not None:
        merged_cfg.exp.n_eval_rollout_threads = int(len(eval_task_specs))
        print(
            "[EvalConfig] override n_eval_rollout_threads={} (from external fixed task file)".format(
                int(merged_cfg.exp.n_eval_rollout_threads)
            )
        )
    _print_domain_randomization_settings(merged_cfg, eval_task_specs)
    envs = make_train_env(merged_cfg)
    eval_envs = make_eval_env(merged_cfg, eval_task_specs) if bool(merged_cfg.eval.use_eval) else None
    eval_envs_target_learn = None
    if bool(merged_cfg.eval.use_eval) and str(merged_cfg.env.target_policy_source).lower() == "learn":
        eval_task_specs_target_learn = _build_target_learn_eval_specs(eval_task_specs)
        eval_envs_target_learn = make_eval_env(merged_cfg, eval_task_specs_target_learn)

    # Step 6: 构建Runner配置
    num_agents = int(merged_cfg.env.num_hunters) + int(merged_cfg.env.num_explorers) + 1
    runner_cfg = {
        "envs": envs,
        "eval_envs": eval_envs,
        "eval_envs_target_learn": eval_envs_target_learn,
        "device": device,
        "run_dir": run_dir,
        "num_agents": num_agents,
        "time_stat": bool(args.time_stat),
    }

    # Step 7: 使用Multi-UAV专用Runner（分层配置 + 角色共享策略）
    from runner.uav.role_runner import RoleBasedRunner as Runner

    runner = Runner(runner_cfg, merged_cfg)
    if bool(args.time_stat):
        runner.run_time_stat()
    else:
        runner.run()

    # Step 8: 收尾
    envs.close()
    if bool(merged_cfg.eval.use_eval) and eval_envs is not envs:
        eval_envs.close()
    if bool(merged_cfg.eval.use_eval) and eval_envs_target_learn is not None and eval_envs_target_learn is not envs:
        eval_envs_target_learn.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    cli_args = parser.parse_args()
    main(cli_args)
