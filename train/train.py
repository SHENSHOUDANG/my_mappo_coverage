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
from pathlib import Path

import numpy as np
import setproctitle
import torch

from utils.util import load_config
from envs.env_wrappers import DummyVecEnv


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
            env.seed(int(merged_cfg.exp.seed) + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(int(merged_cfg.exp.n_rollout_threads))])


def make_eval_env(merged_cfg):
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
            env.seed(int(merged_cfg.exp.seed) + rank * 1000)
            return env

        return init_env

    return DummyVecEnv(
        [get_env_fn(i) for i in range(int(merged_cfg.exp.n_eval_rollout_threads))]
    )


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
    envs = make_train_env(merged_cfg)
    eval_envs = make_eval_env(merged_cfg) if bool(merged_cfg.eval.use_eval) else None

    # Step 6: 构建Runner配置
    num_agents = int(merged_cfg.env.num_hunters) + int(merged_cfg.env.num_explorers) + 1
    runner_cfg = {
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir,
        "num_agents": num_agents,
    }

    # Step 7: 使用Multi-UAV专用Runner（分层配置 + 角色共享策略）
    from runner.uav.role_runner import RoleBasedRunner as Runner

    runner = Runner(runner_cfg, merged_cfg)
    runner.run()

    # Step 8: 收尾
    envs.close()
    if bool(merged_cfg.eval.use_eval) and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    cli_args = parser.parse_args()
    main(cli_args)
