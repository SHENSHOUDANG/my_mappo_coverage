#!/usr/bin/env python3
"""
对比多个 eval_hunter_bucket_metrics_ep_*.json 的关键性能曲线。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """
    功能:
        解析命令行参数。
    输入:
        无。
    输出:
        argparse.Namespace: 解析后的参数对象。
    """
    # Step 1: 构建参数解析器并注册必需参数
    parser = argparse.ArgumentParser(
        description="Compare bucket metrics from multiple eval_hunter_bucket_metrics json files."
    )
    parser.add_argument(
        "--jsons",
        type=str,
        nargs="+",
        required=True,
        help="多个 eval_hunter_bucket_metrics_ep_*.json 文件路径。",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        required=True,
        help="与 --jsons 一一对应的曲线名称。",
    )
    parser.add_argument(
        "--ls",
        action="store_true",
        help="为 True 时用线型区分；不传则为 False，使用颜色区分。",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="可选：指定使用的 bucket 名称（默认自动选择第一个）。",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="可选：输出图片路径；不填则直接弹窗展示。",
    )
    return parser.parse_args()


def _load_metric_xy(
    json_path: Path,
    bucket_name: str | None,
    metric_name: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    功能:
        从单个bucket指标JSON中读取指定metric的x/y序列。
    输入:
        json_path (Path): 指标JSON文件路径。
        bucket_name (str | None): 指定bucket名；为None时自动选择第一个。
        metric_name (str): 指标名（如eval_reward/capture_rate/capture_steps）。
    输出:
        Tuple[np.ndarray, np.ndarray, str]:
            - x坐标数组；
            - y坐标数组；
            - 实际使用的bucket名称。
    """
    # Step 1: 读取并检查JSON结构
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid json object: {json_path}")
    buckets = data.get("buckets", {})
    if not isinstance(buckets, dict) or len(buckets) == 0:
        raise ValueError(f"Missing non-empty 'buckets' in: {json_path}")

    # Step 2: 解析bucket并读取目标metric
    used_bucket = str(bucket_name) if bucket_name is not None else str(next(iter(buckets.keys())))
    if used_bucket not in buckets:
        raise ValueError(f"Bucket '{used_bucket}' not found in: {json_path}")
    bucket_metrics = buckets.get(used_bucket, {})
    if not isinstance(bucket_metrics, dict):
        raise ValueError(f"Bucket '{used_bucket}' is not a dict in: {json_path}")
    metric_obj = bucket_metrics.get(metric_name)
    if not isinstance(metric_obj, dict):
        raise ValueError(f"Metric '{metric_name}' missing in {json_path} bucket='{used_bucket}'")

    x_raw = metric_obj.get("x", [])
    y_raw = metric_obj.get("y", [])
    x = np.asarray(x_raw, dtype=np.float32)
    y = np.asarray(y_raw, dtype=np.float32)
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Metric '{metric_name}' x/y length mismatch in {json_path}: {x.shape[0]} vs {y.shape[0]}"
        )
    return x, y, used_bucket


def _build_styles(num_curves: int, use_line_style: bool) -> List[Dict[str, str]]:
    """
    功能:
        生成每条曲线的绘图风格（颜色/线型）。
    输入:
        num_curves (int): 曲线数量。
        use_line_style (bool): 是否启用线型区分模式。
    输出:
        List[Dict[str, str]]: 每条曲线的style字典。
    """
    # Step 1: 准备颜色与线型候选
    color_cycle = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    line_cycle = ["-", "--", "-.", ":"]
    styles: List[Dict[str, str]] = []

    # Step 2: 根据参数选择区分策略
    for idx in range(num_curves):
        if use_line_style:
            styles.append({"color": "tab:blue", "linestyle": line_cycle[idx % len(line_cycle)]})
        else:
            styles.append({"color": color_cycle[idx % len(color_cycle)], "linestyle": "-"})
    return styles


def _plot_one_metric(
    ax,
    json_paths: List[Path],
    names: List[str],
    styles: List[Dict[str, str]],
    bucket_name: str | None,
    metric_name: str,
    title: str,
    y_label: str,
) -> str:
    """
    功能:
        在单个子图中绘制同一指标的多条对比曲线。
    输入:
        ax: matplotlib子图对象。
        json_paths (List[Path]): 多个指标JSON路径。
        names (List[str]): 对应曲线名称列表。
        styles (List[Dict[str, str]]): 对应曲线样式列表。
        bucket_name (str | None): 指定bucket名；None时自动选第一个。
        metric_name (str): 指标名。
        title (str): 子图标题。
        y_label (str): y轴标签。
    输出:
        str: 实际使用的bucket名称（用于总标题展示）。
    """
    # Step 1: 循环加载并绘制每条曲线
    used_bucket_final = ""
    for idx, (path, name) in enumerate(zip(json_paths, names)):
        x, y, used_bucket = _load_metric_xy(path, bucket_name, metric_name)
        used_bucket_final = used_bucket
        y = np.where(np.isfinite(y), y, np.nan)
        ax.plot(
            x,
            y,
            label=str(name),
            color=styles[idx]["color"],
            linestyle=styles[idx]["linestyle"],
            linewidth=2.0,
            marker="o",
            markersize=3.5,
        )

    # Step 2: 设置坐标轴样式
    ax.set_title(title)
    ax.set_xlabel("Num Hunters")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    return used_bucket_final


def main() -> None:
    """
    功能:
        程序入口：读取多组bucket指标JSON并绘制reward/capture_rate/capture_steps对比图。
    输入:
        无（从CLI读取）。
    输出:
        无。
    """
    # Step 1: 参数校验与路径准备
    args = parse_args()
    if len(args.jsons) != len(args.names):
        raise ValueError(
            f"--jsons count ({len(args.jsons)}) must equal --names count ({len(args.names)})"
        )
    json_paths = [Path(p) for p in args.jsons]
    for path in json_paths:
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

    # Step 2: 创建画布并绘制3个核心指标
    styles = _build_styles(num_curves=len(json_paths), use_line_style=bool(args.ls))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    used_bucket = _plot_one_metric(
        ax=axes[0],
        json_paths=json_paths,
        names=list(args.names),
        styles=styles,
        bucket_name=args.bucket,
        metric_name="eval_reward",
        title="Eval Reward",
        y_label="Reward",
    )
    _plot_one_metric(
        ax=axes[1],
        json_paths=json_paths,
        names=list(args.names),
        styles=styles,
        bucket_name=used_bucket,
        metric_name="capture_rate",
        title="Capture Rate",
        y_label="Rate",
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_yticks(np.linspace(0.0, 1.0, 6))
    axes[1].set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0.0, 1.0, 6)])
    _plot_one_metric(
        ax=axes[2],
        json_paths=json_paths,
        names=list(args.names),
        styles=styles,
        bucket_name=used_bucket,
        metric_name="capture_steps",
        title="Capture Steps",
        y_label="Steps",
    )
    axes[2].set_ylim(0.0, 300.0)
    mode_desc = "LineStyle" if bool(args.ls) else "Color"
    fig.suptitle(f"Bucket Compare ({used_bucket}) | Style={mode_desc}")

    # Step 3: 输出图片或直接展示
    if args.out is not None and len(str(args.out).strip()) > 0:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        print(f"[Saved] {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
