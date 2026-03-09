"""
Utility helpers for exporting actor/critic module architecture diagrams.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Tuple


def _iter_module_edges(module) -> Iterable[Tuple[str, str, str]]:
    """
    功能:
        遍历模型的层级结构并生成父子连接关系。
    输入:
        module (torch.nn.Module): 待导出的网络模块。
    输出:
        Iterable[Tuple[str, str, str]]:
            (parent_name, child_name, child_label) 的可迭代对象。
    """
    for parent_name, parent_mod in module.named_modules():
        for child_name, child_mod in parent_mod.named_children():
            full_child_name = child_name if parent_name == "" else f"{parent_name}.{child_name}"
            label = child_mod.__class__.__name__
            yield parent_name, full_child_name, label


def build_module_tree_dot(module, graph_name: str) -> str:
    """
    功能:
        将 PyTorch 模块层级转换为 Graphviz DOT 文本。
    输入:
        module (torch.nn.Module): 目标网络（如 actor / critic）。
        graph_name (str): 图名称。
    输出:
        str: DOT 格式字符串。
    """
    lines = [
        f'digraph "{graph_name}" {{',
        "  rankdir=LR;",
        "  node [shape=box, style=rounded, fontsize=10, fontname=Helvetica];",
        '  root [label="root", shape=ellipse, style=solid];',
    ]

    added_nodes = {""}
    for parent_name, child_name, child_label in _iter_module_edges(module):
        parent_id = "root" if parent_name == "" else parent_name.replace(".", "_")
        child_id = child_name.replace(".", "_")

        if child_name not in added_nodes:
            node_label = f"{child_name}\\n({child_label})"
            lines.append(f'  {child_id} [label="{node_label}"];')
            added_nodes.add(child_name)

        lines.append(f"  {parent_id} -> {child_id};")

    lines.append("}")
    return "\n".join(lines) + "\n"


def export_network_diagram(module, output_stem: Path, graph_name: str) -> Path:
    """
    功能:
        导出网络结构图为 DOT 文件，并在系统支持时渲染 PNG。
    输入:
        module (torch.nn.Module): 待导出网络模块。
        output_stem (Path): 输出文件前缀（不含扩展名）。
        graph_name (str): 图名称。
    输出:
        Path: DOT 文件路径。
    """
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    dot_path = output_stem.with_suffix(".dot")
    dot_text = build_module_tree_dot(module, graph_name=graph_name)
    dot_path.write_text(dot_text, encoding="utf-8")

    if shutil.which("dot"):
        png_path = output_stem.with_suffix(".png")
        subprocess.run(
            ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
            check=False,
        )

    return dot_path
