#!/usr/bin/env python3
"""题库里的 Python 片段做零依赖静态体检（只用标准库 ast，不需要 torch、不联网）。

查三类问题 —— 都是本仓库真踩过的坑：
  1. 语法错误（ast.parse 失败）
  2. 未定义的名字：忘了定义变量、忘 import、调用的名字和签名对不上
     （如 max_len / draft_probs / w1 而签名写的是 weight1）
  3. self.X 用了但类里从没赋值过：调了不存在的方法（如 self._get_free_blocks）

用法：
  python3 scripts/code-lint.py <存放 .py 片段的目录>

输出：每处问题一行 "<片段名>:<行号>: <说明>"，无输出即全部通过。
片段名由 scripts/check-code.mjs 生成（形如 f007.py），那里负责翻译回题目/文档名。

三处免检名单，越靠后越窄，优先用窄的：
  - ALLOW      跨片段通用的外部符号（题库里别的题实现好的组件）
  - ALLOW_SELF 框架基类提供的属性（nn.Module / torch.optim.Optimizer 等）
  - 片段内注释  "# lint:allow a b c"：这一片段有意引用调用方给的名字
                （如训练循环片段里的 model / optimizer / dataloader）

不做严格的词法域分析 —— 只判断"这个名字在整个片段里有没有出现过定义"，
宁可漏报也不误报：片段是给人读的，不是能直接跑的完整程序。
"""
import ast
import builtins
import pathlib
import re
import sys

# 有意引用的外部符号：由题库里别的题实现，这里只当已有组件调用
ALLOW = {
    "paged_attention_kernel",   # 真实实现是 CUDA/Triton kernel
    "compute_gae",              # GAE 在另一道题里实现
    "ppo_loss",                 # PPO 的损失函数在另一道题里实现
    "RMSNorm",                  # 以下四个都是题库里其他题给出的组件
    "MultiHeadAttention",
    "SwiGLU",
    "FFN",
    "build_model",              # 组装模型的辅助函数
}

# 框架基类直接提供、不需要在本片段里赋值的属性
ALLOW_SELF = {
    "register_buffer",          # nn.Module
    "param_groups",             # torch.optim.Optimizer
    "state",                    # torch.optim.Optimizer
}

BUILTINS = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "self", "cls"}

ALLOW_MARKER = re.compile(r"#\s*lint:allow\s+(.*)")


def snippet_allow(source):
    """片段自己声明的外部名字：# lint:allow model optimizer dataloader"""
    names = set()
    for line in source.splitlines():
        match = ALLOW_MARKER.search(line)
        if match:
            names.update(match.group(1).split("#")[0].split())
    return names


def defined_names(tree):
    """片段里出现过的所有"定义"：赋值目标、参数、函数/类名、import、with/except 别名等。"""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def class_attributes(cls):
    """类里 self.X 的合法取值：被赋值过的属性 + 类里定义的方法名。

    self.register_buffer("cos", ...) 这种也要算数 —— 属性名在第一个字符串参数里，
    AST 上看不到 self.cos 的赋值（rope 那题就踩在这条上）。
    """
    attrs = set()
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in ("self", "cls")
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            attrs.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            attrs.add(node.name)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register_buffer"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            attrs.add(node.args[0].value)
    return attrs


def check(source):
    """返回 [(行号, 说明)]。"""
    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        return [(err.lineno or 1, f"语法错误: {err.msg}")]

    known = BUILTINS | ALLOW | snippet_allow(source) | defined_names(tree)
    problems = []

    for node in ast.walk(tree):
        # 未定义的名字
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in known:
            problems.append((node.lineno, f"未定义的名字: {node.id}"))

        # self.X 用了但类里没有
        elif isinstance(node, ast.ClassDef):
            attrs = class_attributes(node)
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"
                    and isinstance(sub.ctx, ast.Load)
                    and sub.attr not in attrs
                    and sub.attr not in ALLOW_SELF
                ):
                    problems.append((sub.lineno, f"self.{sub.attr} 在类里没有定义"))

    # 同一行可能重复报（如 a = a + 1），去重后按行号排序
    return sorted(set(problems))


def main(argv):
    if len(argv) != 2 or not pathlib.Path(argv[1]).is_dir():
        print("用法: python3 scripts/code-lint.py <片段目录>", file=sys.stderr)
        return 2

    # Windows 管道默认按 GBK 编码，中文说明到 Node 那边会变成乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    for path in sorted(pathlib.Path(argv[1]).glob("*.py")):
        for lineno, message in check(path.read_text(encoding="utf-8")):
            print(f"{path.name}:{lineno}: {message}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))