import ast
import math
import operator
from typing import Any


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "pow": pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
}

_ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _normalize_expression(expression: str) -> str:
    normalized = expression.strip()
    replacements = {
        "（": "(",
        "）": ")",
        "【": "(",
        "】": ")",
        "×": "*",
        "x": "*",
        "X": "*",
        "÷": "/",
        "，": ",",
        "^": "**",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target) # 把normalized中出现的所有source用target替换
    return normalized


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持整数和浮点数常量。")

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)
        if operator_type not in _BINARY_OPERATORS:
            raise ValueError(f"不支持的运算符: {operator_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _BINARY_OPERATORS[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)
        if operator_type not in _UNARY_OPERATORS:
            raise ValueError(f"不支持的单目运算符: {operator_type.__name__}")
        operand = _safe_eval(node.operand)
        return _UNARY_OPERATORS[operator_type](operand)

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"不支持的变量或常量: {node.id}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("仅支持直接调用白名单函数。")
        function_name = node.func.id
        function = _ALLOWED_FUNCTIONS.get(function_name)
        if function is None:
            raise ValueError(f"不支持的函数: {function_name}")
        args = [_safe_eval(arg) for arg in node.args]
        return function(*args)

    raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


def calculator(expression: str) -> str:
    """
    一个安全的计算器工具，支持四则运算、括号、幂运算以及部分常见数学函数。
    """
    normalized_expression = _normalize_expression(expression)
    print(f"🧮 正在计算: {normalized_expression}")

    try:
        parsed_expression = ast.parse(normalized_expression, mode="eval")
        result = _safe_eval(parsed_expression)
        return f"{normalized_expression} = {result}"
    except ZeroDivisionError:
        return "计算错误:除数不能为 0。"
    except Exception as exc:
        return f"计算错误:{exc}"
