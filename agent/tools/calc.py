from __future__ import annotations

import ast
import math

from agent.tool import Tool, ToolException


class _SafeEval(ast.NodeVisitor):
    _bin_ops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
    }
    _unary_ops = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }
    _names = {"pi": math.pi, "e": math.e, "tau": math.tau}

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolException("calc: only int/float constants are allowed")

    def visit_Name(self, node: ast.Name):
        if node.id in self._names:
            return self._names[node.id]
        raise ToolException(f"calc: unknown name '{node.id}'")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in self._unary_ops:
            raise ToolException("calc: unsupported unary operator")
        return self._unary_ops[op_type](self.visit(node.operand))

    def visit_BinOp(self, node: ast.BinOp):
        op_type = type(node.op)
        if op_type not in self._bin_ops:
            raise ToolException("calc: unsupported operator")
        return self._bin_ops[op_type](self.visit(node.left), self.visit(node.right))

    def generic_visit(self, node):
        raise ToolException(f"calc: unsupported syntax ({type(node).__name__})")


class CalcTool(Tool):
    name = "calc"
    description = "Safely evaluate a basic math expression (supports + - * / // % **, parentheses, pi/e/tau)."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression, e.g. '(1+2)*3' or '2*pi'."}
        },
        "required": ["expression"],
    }

    def call(self, expression: str, **kwargs) -> str:
        try:
            tree = ast.parse(expression, mode="eval")
            value = _SafeEval().visit(tree)
            return str(value)
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"calc: {e}")

