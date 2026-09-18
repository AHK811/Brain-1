from __future__ import annotations
import ast, operator as op
_ops={ast.Add:op.add,ast.Sub:op.sub,ast.Mult:op.mul,ast.Div:op.truediv,ast.Pow:op.pow,ast.USub:op.neg,ast.Mod:op.mod}
def _eval(node):
    if isinstance(node,ast.Expression): return _eval(node.body)
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)): return node.value
    if isinstance(node,ast.BinOp) and type(node.op) in _ops: return _ops[type(node.op)](_eval(node.left),_eval(node.right))
    if isinstance(node,ast.UnaryOp) and type(node.op) in _ops: return _ops[type(node.op)](_eval(node.operand))
    raise ValueError("disallowed")
def solve_arithmetic(expression: str) -> float:
    return float(_eval(ast.parse(expression.strip(), mode="eval")))
