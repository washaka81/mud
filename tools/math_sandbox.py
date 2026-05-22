import sys
import ast
import operator as op
import math as _math

# Supported operators
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

# Allowed functions and constants
allowed_funcs = {
    "sqrt": _math.sqrt, "sin": _math.sin, "cos": _math.cos, "tan": _math.tan,
    "log": _math.log, "log10": _math.log10, "log2": _math.log2,
    "exp": _math.exp, "abs": abs, "round": round, "floor": _math.floor,
    "ceil": _math.ceil, "pi": _math.pi, "e": _math.e, "tau": _math.tau,
}

def eval_expr(expr):
    """
    Safely evaluate a mathematical string.
    Supports basic arithmetic + math functions: sqrt, sin, cos, tan, log, exp, abs, round, pi, e
    """
    try:
        node = ast.parse(expr, mode='eval').body

        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](_eval(node.operand))
            elif isinstance(node, ast.Call):
                func_name = node.func.id if isinstance(node.func, ast.Name) else None
                if func_name in allowed_funcs:
                    args = [_eval(a) for a in node.args]
                    return allowed_funcs[func_name](*args)
                raise TypeError(f"Function not allowed: {func_name}")
            elif isinstance(node, ast.Name):
                if node.id in allowed_funcs:
                    return allowed_funcs[node.id]
                raise TypeError(f"Unknown symbol: {node.id}")
            else:
                raise TypeError(f"Unsupported mathematical operation: {type(node)}")

        result = _eval(node)
        print(f"SUCCESS:{result}")
    except Exception as e:
        print(f"ERROR:{str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        expression = " ".join(sys.argv[1:])
        eval_expr(expression)
    else:
        print("ERROR:No expression provided")
