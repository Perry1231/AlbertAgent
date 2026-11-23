from smolagents import tool


@tool
def execute_python_code(code: str) -> str:
    """
    Executes Python code and returns stdout or the value
    of the last expression.

    Args:
        code: A string containing valid Python code.

    Returns:
        Captured output, last expression value, or error.
    """

    import ast
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output

    try:
        # Parse the code so we can detect the final expression.
        tree = ast.parse(code, mode="exec")

        last_value = None

        # If the last statement is an expression,
        # evaluate it separately so its value is returned.
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expression = tree.body.pop()

            code_without_last = compile(
                tree,
                "<agent_code>",
                "exec",
            )

            namespace = {}

            exec(
                code_without_last,
                namespace,
                namespace,
            )

            last_value = eval(
                compile(
                    ast.Expression(last_expression.value),
                    "<agent_expression>",
                    "eval",
                ),
                namespace,
                namespace,
            )

        else:
            namespace = {}

            exec(
                compile(tree, "<agent_code>", "exec"),
                namespace,
                namespace,
            )

        output = redirected_output.getvalue().strip()

        if output:
            return output

        if last_value is not None:
            return str(last_value)

        return "Code executed successfully with no output."

    except Exception as e:
        return f"Error executing code: {type(e).__name__}: {e}"

    finally:
        sys.stdout = old_stdout