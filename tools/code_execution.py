from smolagents import tool

@tool
def execute_python_code(code: str) -> str:
    """Executes arbitrary Python code and returns the captured output or error.

    Args:
        code: A string containing valid Python code to execute.

    Returns:
        str: Captured stdout output or formatted exception details.
    """
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    try:
        exec(code)
        output = redirected_output.getvalue()
        return output if output else "Code executed successfully with no output."
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        sys.stdout = old_stdout