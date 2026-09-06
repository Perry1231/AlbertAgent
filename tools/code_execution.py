import sys
from io import StringIO
from smolagents import tool


@tool
def execute_python_code(code: str) -> str:
    """
    Executes a snippet of Python code in a controlled environment and returns its standard output (print statements).
    Useful for complex mathematical calculations, data transformations, array manipulations, or string parsing.

    Args:
        code: A valid Python code string to be executed (e.g., 'result = sum([x**2 for x in range(10)])\nprint(result)').
    """
    # Redirect standard output to capture print statements
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    # Dictionary to capture local variables created during execution
    local_scope = {}
    
    try:
        # Execute the provided Python code
        exec(code, {}, local_scope)
        
        # Retrieve printed output
        output = redirected_output.getvalue().strip()
        
        if output:
            return f"Execution Output:\n{output}"
        elif local_scope:
            # If nothing was printed, display the last calculated variables
            vars_str = "\n".join([f"{k} = {v}" for k, v in local_scope.items() if not k.startswith("__")])
            return f"Code executed successfully. Environment variables:\n{vars_str}"
        else:
            return "Code executed successfully with no output."

    except Exception as e:
        return f"Execution Error ({type(e).__name__}): {str(e)}"
        
    finally:
        # Restore standard output
        sys.stdout = old_stdout