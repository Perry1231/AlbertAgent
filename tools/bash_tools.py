import subprocess
from smolagents import tool

@tool
def run_bash_command(command: str) -> str:
    """Runs a Bash command in the terminal and returns stdout or stderr.

    Args:
        command: The specific console command to execute (e.g., 'ls -la', 'git status').
    """
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout if result.stdout else "Command executed without output."
        return f"Error (code {result.returncode}): {result.stderr}"
    except Exception as e:
        return f"Error executing command: {str(e)}"