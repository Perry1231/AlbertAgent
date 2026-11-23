from smolagents import tool

@tool
def run_bash_command(command: str) -> str:
    """Executes a bash command and returns stdout or stderr.

    Args:
        command: The bash command string to execute.

    Returns:
        str: Command output or error details.
    """
    import subprocess

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return f"Error executing command: {str(e)}"

@tool
def execute_bash_command(command: str) -> str:
    """Executes a system shell command safely.

    Args:
        command: Command to execute.

    Returns:
        str: Output of the command.
    """
    import subprocess

    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True, timeout=30)
        return output
    except subprocess.CalledProcessError as e:
        return e.output
    except Exception as e:
        return str(e)

@tool
def install_pip_package(package_name: str) -> str:
    """Installs a python package via pip.

    Args:
        package_name: Name of the package to install.

    Returns:
        str: Installation status.
    """
    import subprocess

    try:
        subprocess.check_call([subprocess.sys.executable, "-m", "pip", "install", package_name])
        return f"Package '{package_name}' installed successfully."
    except Exception as e:
        return f"Failed to install '{package_name}': {str(e)}"