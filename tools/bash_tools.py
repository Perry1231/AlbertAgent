import subprocess
from smolagents import tool
import sys

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



@tool
def execute_bash_command(command: str) -> str:
    """Виконує Bash-команду в системній консолі та повертає stdout або stderr.
    Підходить для запуску CLI-утиліт (ffmpeg, curl, git, pandoc), роботи з архівами чи файлами.

    Args:
        command: Конкретна консольна команда для виконання (наприклад, 'ffmpeg -i input.mp4 output.mp3').
    """
    try:
        # Виконуємо команду з таймаутом у 60 секунд, щоб уникнути зависання
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = []
        if result.stdout.strip():
            output.append(f"--- STDOUT ---\n{result.stdout.strip()}")
        if result.stderr.strip():
            output.append(f"--- STDERR ---\n{result.stderr.strip()}")
            
        if not output:
            return f"Команда виконана успішно з кодом повернення {result.returncode} (порожній вивід)."
            
        return "\n".join(output)
        
    except subprocess.TimeoutExpired:
        return "Помилка: Час виконання Bash-команди перевищив ліміт у 60 секунд."
    except Exception as e:
        return f"Помилка при виконанні команди: {str(e)}"



@tool
def install_pip_package(package_name: str) -> str:
    """Динамічно встановлює Python-пакет через pip прямо під час виконання завдання.

    Args:
        package_name: Назва пакета для встановлення (наприклад, 'pdfplumber' або 'yt-dlp').
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            return f"Пакети '{package_name}' успішно встановлено."
        return f"Помилка встановлення пакетів:\n{result.stderr}"
    except Exception as e:
        return f"Не вдалося встановити пакет: {str(e)}"