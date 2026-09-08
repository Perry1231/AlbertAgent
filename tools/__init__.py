# tools/__init__.py   Init all tools
from tools.bash_tools import run_bash_command
from tools.file_tools import read_json_file, save_text_to_file
from tools.tools import calculate_discount ,get_current_time
from tools.math_tools import sum_tool, divide_tool, multiply_tool, subtract_tool
from tools.web_tool import visit_webpage
from tools.code_execution import execute_python_code
from tools.api_tools import get_crypto_price, fetch_json_api, get_weather_forecast   



# Form a list of all tools to be exported
ALL_TOOLS = [
    calculate_discount,
    sum_tool,
    divide_tool,
    multiply_tool,
    subtract_tool,
    get_current_time,
    visit_webpage,
    execute_python_code,
    get_crypto_price,
    fetch_json_api,
    get_weather_forecast,
    run_bash_command,
    read_json_file,
    save_text_to_file

]