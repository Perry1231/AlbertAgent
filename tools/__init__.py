# tools/__init__.py   Init all tools
from tools.bash_tools import install_pip_package, run_bash_command
from tools.file_tools import read_json_file, save_text_to_file
from tools.tools import calculate_discount ,get_current_time
from tools.math_tools import sum_tool, divide_tool, multiply_tool, subtract_tool
from tools.web_tool import visit_webpage, smart_web_scraper
from tools.code_execution import execute_python_code
from tools.api_tools import get_crypto_price, fetch_json_api, get_weather_forecast   
from tools.image_tools import process_image_tool    
from tools.audio_tools import transcribe_audio_tool
from tools.video_tools import extract_video_frames_tool
from tools.bash_tools import execute_bash_command
from smolagents import tool



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
    save_text_to_file,
    smart_web_scraper,
    transcribe_audio_tool,
    process_image_tool,
    extract_video_frames_tool,
    execute_bash_command,
    install_pip_package

]