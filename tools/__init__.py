# tools/__init__.py   Init all tools
from tools.tools import calculate_discount, sum_tool ,get_current_time
from tools.math_tools import sum_tool, divide_tool, multiply_tool, subtract_tool    


# Form a list of all tools to be exported
ALL_TOOLS = [
    calculate_discount,
    sum_tool,
    divide_tool,
    multiply_tool,
    subtract_tool,
    get_current_time
]