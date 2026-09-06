# tools/my_tools.py
import datetime
import pytz
from smolagents import tool

@tool
def calculate_discount(price: float, discount_percent: float) -> str:
    """
    Обчислює підсумкову ціну товару зі знижкою.

    Args:
        price: Початкова ціна товару.
        discount_percent: Відсоток знижки (наприклад, 20 для 20%).
    """
    final_price = price - (price * (discount_percent / 100))
    return f"Ціна зі знижкою {discount_percent}% становить: {final_price:.2f}"



@tool
def get_current_time(timezone: str) -> str:
    """
    Getts the current time in the specified timezone.
    Args:
        timezone: The timezone for which to get the current time (e.g., 'Europe/Kyiv', 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"Current time in {timezone}: {local_time}"
    except Exception as e:
        return f"Error with timezone '{timezone}': {str(e)}"


@tool
def sum_tool(a: float, b: float) -> str:
    """
    Returns the sum of two numbers.
    Args:
        a: First number.
        b: Second number.
    """
    return f"The sum of {a} and {b} is: {a + b}"