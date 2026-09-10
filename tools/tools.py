from datetime import datetime
from smolagents import tool

@tool
def get_current_time() -> str:
    """Returns the current UTC time as a formatted string.

    Returns:
        str: Current UTC date and time in YYYY-MM-DD HH:MM:SS format.
    """
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate_discount(price: float, discount_percent: float) -> str:
    """Calculates the final price of an item after applying a percentage discount.

    Args:
        price: The original price of the item.
        discount_percent: The percentage discount to apply (e.g., 20 for 20%).

    Returns:
        str: Formatted sentence stating the final price after discount.
    """
    final_price = price - (price * (discount_percent / 100))
    return f"Final price with a {discount_percent}% discount is: {final_price:.2f}"