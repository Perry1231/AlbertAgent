from smolagents import tool
@tool
def sum_tool(a: float, b: float) -> str:
    """
    Returns the sum of two numbers.
    Args:
        a: First number.
        b: Second number.
    """
    return f"The sum of {a} and {b} is: {a + b}"

@tool
def divide_tool(a: float, b: float) -> str:
    """
    Returns the division of two numbers.
    Args:
        a: Numerator.
        b: Denominator.
    """
    if b == 0:
        return "Error: Division by zero is not allowed."
    return f"The result of {a} divided by {b} is: {a / b}"

@tool
def multiply_tool(a: float, b: float) -> str:
    """
    Returns the product of two numbers.
    Args:
        a: First number.
        b: Second number.
    """
    return f"The product of {a} and {b} is: {a * b}"

@tool
def subtract_tool(a: float, b: float) -> str:
    """
    Returns the difference of two numbers.
    Args:
        a: Minuend.
        b: Subtrahend.
    """
    return f"The result of {a} minus {b} is: {a - b}"