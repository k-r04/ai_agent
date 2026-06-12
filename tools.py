from langchain_core.tools import tool
from datetime import datetime

@tool
def get_current_time(query: str) -> str:
    """Returns the current date and time."""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def calculator(expression: str) -> str:
    """Evaluates a math expression. Example: '2 + 2' or '10 * 5'"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

tools = [get_current_time, calculator]