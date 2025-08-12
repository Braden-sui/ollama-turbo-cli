"""Calculator tool plugin with safe expression evaluator"""
from __future__ import annotations

import math
import re

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate_math",
        "description": "Evaluate mathematical expressions including basic operations and functions like sin, cos, sqrt, log, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sin(pi/2)', 'sqrt(16)')"
                }
            },
            "required": ["expression"]
        }
    }
}

def calculate_math(expression: str) -> str:
    """Safe mathematical expression evaluator with comprehensive operations."""
    try:
        # Remove whitespace
        expression_orig = expression
        expression = expression.strip()
        
        # Define allowed characters for initial check
        basic_pattern = r'^[0-9+\-*/^().\s]+$'
        
        # Process the expression - handle special functions and constants
        expression_clean = expression.lower()
        
        # Replace math constants
        expression_clean = expression_clean.replace('pi', str(math.pi))
        expression_clean = expression_clean.replace('e', str(math.e))
        
        # Handle math functions with proper Python syntax
        expression_clean = re.sub(r'sqrt\(([^)]+)\)', r'math.sqrt(\1)', expression_clean)
        expression_clean = re.sub(r'sin\(([^)]+)\)', r'math.sin(\1)', expression_clean)
        expression_clean = re.sub(r'cos\(([^)]+)\)', r'math.cos(\1)', expression_clean)
        expression_clean = re.sub(r'tan\(([^)]+)\)', r'math.tan(\1)', expression_clean)
        expression_clean = re.sub(r'log\(([^)]+)\)', r'math.log(\1)', expression_clean)
        expression_clean = re.sub(r'log10\(([^)]+)\)', r'math.log10(\1)', expression_clean)
        expression_clean = re.sub(r'exp\(([^)]+)\)', r'math.exp(\1)', expression_clean)
        expression_clean = re.sub(r'abs\(([^)]+)\)', r'abs(\1)', expression_clean)
        expression_clean = re.sub(r'pow\(([^,]+),([^)]+)\)', r'pow(\1,\2)', expression_clean)
        
        # Replace ^ with ** for Python power operator
        expression_clean = expression_clean.replace('^', '**')
        
        # Check if only allowed characters remain after removing math. prefix
        test_expr = expression_clean.replace('math.', '').replace('abs', '').replace('pow', '')
        if not re.match(basic_pattern, test_expr):
            return f"Error: Invalid characters in expression '{expression_orig}'. Allowed: numbers, +, -, *, /, ^, (), ., sin, cos, tan, sqrt, log, exp, pi, e, abs, pow"
        
        # Evaluate the expression safely using eval with restricted namespace
        safe_dict = {
            "math": math,
            "abs": abs,
            "pow": pow,
            "__builtins__": {}
        }
        
        result = eval(expression_clean, safe_dict)
        
        # Format result nicely
        if isinstance(result, float):
            if abs(result - round(result)) < 1e-10:
                result = int(round(result))
            else:
                result = round(result, 8)
        
        return f"Result: {expression_orig} = {result}"
        
    except ZeroDivisionError:
        return f"Error: Division by zero in expression '{expression}'"
    except ValueError as e:
        return f"Error: Invalid mathematical operation in '{expression}': {str(e)}"
    except SyntaxError:
        return f"Error: Invalid syntax in expression '{expression}'"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


TOOL_IMPLEMENTATION = calculate_math
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
