
# ============================================================
# PUSL 3205 - LAB SHEET 06
# PART A: BUILDING A CALCULATOR AI AGENT
# ============================================================

import ast
import operator
import subprocess
import time

import pyautogui
import pygetwindow as gw

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)


# ============================================================
# STEP 1: INITIALIZE THE LLM
# ============================================================

MODEL_NAME = "llama3.2:3b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)

print("[INFO] Ollama model initialized:", MODEL_NAME)


# ============================================================
# STEP 2: CREATE A SAFE CALCULATOR FUNCTION
# ============================================================

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv
}


def safe_calculate(expression):

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):

        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        elif isinstance(node, ast.Constant):

            if type(node.value) in (int, float):
                return node.value

        elif isinstance(node, ast.BinOp):

            operation = OPERATORS.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported operator")

            left = evaluate(node.left)
            right = evaluate(node.right)

            return operation(left, right)

        elif isinstance(node, ast.UnaryOp):

            if isinstance(node.op, ast.USub):
                return -evaluate(node.operand)

            if isinstance(node.op, ast.UAdd):
                return evaluate(node.operand)

        raise ValueError("Invalid mathematical expression")

    result = evaluate(tree)

    if isinstance(result, complex):
        raise ValueError("Complex results are not supported")

    return result


# ============================================================
# STEP 3: OPEN WINDOWS CALCULATOR
# ============================================================

def open_windows_calculator(expression):

    subprocess.Popen("calc.exe")

    time.sleep(2)

    windows = gw.getWindowsWithTitle("Calculator")

    if not windows:
        return "Calculator window not found."

    calculator = windows[0]

    if calculator.isMinimized:
        calculator.restore()

    calculator.activate()

    time.sleep(1)

    # Clear the calculator
    pyautogui.press("esc")

    # Enter the expression
    pyautogui.write(expression, interval=0.08)

    pyautogui.press("enter")

    return "Windows Calculator opened and expression entered."


# ============================================================
# STEP 4: CREATE CALCULATOR TOOL
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Performs mathematical calculations and opens Windows Calculator.

    Examples:
    25 * 4
    150 + 250
    500 / 25
    (25 + 15) * 4
    """

    try:

        result = safe_calculate(expression)

        try:
            gui_result = open_windows_calculator(expression)

        except Exception as gui_error:
            gui_result = str(gui_error)

        return (
            f"Expression: {expression}\n"
            f"Calculated result: {result}\n"
            f"Windows Calculator: {gui_result}"
        )

    except Exception as error:

        return f"Calculation failed: {error}"


# ============================================================
# STEP 5: REGISTER TOOL WITH LLM
# ============================================================

tools = [calculator]

tool_map = {
    "calculator": calculator
}

llm_with_tools = llm.bind_tools(tools)

print("[INFO] Calculator tool registered successfully.")


# ============================================================
# STEP 6: AGENT EXECUTION LOOP
# ============================================================

messages = [

    SystemMessage(
        content="""
        You are an intelligent calculator AI agent.

        Your responsibilities:

        1. Understand mathematical questions.
        2. Use the calculator tool whenever a calculation is required.
        3. Convert the mathematical question into a valid expression.
        4. Never guess calculation results.
        5. Return the final answer clearly.
        """
    )

]


def run_agent(user_input):

    messages.append(
        HumanMessage(content=user_input)
    )

    for iteration in range(5):

        response = llm_with_tools.invoke(messages)

        messages.append(response)

        # Check whether the LLM requested a tool
        if not response.tool_calls:

            print("\n[AGENT RESPONSE]")
            print(response.content)

            return

        # Execute requested tools
        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            arguments = tool_call["args"]

            call_id = tool_call["id"]

            print("\n[TOOL SELECTED]", tool_name)

            print("[TOOL ARGUMENTS]", arguments)

            if tool_name in tool_map:

                result = tool_map[tool_name].invoke(arguments)

            else:

                result = "Tool not found."

            print("[TOOL RESULT]")
            print(result)

            messages.append(

                ToolMessage(
                    content=str(result),
                    tool_call_id=call_id
                )

            )

    print("[WARNING] Maximum tool execution limit reached.")


# ============================================================
# STEP 7: MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("             CALCULATOR AI AGENT")
    print("=" * 60)

    print("LLM:", MODEL_NAME)

    print("Available tool: Calculator")

    print("\nType 'exit' to stop the agent.")

    while True:

        try:

            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["exit", "quit"]:

                print("\n[INFO] Agent stopped.")

                break

            if not user_input:
                continue

            run_agent(user_input)

        except KeyboardInterrupt:

            print("\n[INFO] Agent stopped.")

            break

        except Exception as error:

            print("\n[ERROR]", error)
