
# ============================================================
# PUSL 3205 - LARGE LANGUAGE MODELS AND GENERATIVE AI
# LAB SHEET 06
# PART B: GIVE YOUR AI AGENT MEMORY
# ============================================================

import sqlite3
import ast
import operator
import re

from datetime import datetime

from langchain_ollama import ChatOllama

from langchain_core.tools import tool

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

MODEL_NAME = "llama3.2:3b"

DATABASE = "agent_memory.db"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)

print("[INFO] Ollama model initialized:", MODEL_NAME)


# ============================================================
# STEP 2: SQLITE DATABASE
# LONG-TERM MEMORY
# ============================================================

def initialize_database():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            fact TEXT NOT NULL UNIQUE,

            created_at TEXT NOT NULL

        )
    """)

    connection.commit()

    connection.close()

    print("[INFO] SQLite memory database initialized.")


# ============================================================
# STEP 3: SAVE INFORMATION INTO LONG-TERM MEMORY
# ============================================================

def save_memory(fact):

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO memories
        (fact, created_at)
        VALUES (?, ?)
    """, (
        fact,
        datetime.now().isoformat(timespec="seconds")
    ))

    connection.commit()

    saved = cursor.rowcount > 0

    connection.close()

    if saved:
        print("[MEMORY] Saved:", fact)
    else:
        print("[MEMORY] Information already exists.")

    return saved


# ============================================================
# STEP 4: DETECT USEFUL USER INFORMATION
# ============================================================

def detect_user_information(user_input):

    text = user_input.strip()

    patterns = [

        (
            r"\bmy name is\s+(.+?)[.!?]*$",
            "User's name is {}"
        ),

        (
            r"\bi am\s+(\d{1,3})\s+years?\s+old\b",
            "User is {} years old"
        ),

        (
            r"\bi study\s+(.+?)[.!?]*$",
            "User studies {}"
        ),

        (
            r"\bi am studying\s+(.+?)[.!?]*$",
            "User is studying {}"
        ),

        (
            r"\bmy favorite (?:subject|color|food|sport) is\s+(.+?)[.!?]*$",
            "User's favorite choice is {}"
        ),

        (
            r"\bi live in\s+(.+?)[.!?]*$",
            "User lives in {}"
        )

    ]

    for pattern, template in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            if value:
                fact = template.format(value)

                save_memory(fact)


# ============================================================
# STEP 5: MEMORY SEARCH TOOL
# ============================================================

@tool
def search_student_memory(query: str) -> str:
    """
    Search previously stored user information in SQLite.

    Use this tool whenever the user asks about something
    they previously told the agent.
    """

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT fact, created_at
        FROM memories
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    connection.close()

    if not records:
        return "No stored memories found."

    query_lower = query.lower()

    # Extract useful search terms
    keywords = re.findall(
        r"\w+",
        query_lower
    )

    # Remove common question words
    stop_words = {
        "what", "is", "my", "the", "a",
        "an", "do", "i", "know", "tell",
        "me", "about", "remember", "please"
    }

    keywords = [
        word for word in keywords
        if word not in stop_words
    ]

    # Special handling for name questions
    if "name" in query_lower:

        matches = [
            record for record in records
            if "name" in record[0].lower()
        ]

    else:

        matches = []

        for record in records:

            fact = record[0].lower()

            if any(
                keyword in fact
                for keyword in keywords
            ):

                matches.append(record)

    if not matches:

        return "No matching memories found."

    result = "\n".join(
        [
            f"- {record[0]}"
            for record in matches
        ]
    )

    return "Matching memories:\n" + result


# ============================================================
# STEP 6: SAFE CALCULATOR FUNCTION
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

    tree = ast.parse(
        expression,
        mode="eval"
    )

    def evaluate(node):

        if isinstance(node, ast.Expression):

            return evaluate(node.body)

        elif isinstance(node, ast.Constant):

            if type(node.value) in (int, float):
                return node.value

        elif isinstance(node, ast.BinOp):

            operation = OPERATORS.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError("Unsupported operator")

            return operation(
                evaluate(node.left),
                evaluate(node.right)
            )

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
# STEP 7: CALCULATOR TOOL
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Performs mathematical calculations.

    Examples:
    25 * 4
    100 + 200
    (25 + 15) * 4
    """

    try:

        result = safe_calculate(expression)

        return (
            f"Expression: {expression}\n"
            f"Result: {result}"
        )

    except Exception as error:

        return f"Calculation error: {error}"


# ============================================================
# STEP 8: REGISTER AGENT TOOLS
# ============================================================

tools = [

    calculator,
    search_student_memory

]

tool_map = {

    "calculator": calculator,
    "search_student_memory": search_student_memory

}

llm_with_tools = llm.bind_tools(tools)

print("[INFO] Calculator tool registered.")
print("[INFO] Memory search tool registered.")


# ============================================================
# STEP 9: SHORT-TERM MEMORY
# CONVERSATION HISTORY
# ============================================================

messages = [

    SystemMessage(

        content="""
        You are an intelligent AI assistant with memory.

        You have access to two tools:

        1. calculator:
           Performs mathematical calculations.

        2. search_student_memory:
           Retrieves previously stored user information
           from the SQLite database.

        IMPORTANT INSTRUCTIONS:

        - Remember the current conversation.
        - Use the memory search tool when the user asks
          about previously shared personal information.
        - Never invent stored information.
        - If memory search finds no matching information,
          clearly say that you do not have that information.
        - Use the calculator tool for mathematical questions.
        - Give friendly and clear responses.
        """

    )

]


# ============================================================
# STEP 10: AGENT EXECUTION LOOP
# ============================================================

def run_agent(user_input):

    # Detect and save useful information
    detect_user_information(user_input)

    # Add user message to short-term memory
    messages.append(
        HumanMessage(content=user_input)
    )

    # Execute agent
    for iteration in range(5):

        response = llm_with_tools.invoke(messages)

        messages.append(response)

        # No tool requested
        if not response.tool_calls:

            print("\n[AGENT RESPONSE]")
            print(response.content)

            return

        # Execute selected tools
        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            arguments = tool_call["args"]

            call_id = tool_call["id"]

            print("\n[TOOL SELECTED]", tool_name)

            print("[TOOL ARGUMENTS]", arguments)

            if tool_name in tool_map:

                result = tool_map[
                    tool_name
                ].invoke(arguments)

            else:

                result = "Unknown tool."

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
# STEP 11: MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    initialize_database()

    print("\n" + "=" * 60)

    print("          INTELLIGENT STUDENT AI AGENT")

    print("=" * 60)

    print("SHORT-TERM MEMORY: ACTIVE")

    print("LONG-TERM SQLITE MEMORY: ACTIVE")

    print("WORKING MEMORY: ACTIVE")

    print("=" * 60)

    print("\nType 'exit' to stop the agent.")

    while True:

        try:

            user_input = input("\nYou: ").strip()

            if user_input.lower() in [
                "exit",
                "quit"
            ]:

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
