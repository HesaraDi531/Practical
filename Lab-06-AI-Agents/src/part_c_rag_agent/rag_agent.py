
# ============================================================
# PUSL 3205 - LARGE LANGUAGE MODELS AND GENERATIVE AI
# LAB SHEET 06
# PART C: RAG POWERED AI AGENT
# ============================================================

import ast
import operator
import sqlite3
import re

from pathlib import Path

from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings
)

from langchain_chroma import Chroma

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

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

RAG_DATABASE_PATH = r"D:\LLM\RAG_Project\chroma_db"

COLLECTION_NAME = "university_knowledge"

TOP_K = 4

BASE_DIR = Path(__file__).resolve().parent

MEMORY_DATABASE = (
    BASE_DIR.parent
    / "part_b_memory"
    / "agent_memory.db"
)


# ============================================================
# STEP 2: INITIALIZE OLLAMA LLM
# ============================================================

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)

print("[INFO] LLM initialized:", MODEL_NAME)


# ============================================================
# STEP 3: CONNECT TO EXISTING CHROMADB
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=RAG_DATABASE_PATH,
    embedding_function=embeddings
)

collection_count = vector_store._collection.count()

print("[INFO] ChromaDB connected.")

print("[INFO] Collection:", COLLECTION_NAME)

print("[INFO] Available chunks:", collection_count)


# ============================================================
# STEP 4: CALCULATOR TOOL
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


@tool
def calculator(expression: str) -> str:
    """
    Performs mathematical calculations.

    Example:
    25 * 4
    150 + 250
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
# STEP 5: MEMORY SEARCH TOOL
# ============================================================

@tool
def search_student_memory(query: str) -> str:
    """
    Searches previously saved user information
    from the Part B SQLite memory database.

    Use when the user asks about previously
    shared personal information.
    """

    if not MEMORY_DATABASE.exists():

        return (
            "Memory database does not exist. "
            "Complete Part B first."
        )

    connection = sqlite3.connect(
        MEMORY_DATABASE
    )

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

    keywords = re.findall(
        r"\w+",
        query_lower
    )

    stop_words = {
        "what", "is", "my", "the", "a",
        "an", "do", "i", "know", "tell",
        "me", "about", "remember", "please"
    }

    keywords = [
        word for word in keywords
        if word not in stop_words
    ]

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
        f"- {record[0]}"
        for record in matches
    )

    return "Matching memories:\n" + result


# ============================================================
# STEP 6: RAG SEARCH TOOL
# ============================================================

@tool
def search_university_knowledge(query: str) -> str:
    """
    Searches the university knowledge base
    using the existing Lab Sheet 5 ChromaDB.

    Use this tool for questions about university
    programmes, modules, subjects, academic
    information and university documents.
    """

    try:

        print(
            "\n[EXECUTING] Searching university knowledge:",
            query
        )

        documents = vector_store.similarity_search(
            query,
            k=TOP_K
        )

        if not documents:

            return (
                "No relevant university information "
                "was found in the knowledge base."
            )

        results = []

        for index, document in enumerate(
            documents,
            start=1
        ):

            metadata = document.metadata

            source = metadata.get(
                "source",
                "Unknown source"
            )

            page = metadata.get(
                "page",
                "Unknown page"
            )

            content = document.page_content

            results.append(

                f"""
                RESULT {index}

                SOURCE: {source}

                PAGE: {page}

                CONTENT:
                {content}
                """

            )

        return "\n".join(results)

    except Exception as error:

        return f"RAG search error: {error}"


# ============================================================
# STEP 7: REGISTER ALL THREE TOOLS
# ============================================================

tools = [

    calculator,

    search_student_memory,

    search_university_knowledge

]

tool_map = {

    "calculator": calculator,

    "search_student_memory": search_student_memory,

    "search_university_knowledge": search_university_knowledge

}

llm_with_tools = llm.bind_tools(tools)

print("\n[INFO] Registered tools:")

for item in tools:

    print("-", item.name)


# ============================================================
# STEP 8: SHORT-TERM CONVERSATION MEMORY
# ============================================================

messages = [

    SystemMessage(

        content="""
        You are an intelligent University AI Agent.

        You have access to three tools:

        1. calculator
           Performs mathematical calculations.

        2. search_student_memory
           Retrieves previously saved user information
           from the SQLite database.

        3. search_university_knowledge
           Retrieves university information from the
           existing university knowledge base.

        IMPORTANT INSTRUCTIONS:

        - Use the calculator tool for mathematical questions.
        - Use memory search for questions about previously
          shared user information.
        - Use RAG search for university-related questions.
        - Do not invent university information.
        - Base university answers on retrieved documents.
        - Mention the source and page when available.
        - If no relevant information is retrieved,
          clearly say that the knowledge base does not
          contain sufficient information.
        - Do not confuse user memories with university
          knowledge.
        - Give clear and friendly responses.
        """

    )

]


# ============================================================
# STEP 9: AGENT EXECUTION LOOP
# ============================================================

def run_agent(user_input):

    messages.append(
        HumanMessage(content=user_input)
    )

    for iteration in range(5):

        response = llm_with_tools.invoke(
            messages
        )

        messages.append(response)

        if not response.tool_calls:

            print("\n[AGENT RESPONSE]")

            print(response.content)

            return

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            arguments = tool_call["args"]

            call_id = tool_call["id"]

            print(
                "\n[TOOL SELECTED]",
                tool_name
            )

            print(
                "[TOOL ARGUMENTS]",
                arguments
            )

            if tool_name in tool_map:

                result = tool_map[
                    tool_name
                ].invoke(arguments)

            else:

                result = "Unknown tool."

            print("\n[TOOL RESULT]")

            print(result)

            messages.append(

                ToolMessage(
                    content=str(result),
                    tool_call_id=call_id
                )

            )

    print(
        "[WARNING] Maximum tool execution limit reached."
    )


# ============================================================
# STEP 10: MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)

    print("             RAG POWERED UNIVERSITY AI AGENT")

    print("=" * 65)

    print("LLM:", MODEL_NAME)

    print("Embedding Model:", EMBEDDING_MODEL)

    print("Vector Database: ChromaDB")

    print("Collection:", COLLECTION_NAME)

    print("Available Tools:")

    print("1. Calculator Tool")

    print("2. Memory Search Tool")

    print("3. RAG Search Tool")

    print("=" * 65)

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
