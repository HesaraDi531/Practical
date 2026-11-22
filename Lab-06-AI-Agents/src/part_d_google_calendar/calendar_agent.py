
# ============================================================
# PUSL 3205 - LAB SHEET 06
# PART D: GOOGLE CALENDAR AI AGENT
# ============================================================

import os
import json
from pathlib import Path

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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

BASE_DIR = Path(__file__).resolve().parent

CREDENTIALS_FILE = BASE_DIR / "credentials.json"

TOKEN_FILE = BASE_DIR / "token.json"

MODEL_NAME = "llama3.2:3b"

TIMEZONE = "Asia/Colombo"

TZ = ZoneInfo(TIMEZONE)

# Read and write access to the primary Google Calendar.
SCOPES = [
    "https://www.googleapis.com/auth/calendar"
]


# ============================================================
# STEP 2: INITIALIZE OLLAMA
# ============================================================

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)

print("[INFO] Ollama initialized:", MODEL_NAME)


# ============================================================
# STEP 3: GOOGLE OAUTH AUTHENTICATION
# ============================================================

def authenticate_google_calendar():

    if not CREDENTIALS_FILE.exists():

        raise FileNotFoundError(
            "credentials.json not found. "
            "Download your Desktop OAuth credentials."
        )

    credentials = None

    # Load existing authorization token.
    if TOKEN_FILE.exists():

        credentials = Credentials.from_authorized_user_file(
            str(TOKEN_FILE),
            SCOPES
        )

    # Refresh expired token or authenticate.
    if not credentials or not credentials.valid:

        if (
            credentials
            and credentials.expired
            and credentials.refresh_token
        ):

            print("[INFO] Refreshing Google access token.")

            credentials.refresh(Request())

        else:

            print("[INFO] Starting Google OAuth login.")

            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE),
                SCOPES
            )

            credentials = flow.run_local_server(
                port=0
            )

        # Save token for future use.
        with open(TOKEN_FILE, "w", encoding="utf-8") as token:

            token.write(credentials.to_json())

    print("[SUCCESS] Google authentication completed.")

    return credentials


# ============================================================
# STEP 4: CONNECT TO GOOGLE CALENDAR
# ============================================================

credentials = authenticate_google_calendar()

service = build(
    "calendar",
    "v3",
    credentials=credentials
)

print("[SUCCESS] Google Calendar connected.")


# ============================================================
# STEP 5: CHECK TODAY'S SCHEDULE TOOL
# ============================================================

@tool
def check_today_schedule() -> str:
    """
    Retrieves all events scheduled for today
    from the user's primary Google Calendar.

    Use this tool when the user asks:
    - What is my schedule today?
    - Show today's calendar events.
    - What meetings do I have today?
    """

    try:

        now = datetime.now(TZ)

        start_of_day = now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        end_of_day = start_of_day + timedelta(days=1)

        events_result = service.events().list(

            calendarId="primary",

            timeMin=start_of_day.isoformat(),

            timeMax=end_of_day.isoformat(),

            maxResults=100,

            singleEvents=True,

            orderBy="startTime"

        ).execute()

        events = events_result.get("items", [])

        if not events:

            return (
                "You have no scheduled events for today."
            )

        results = []

        for event in events:

            summary = event.get(
                "summary",
                "Untitled event"
            )

            start = event.get("start", {})

            end = event.get("end", {})

            start_time = start.get(
                "dateTime",
                start.get("date", "Unknown")
            )

            end_time = end.get(
                "dateTime",
                end.get("date", "Unknown")
            )

            results.append(

                f"""
                Event: {summary}
                Start: {start_time}
                End: {end_time}
                Description: {event.get('description', 'N/A')}
                """

            )

        return (
            f"Today's schedule ({now.strftime('%Y-%m-%d')}):\n"
            + "\n".join(results)
        )

    except HttpError as error:

        return f"Google Calendar API error: {error}"


# ============================================================
# STEP 6: CREATE NEW CALENDAR EVENT TOOL
# ============================================================

@tool
def create_calendar_event(
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str = ""
) -> str:
    """
    Creates a new event in the user's primary Google Calendar.

    Datetime format:
    YYYY-MM-DD HH:MM

    Example:
    title = Project Meeting
    start_datetime = 2026-11-20 10:00
    end_datetime = 2026-11-20 11:00

    Ask the user for confirmation before creating an event.
    """

    try:

        start = datetime.strptime(
            start_datetime,
            "%Y-%m-%d %H:%M"
        ).replace(tzinfo=TZ)

        end = datetime.strptime(
            end_datetime,
            "%Y-%m-%d %H:%M"
        ).replace(tzinfo=TZ)

        now = datetime.now(TZ)

        if start <= now:

            return (
                "Cannot create an event in the past. "
                "Please provide a future start time."
            )

        if end <= start:

            return (
                "End time must be later than start time."
            )

        event = {

            "summary": title,

            "description": description,

            "start": {

                "dateTime": start.isoformat(),

                "timeZone": TIMEZONE

            },

            "end": {

                "dateTime": end.isoformat(),

                "timeZone": TIMEZONE

            }

        }

        created_event = service.events().insert(

            calendarId="primary",

            body=event

        ).execute()

        return (

            "EVENT CREATED SUCCESSFULLY\n"

            f"Title: {created_event.get('summary')}\n"

            f"Start: {created_event['start'].get('dateTime')}\n"

            f"End: {created_event['end'].get('dateTime')}\n"

            f"Event ID: {created_event.get('id')}\n"

            f"Google Calendar Link: {created_event.get('htmlLink')}"

        )

    except ValueError:

        return (
            "Invalid datetime format. "
            "Use YYYY-MM-DD HH:MM."
        )

    except HttpError as error:

        return f"Google Calendar API error: {error}"


# ============================================================
# STEP 7: REGISTER GOOGLE CALENDAR TOOLS
# ============================================================

tools = [

    check_today_schedule,

    create_calendar_event

]

tool_map = {

    "check_today_schedule": check_today_schedule,

    "create_calendar_event": create_calendar_event

}

llm_with_tools = llm.bind_tools(tools)

print("\n[INFO] Registered Google Calendar tools:")

for item in tools:

    print("-", item.name)


# ============================================================
# STEP 8: AGENT CONVERSATION MEMORY
# ============================================================

messages = [

    SystemMessage(

        content=f"""
        You are an intelligent Google Calendar AI assistant.

        Current timezone: {TIMEZONE}

        Available tools:

        1. check_today_schedule
           Retrieves today's calendar events.

        2. create_calendar_event
           Creates a calendar event.

        Instructions:

        - Use the schedule tool when asked about today's events.
        - Use the event creation tool when asked to schedule something.
        - Always obtain the event title, date, start time and end time.
        - If any required information is missing, ask the user.
        - Before creating an event, summarize the details and
          ask the user for explicit confirmation.
        - Never claim an event was created unless the tool
          successfully confirms it.
        - Use {TIMEZONE} for all event times.
        - Be clear and friendly.
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

        response = llm_with_tools.invoke(messages)

        messages.append(response)

        if not response.tool_calls:

            print("\n[AGENT RESPONSE]")

            print(response.content)

            return

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            arguments = tool_call["args"]

            call_id = tool_call["id"]

            print("\n[TOOL SELECTED]", tool_name)

            print("[TOOL ARGUMENTS]", arguments)

            # Require confirmation before creating events.
            if tool_name == "create_calendar_event":

                print("\n[EVENT DETAILS]")

                print(
                    json.dumps(
                        arguments,
                        indent=4
                    )
                )

                confirmation = input(
                    "\nDo you approve creating this event? (yes/no): "
                ).strip().lower()

                if confirmation not in ["yes", "y"]:

                    result = (
                        "Event creation cancelled by the user. "
                        "No event was created."
                    )

                else:

                    result = tool_map[
                        tool_name
                    ].invoke(arguments)

            elif tool_name in tool_map:

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

    print("[WARNING] Maximum tool execution limit reached.")


# ============================================================
# STEP 10: MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)

    print("             GOOGLE CALENDAR AI AGENT")

    print("=" * 65)

    print("LLM:", MODEL_NAME)

    print("Timezone:", TIMEZONE)

    print("Available Tools:")

    print("1. Check Today's Schedule")

    print("2. Create New Calendar Event")

    print("=" * 65)

    print("\nType 'exit' to stop.")

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
