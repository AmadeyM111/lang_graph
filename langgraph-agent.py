# pip install langgraph

from typing import TypedDict
from datetime import date, timedelta
from langgraph.graph import StateGraph, START, END


class UserState(TypedDict):
    name: str
    surname: str
    age: int
    birth_date: date
    today: date  # simulated current date (increments each iteration)
    message: str


# --- Nodes ---

def calculate_age(state: UserState) -> dict:
    """Calculate exact age based on simulated today's date."""
    today = state["today"]
    age = today.year - state["birth_date"].year
    # Subtract 1 if birthday hasn't occurred yet this year
    if (today.month, today.day) < (state["birth_date"].month, state["birth_date"].day):
        age -= 1
    return {"age": age}


def autoincrement_date(state: UserState) -> dict:
    """Advance simulated date by one day."""
    current_date = state["today"]
    new_date = current_date + timedelta(days=1)
    print(f"{current_date} -> {new_date}")
    return {"today": new_date}


def generate_success_message(state: UserState) -> dict:
    return {
        "message": (
            f"Congrats, {state['name']} {state['surname']}! "
            f"You are already {state['age']} years old and you can drive!"
        )
    }


# --- Routing ---

def check_drive(state: UserState) -> str:
    """Return routing key based on whether the person can drive."""
    if state["age"] >= 18:
        return "allowed"
    else:
        return "forbidden"


# --- Graph ---

graph = StateGraph(UserState)

graph.add_node("calculate_age", calculate_age)
graph.add_node("generate_success_message", generate_success_message)
graph.add_node("autoincrement_date", autoincrement_date)

graph.add_edge(START, "calculate_age")

graph.add_conditional_edges(
    "calculate_age",
    check_drive,
    {
        "allowed": "generate_success_message",
        "forbidden": "autoincrement_date",
    }
)

graph.add_edge("generate_success_message", END)
graph.add_edge("autoincrement_date", "calculate_age")

# --- Compile & Run ---

app = graph.compile()

# Note: recursion_limit counts node visits, not loop iterations.
# Each "day" visits ~3 nodes, so for up to ~600 days gap set limit >= 2000.
result = app.invoke(
    {
        "name": "Алексей",
        "surname": "Яковенко",
        "age": 0,
        "birth_date": date.fromisoformat("2008-02-19"),
        "today": date.today(),
        "message": "",
    },
    {"recursion_limit": 2000},
)

print(result["message"])
print(f"Date when allowed to drive: {result['today']}")
