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

def check_age(state: UserState) -> str:
    """ conditional function """
    return "совершеннолетний" if state["age"] >= 18 else "не_совершеннолетний"

def generate_succes_message(state: UserState) -> dict:
    """Генерирует сообщение для совершеннолетних"""
    return {message: f"Вам уже {state['age']} лет и вы можете водить!"}

def generate_failure_message(state: UserState) -> dict:
    """Генерирует сообщение для несовершеннолетних"""
    return {"message": f"Вам еще только {state['age']} лет и вы не можете водить."}

# Create the fiction node

graph = StateGraph(UserState)

# the fiction node - just passes the state it on 
graph.add_node("fake_node", lambda state: state)

# the main processing nodes 
graph.add_node("generate_success_message", generate_success_message)
graph.add_node("generate_failure_message", generate_failure_message)

# the logic of graph

graph.add_edge(START, "fake_node")

graph.add_conditianal_edges(
    "fake_node",
    check_age,
    {
        "совершеннолетний": "generate_success_message",
        "не_совершеннолетний": "generate_failure_message"
    }
)

graph.add_edge("generate_succces_message", END)
graph.add_edge("generate_failure_message", END)

app = graph.compile()

result_minor = app.invoke({"age": 17})
print("Результат для 17 лет:", result_minor)

result_adult = app.invoke({"age": 25})
print("Результат для 25 лет:", result_adult)

# Logging
def log_and_pass(state: UserState) -> UserState:
    """Логирует вход в граф и передает состояние дальше"""
    print(f"Начинаем обработку пользователя с возрастом: {state['age']}")
    return.add_node("log_node", log_and_pass)

graph.add_node("log_node", log_and_pass)

# Validation
def validate_and_pass(state: UserState) -> UserState:
    """Проверка корректности данных и передача состояния дальше"""
    if state["age"] < 0 or state["age"] > 150:
        raise ValueError(f"Некорректный возраст: {state['age']}")
    return state

graph.add_node("validation_node", validate_and_pass)

def initialize_and_pass(state: UserState) -> dict:
    """Inizialize additional fields and pass the state"""
    return {
        "timestamp": datetime.now().isoformat(),
        "processed": True
    }

    graph.add_node("init_node", initialize_and_pass)

    def gen_png_graph(app_obj, name_photo: str = "graph.png") -> None:
        """
        Генерирует PNG-изображение графа и сохраняет его в файл.
    
         Args:
            app_obj: Скомпилированный объект графа
            name_photo: Имя файла для сохранения (по умолчанию "graph.png")
        """
        with open(name_photo, "wb") as f:
            f.write(app_obj.get_graph().draw_mermaid_png())

        app = graph.compile()

        gen_png_graph(app, name_photo="graph_example_4.png")

        print("Граф сохранен как graph_example_4.png")