from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from config import init_llm
from structured_output import parser


# ---------------- Инициализируем LLM -----------------

llm = init_llm()


# ---------------- Создаем шаблон промпта -----------------

prompt_template = PromptTemplate(
    template="""Проанализируй отзыв: {review}

{format_instructions}

ТОЛЬКО JSON!""",
    input_variables=["review"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# ---------------- Создаем цепочку -----------------

analysis_chain = prompt_template | llm | parser


# ---------------- Тестируем -----------------

review = "Товар отличный, быстрая доставка! Очень доволен покупкой."

result = analysis_chain.invoke({"review": review})

print("=== ЧЕРЕЗ ЦЕПОЧКУ ===")
print(f"Результат: {result}")