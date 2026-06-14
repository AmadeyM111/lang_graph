# Техническое задание: развитие структуры LangGraph-проекта

## 1. Назначение документа

Документ описывает целевую структуру репозитория для большого проекта по созданию ИИ-агентов на LangGraph.

Проект включает несколько направлений:

- общие концепции LangGraph;
- работа с состоянием, сообщениями и графами;
- tools и tool calling;
- ReAct-паттерн;
- MAS, то есть multi-agent systems;
- агенты-специалисты;
- приложения, демонстрации, тесты и evals.

Цель реорганизации: привести репозиторий к профессиональной, расширяемой и понятной структуре, где учебные материалы, переиспользуемый код, приложения и эксперименты разделены по зонам ответственности.

## 2. Основные принципы структуры

### 2.1. Разделение библиотеки и примеров

Основная логика должна находиться в `src/`.

Учебные и демонстрационные сценарии должны находиться в `examples/` и импортировать код из `src/`, а не дублировать его.

### 2.2. Минимальный корень репозитория

В корне проекта следует оставить только инфраструктурные файлы:

- `README.md`;
- `pyproject.toml`;
- `.env.example`;
- `.gitignore`;
- файлы конфигурации инструментов разработки.

Рабочие Python-файлы не должны лежать в корне репозитория.

### 2.3. Отсутствие дублирования

Необходимо убрать дубли между корнем проекта и директорией `main_concepts`.

Один концепт должен иметь одно каноническое место:

- библиотечная реализация находится в `src/`;
- учебный пример находится в `examples/`;
- объяснение находится в `docs/`.

### 2.4. Явные границы ответственности

Каждая директория должна отвечать за отдельный слой системы:

- `core/` - базовые сущности;
- `graphs/` - LangGraph-графы;
- `nodes/` - переиспользуемые node-функции;
- `tools/` - инструменты агентов;
- `agents/` - агенты-специалисты;
- `mas/` - multi-agent orchestration;
- `memory/` - стратегии памяти и управления контекстом;
- `apps/` - точки запуска;
- `evals/` - оценка качества агентных сценариев.

## 3. Целевая структура репозитория

```text
langgraph-agents/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── docs/
│   ├── 01-langgraph-concepts.md
│   ├── 02-state-and-messages.md
│   ├── 03-tools.md
│   ├── 04-multi-agent-systems.md
│   ├── 05-specialist-agents.md
│   └── architecture.md
│
├── src/
│   └── langgraph_agents/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── config.py
│       │   ├── llm_factory.py
│       │   ├── state.py
│       │   ├── messages.py
│       │   └── graph_utils.py
│       │
│       ├── graphs/
│       │   ├── assistant_graph.py
│       │   ├── routing_graph.py
│       │   ├── multimodel_graph.py
│       │   └── react_graph.py
│       │
│       ├── nodes/
│       │   ├── expert_consensus.py
│       │   ├── version_control.py
│       │   ├── classification.py
│       │   └── sentiment.py
│       │
│       ├── tools/
│       │   ├── web.py
│       │   ├── files.py
│       │   ├── database.py
│       │   ├── quotes.py
│       │   └── registry.py
│       │
│       ├── memory/
│       │   ├── short_term.py
│       │   ├── long_term.py
│       │   └── context_strategy.py
│       │
│       ├── agents/
│       │   ├── base.py
│       │   ├── researcher.py
│       │   ├── coder.py
│       │   ├── analyst.py
│       │   └── supervisor.py
│       │
│       └── mas/
│           ├── orchestrator.py
│           ├── supervisor.py
│           ├── handoff.py
│           └── state.py
│
├── examples/
│   ├── 01_basic_graph.py
│   ├── 02_messages.py
│   ├── 03_structured_output.py
│   ├── 04_tool_calling.py
│   ├── 05_react_agent.py
│   ├── 06_multi_model_dialog.py
│   ├── 07_multi_agent_system.py
│   └── 08_specialist_agents.py
│
├── apps/
│   ├── cli/
│   │   └── main.py
│   ├── api/
│   │   └── main.py
│   └── playground/
│       └── main.py
│
├── configs/
│   ├── default.yaml
│   ├── local.example.yaml
│   └── agents/
│       ├── researcher.yaml
│       ├── coder.yaml
│       └── analyst.yaml
│
├── prompts/
│   ├── system/
│   ├── agents/
│   └── tools/
│
├── evals/
│   ├── datasets/
│   ├── scenarios/
│   └── run_eval.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── scripts/
    ├── run_agent.py
    ├── run_graph.py
    └── inspect_state.py
```

## 4. Назначение ключевых директорий

### 4.1. `src/langgraph_agents/core/`

Базовый слой проекта.

Сюда следует поместить:

- настройки проекта;
- фабрику LLM;
- базовые state-модели;
- работу с сообщениями;
- общие утилиты для сборки графов;
- общие типы и протоколы.

Примеры файлов:

- `config.py`;
- `llm_factory.py`;
- `state.py`;
- `messages.py`;
- `graph_utils.py`.

### 4.2. `src/langgraph_agents/graphs/`

Готовые LangGraph-графы.

Один файл должен описывать один граф или одну близкую группу графов.

Примеры:

- `assistant_graph.py`;
- `routing_graph.py`;
- `multimodel_graph.py`;
- `react_graph.py`.

### 4.3. `src/langgraph_agents/nodes/`

Переиспользуемые node-функции.

Node-функции не должны содержать код запуска приложения, CLI, HTTP-сервера или демонстрационных сценариев.

Примеры:

- `expert_consensus.py`;
- `version_control.py`;
- `classification.py`;
- `sentiment.py`.

### 4.4. `src/langgraph_agents/tools/`

Инструменты, которые могут вызываться агентами.

Рекомендуется иметь `registry.py`, где собираются наборы инструментов для разных агентов или сценариев.

Примеры:

- `web.py`;
- `files.py`;
- `database.py`;
- `quotes.py`;
- `registry.py`.

### 4.5. `src/langgraph_agents/memory/`

Логика работы с памятью и контекстом.

Сюда относятся:

- краткосрочная память;
- долгосрочная память;
- стратегии сжатия контекста;
- управление историей сообщений;
- политики сохранения и извлечения контекста.

### 4.6. `src/langgraph_agents/agents/`

Высокоуровневые агенты-специалисты.

Каждый агент может объединять:

- граф;
- tools;
- промпты;
- memory;
- параметры модели;
- правила маршрутизации.

Примеры:

- `researcher.py`;
- `coder.py`;
- `analyst.py`;
- `supervisor.py`.

### 4.7. `src/langgraph_agents/mas/`

Multi-agent systems.

Сюда следует вынести:

- orchestrator;
- supervisor;
- shared state;
- handoff-логику;
- маршрутизацию между агентами;
- правила завершения multi-agent workflow.

### 4.8. `examples/`

Исполняемые учебные примеры.

Файлы в `examples/` должны быть простыми и понятными. Они должны показывать, как пользоваться кодом из `src/`.

Примеры:

- `01_basic_graph.py`;
- `02_messages.py`;
- `03_structured_output.py`;
- `04_tool_calling.py`;
- `05_react_agent.py`;
- `06_multi_model_dialog.py`;
- `07_multi_agent_system.py`;
- `08_specialist_agents.py`.

### 4.9. `apps/`

Точки запуска реальных приложений.

Рекомендуемые приложения:

- `apps/cli/` - CLI-интерфейс;
- `apps/api/` - HTTP API;
- `apps/playground/` - локальная песочница для экспериментов.

### 4.10. `configs/`

Конфигурации проекта и агентов.

Сюда следует вынести параметры, которые не должны быть зашиты в код:

- модель;
- temperature;
- max tokens;
- включенные tools;
- настройки memory;
- настройки конкретных агентов.

### 4.11. `prompts/`

Промпты проекта.

Рекомендуется отделить:

- системные промпты;
- промпты агентов;
- промпты для tools;
- шаблоны инструкций.

### 4.12. `evals/`

Оценка качества агентных сценариев.

Для агентных систем это критичный слой, потому что обычные unit-тесты не покрывают качество рассуждений, маршрутизации и tool calling.

Рекомендуемые проверки:

- корректность маршрутизации;
- качество structured output;
- корректность tool calling;
- устойчивость multi-agent handoff;
- отсутствие очевидных hallucinations;
- качество ответов агентов-специалистов.

### 4.13. `tests/`

Автоматические тесты.

Рекомендуемая структура:

- `tests/unit/` - быстрые тесты отдельных функций;
- `tests/integration/` - тесты графов и связок;
- `tests/fixtures/` - тестовые данные.

Минимально стоит покрыть:

- state-модели;
- routing;
- registry tools;
- structured output;
- memory/context strategies;
- MAS handoff.

## 5. Рекомендации по миграции текущих файлов

Текущие файлы следует разнести следующим образом:

```text
state.py                         -> src/langgraph_agents/core/state.py
llm_factory.py                   -> src/langgraph_agents/core/llm_factory.py
settings.py                      -> src/langgraph_agents/core/config.py
config.py                        -> src/langgraph_agents/core/config.py
work_with_messages.py            -> src/langgraph_agents/core/messages.py
manage_context_strategy.py       -> src/langgraph_agents/memory/context_strategy.py
optimized_state.py               -> src/langgraph_agents/core/state.py или src/langgraph_agents/memory/

assistant_graph.py               -> src/langgraph_agents/graphs/assistant_graph.py
routing.py                       -> src/langgraph_agents/graphs/routing_graph.py
multi_model_graph.py             -> src/langgraph_agents/graphs/multimodel_graph.py
langgraph-agent.py               -> examples/01_basic_graph.py или src/langgraph_agents/graphs/

nodes/version_config_control.py  -> src/langgraph_agents/nodes/version_control.py
nodes/expert_consensus_node.py   -> src/langgraph_agents/nodes/expert_consensus.py

tools/get_quote.py               -> src/langgraph_agents/tools/quotes.py
tools/binding_tools.py           -> src/langgraph_agents/tools/registry.py
tools/set_of_tools.py            -> src/langgraph_agents/tools/registry.py
tools/react_approach.py          -> examples/05_react_agent.py или src/langgraph_agents/graphs/react_graph.py
tools/model_prepare.py           -> src/langgraph_agents/core/llm_factory.py

MAS/main.py                      -> examples/07_multi_agent_system.py или apps/cli/main.py
MAS/tools/orchestrator_state.py  -> src/langgraph_agents/mas/state.py
MAS/tools/file_tools.py          -> src/langgraph_agents/tools/files.py
MAS/tools/web_tools.py           -> src/langgraph_agents/tools/web.py
MAS/tools/db_tools.py            -> src/langgraph_agents/tools/database.py

react-pattern/tools/...          -> src/langgraph_agents/graphs/react_graph.py,
                                    src/langgraph_agents/tools/,
                                    examples/05_react_agent.py

main_concepts/*.py               -> examples/ или docs/
```

## 6. Правила именования

### 6.1. Название Python-пакета

Рекомендуемое имя пакета:

```text
langgraph_agents
```

Не рекомендуется называть пакет `langgraph`, чтобы не конфликтовать с официальной библиотекой LangGraph.

### 6.2. Имена файлов

Файлы должны использовать `snake_case`.

Нежелательно использовать дефисы в Python-файлах:

```text
chat-model.py      -> chat_model.py
http-rq.py         -> http_request.py
langgraph-agent.py -> langgraph_agent.py
fake-ai-message.py -> fake_ai_message.py
```

### 6.3. Имена директорий

Директории должны быть в нижнем регистре.

Желательно заменить:

```text
MAS/ -> mas/
```

## 7. Рекомендуемый `README.md`

Главный README должен быть кратким входом в проект.

Рекомендуемые разделы:

```text
# LangGraph Agents

## Overview
Краткое описание проекта.

## Architecture
- Core
- Graphs
- Tools
- MAS
- Specialist agents

## Quickstart
Установка, переменные окружения, запуск первого агента.

## Examples
Список учебных сценариев.

## Development
Тесты, форматирование, evals.

## Project Structure
Краткое описание директорий.
```

## 8. Этапы внедрения

### Этап 1. Подготовка основы

- Создать `src/langgraph_agents/`.
- Создать базовые директории: `core/`, `graphs/`, `nodes/`, `tools/`, `agents/`, `mas/`, `memory/`.
- Добавить `__init__.py`.
- Подготовить `pyproject.toml`.

### Этап 2. Перенос общего кода

- Перенести `state.py`, `llm_factory.py`, `settings.py`, `config.py`.
- Объединить дублирующуюся конфигурацию.
- Настроить импорты через пакет `langgraph_agents`.

### Этап 3. Перенос графов и nodes

- Перенести графы в `src/langgraph_agents/graphs/`.
- Перенести node-функции в `src/langgraph_agents/nodes/`.
- Убрать код запуска из библиотечных файлов.

### Этап 4. Перенос tools

- Разнести tools по смысловым файлам.
- Добавить `tools/registry.py`.
- Разделить web, file, database и domain-specific tools.

### Этап 5. Оформление MAS

- Перенести multi-agent логику в `src/langgraph_agents/mas/`.
- Выделить orchestrator, supervisor, handoff и shared state.
- Подготовить пример запуска MAS в `examples/07_multi_agent_system.py`.

### Этап 6. Оформление examples и docs

- Перенести учебные сценарии в `examples/`.
- Перенести объяснения в `docs/`.
- Удалить дубли между корнем и `main_concepts`.

### Этап 7. Тесты и evals

- Добавить базовые unit-тесты.
- Добавить integration-тесты для графов.
- Добавить eval-сценарии для agent workflows.

## 9. Критерии готовности

Реорганизация считается выполненной, если:

- в корне проекта нет рабочих Python-файлов;
- основной код находится в `src/langgraph_agents/`;
- учебные сценарии находятся в `examples/`;
- документация находится в `docs/`;
- multi-agent logic находится в `src/langgraph_agents/mas/`;
- tools разнесены по смысловым модулям;
- отсутствуют дубли между корнем и `main_concepts`;
- все импорты работают из установленного пакета;
- есть базовые тесты;
- README объясняет структуру и быстрый старт.

## 10. Итоговая архитектурная идея

Репозиторий должен быть разделен на три понятных слоя:

```text
docs/ + examples/  -> обучение и демонстрации
src/               -> переиспользуемая библиотека агентов
apps/              -> реальные точки запуска
```

Такой подход позволит развивать проект одновременно как обучающий материал, экспериментальную лабораторию и основу для production-ready LangGraph-агентов.
