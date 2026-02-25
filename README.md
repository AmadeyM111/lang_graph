# LangGraph Agent Demo

A minimal example of a **state machine workflow** built with [LangGraph](https://github.com/langchain-ai/langgraph), demonstrating conditional routing and looping via a driving eligibility checker.

## What It Does

The agent simulates date progression day by day until a person reaches the legal driving age (18 years old), then outputs a congratulatory message with the exact date when driving becomes allowed.

### Workflow Graph

```
START → calculate_age
            ├─ age >= 18 → generate_success_message → END
            └─ age <  18 → autoincrement_date ──────→ (loop)
```

## State

| Field        | Type   | Description                              |
|--------------|--------|------------------------------------------|
| `name`       | `str`  | First name                               |
| `surname`    | `str`  | Last name                                |
| `birth_date` | `date` | Date of birth                            |
| `today`      | `date` | Simulated current date (advances by day) |
| `age`        | `int`  | Calculated age on current simulated date |
| `message`    | `str`  | Result message on success                |

## Requirements

```
langgraph
```

```bash
pip install langgraph
```

## Usage

```bash
python langgraph-agent.py
```

Example output:

```
2026-02-19 -> 2026-02-20
...
Congrats, Алексей Яковенко! You are already 18 years old and you can drive!
Date when allowed to drive: 2026-02-19
```

## Notes

- `recursion_limit` is set to `2000` — enough to cover ~3 graph steps per simulated day over a multi-year span.
- Date arithmetic correctly accounts for whether the birthday has occurred yet in the current year.
