#!/usr/bin/env python3
"""Restructure the repository into a professional LangGraph agents layout.

The script is intentionally conservative:
- root-level files are treated as canonical when a duplicate exists in
  main_concepts;
- exact duplicates from main_concepts are deleted;
- divergent main_concepts files are preserved under examples/_main_concepts_legacy;
- existing destination files are not overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_MAIN = Path("examples/_main_concepts_legacy")


MOVE_MAP = {
    "state.py": "src/langgraph_agents/core/state.py",
    "llm_factory.py": "src/langgraph_agents/core/llm_factory.py",
    "work_with_messages.py": "src/langgraph_agents/core/messages.py",
    "settings.py": "src/langgraph_agents/core/settings.py",
    "config.py": "src/langgraph_agents/core/config.py",
    "optimized_state.py": "src/langgraph_agents/core/optimized_state.py",
    "manage_context_strategy.py": "src/langgraph_agents/memory/context_strategy.py",
    "assistant_graph.py": "src/langgraph_agents/graphs/assistant_graph.py",
    "routing.py": "src/langgraph_agents/graphs/routing_graph.py",
    "multi_model_graph.py": "src/langgraph_agents/graphs/multimodel_graph.py",
    "classification.py": "src/langgraph_agents/nodes/classification.py",
    "sentiment_parser_in_one_chain.py": "src/langgraph_agents/nodes/sentiment.py",
    "nodes/version_config_control.py": "src/langgraph_agents/nodes/version_control.py",
    "nodes/expert_consensus_node.py": "src/langgraph_agents/nodes/expert_consensus.py",
    "tools/get_quote.py": "src/langgraph_agents/tools/quotes.py",
    "tools/binding_tools.py": "src/langgraph_agents/tools/binding_tools.py",
    "tools/set_of_tools.py": "src/langgraph_agents/tools/set_of_tools.py",
    "tools/model_prepare.py": "src/langgraph_agents/tools/model_prepare.py",
    "tools/react_approach.py": "examples/05_react_agent.py",
    "MAS/tools/orchestrator_state.py": "src/langgraph_agents/mas/state.py",
    "MAS/tools/file_tools.py": "src/langgraph_agents/tools/files.py",
    "MAS/tools/web_tools.py": "src/langgraph_agents/tools/web.py",
    "MAS/tools/db_tools.py": "src/langgraph_agents/tools/database.py",
    "MAS/nodes/researcher.py": "src/langgraph_agents/agents/researcher.py",
    "MAS/nodes/editor.py": "src/langgraph_agents/agents/editor.py",
    "MAS/nodes/data-engineer.py": "src/langgraph_agents/agents/data_engineer.py",
    "MAS/nodes/orchestrator.py": "src/langgraph_agents/mas/orchestrator.py",
    "MAS/nodes/main.py": "examples/08_mas_nodes.py",
    "MAS/main.py": "examples/07_multi_agent_system.py",
    "main.py": "apps/cli/main.py",
    "explicit_messages_work.py": "examples/02_messages.py",
    "structured_output.py": "examples/03_structured_output.py",
    "langgraph-agent.py": "examples/01_basic_graph.py",
    "triple_model_mode.py": "examples/06_triple_model_mode.py",
    "multimodel_dialogs.py": "examples/06_multi_model_dialog.py",
    "chat-model.py": "examples/chat_model.py",
    "defining_the_structure_of_a_pydentic_data_model.py": "examples/pydantic_data_model_structure.py",
    "messages_metadata_analysis.py": "examples/messages_metadata_analysis.py",
    "fake-ai-message.py": "examples/fake_ai_message.py",
    "http-rq.py": "examples/http_request.py",
    "react-pattern/tools/mcp_integration.py": "examples/react_pattern/mcp_integration.py",
    "react-pattern/tools/main_work_cicle.py": "examples/react_pattern/main_work_cycle.py",
    "react-pattern/tools/lifecircle_manger.py": "examples/react_pattern/lifecycle_manager.py",
    "react-pattern/tools/create_agent_with_memory.py": "examples/react_pattern/create_agent_with_memory.py",
    "react-pattern/tools/api_service.py": "examples/react_pattern/api_service.py",
}


MAIN_CONCEPTS_MAP = {
    f"main_concepts/{src}": dst
    for src, dst in MOVE_MAP.items()
    if not src.startswith("MAS/") and not src.startswith("react-pattern/")
}


PACKAGE_DIRS = [
    "src/langgraph_agents",
    "src/langgraph_agents/core",
    "src/langgraph_agents/graphs",
    "src/langgraph_agents/nodes",
    "src/langgraph_agents/tools",
    "src/langgraph_agents/memory",
    "src/langgraph_agents/agents",
    "src/langgraph_agents/mas",
    "apps",
    "apps/cli",
    "tests",
    "tests/unit",
    "tests/integration",
]


SCAFFOLD_FILES = {
    "configs/default.yaml": "",
    "configs/local.example.yaml": "",
    "evals/README.md": (
        "# Evals\n\n"
        "Сценарии оценки качества агентных workflow: routing, tool calling, "
        "structured output и handoff между агентами.\n"
    ),
    "prompts/README.md": (
        "# Prompts\n\n"
        "Каталог для системных промптов, промптов агентов и шаблонов инструкций для tools.\n"
    ),
    "tests/fixtures/.gitkeep": "",
    "scripts/.gitkeep": "",
    "pyproject.toml": (
        "[project]\n"
        'name = "langgraph-agents"\n'
        'version = "0.1.0"\n'
        'description = "LangGraph agents, tools, examples, and multi-agent system experiments."\n'
        'readme = "README.md"\n'
        'requires-python = ">=3.10"\n'
        "dependencies = []\n\n"
        "[tool.pytest.ini_options]\n"
        'pythonpath = ["src"]\n'
        'testpaths = ["tests"]\n'
    ),
}


def rel(path: str | Path) -> Path:
    return ROOT / Path(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_file(path: Path, content: str, report: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    report.append(f"created {path.relative_to(ROOT)}")


def move_file(src_rel: str, dst_rel: str, report: list[str]) -> None:
    src = rel(src_rel)
    dst = rel(dst_rel)

    if not src.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if src.is_file() and dst.is_file() and sha256(src) == sha256(dst):
            src.unlink()
            report.append(f"deleted exact duplicate {src_rel}")
            return

        conflict = rel("examples/_migration_conflicts") / src_rel
        conflict.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(conflict))
        report.append(f"preserved conflict {src_rel} -> {conflict.relative_to(ROOT)}")
        return

    shutil.move(str(src), str(dst))
    report.append(f"moved {src_rel} -> {dst_rel}")


def handle_main_concepts(report: list[str]) -> None:
    readme = rel("main_concepts/README.md")
    if readme.exists():
        move_file("main_concepts/README.md", "docs/main-concepts.md", report)

    env_example = rel("main_concepts/.env.example")
    root_env = rel(".env.example")
    if env_example.exists() and root_env.exists() and sha256(env_example) == sha256(root_env):
        env_example.unlink()
        report.append("deleted exact duplicate main_concepts/.env.example")

    for src_rel, canonical_rel in MAIN_CONCEPTS_MAP.items():
        src = rel(src_rel)
        canonical = rel(canonical_rel)
        if not src.exists():
            continue

        if canonical.exists() and src.is_file() and sha256(src) == sha256(canonical):
            src.unlink()
            report.append(f"deleted exact duplicate {src_rel}")
            continue

        legacy_rel = LEGACY_MAIN / Path(src_rel).relative_to("main_concepts")
        move_file(src_rel, str(legacy_rel), report)

    main_dir = rel("main_concepts")
    if main_dir.exists():
        for path in sorted(main_dir.rglob("*.py")):
            legacy_rel = LEGACY_MAIN / path.relative_to(main_dir)
            move_file(str(path.relative_to(ROOT)), str(legacy_rel), report)


def write_report(report: list[str]) -> None:
    body = [
        "# Migration Report",
        "",
        "## Выполненные действия",
        "",
        *[f"- {line}" for line in report],
        "",
        "## Проверка после миграции",
        "",
        "```bash",
        "python -m compileall src examples apps",
        "pytest",
        "git status --short",
        "```",
        "",
    ]
    path = rel("docs/migration-report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true", help="Create a git commit after migration.")
    parser.add_argument("--push", action="store_true", help="Push the current branch after committing.")
    parser.add_argument(
        "--message",
        default="Restructure LangGraph agent project",
        help="Commit message for --commit.",
    )
    args = parser.parse_args()

    report: list[str] = []

    for directory in PACKAGE_DIRS:
        rel(directory).mkdir(parents=True, exist_ok=True)
        ensure_file(rel(directory) / "__init__.py", "", report)

    for file_rel, content in SCAFFOLD_FILES.items():
        ensure_file(rel(file_rel), content, report)

    for src_rel, dst_rel in MOVE_MAP.items():
        move_file(src_rel, dst_rel, report)

    handle_main_concepts(report)
    write_report(report)

    print("\n".join(report) if report else "No changes were needed.")

    if args.commit or args.push:
        run(["git", "add", "."])
        run(["git", "commit", "-m", args.message])

    if args.push:
        run(["git", "push"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
