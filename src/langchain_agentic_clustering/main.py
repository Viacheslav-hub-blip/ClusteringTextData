"""Entry point for the LangChain agentic clustering project."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.model import embeddings, model
from src.langchain_agentic_clustering.orchestrator import LangChainAgentOrchestrator
from src.langchain_agentic_clustering.services.session import AgenticClusteringSession
from src.langchain_agentic_clustering.services.text_utils import dump_json

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

EXCEL_PATH = PROJECT_ROOT / "data" / "comments_1000.xlsx"
TEXT_COLUMN = "comment"
ID_COLUMN = "comment_id"
SHEET_NAME = None
LIMIT = 10

MAX_REPAIRS = 3
AGENT_RECURSION_LIMIT = 40

OUTPUT_DIR = PACKAGE_ROOT / "output"
OUTPUT_JSON_PATH = OUTPUT_DIR / "agentic_clustering_result.json"
DEBUG_JSON_PATH = OUTPUT_DIR / "agentic_clustering_debug.json"
AGENT_TRACE_PATH = OUTPUT_DIR / "agentic_clustering_agent_trace.json"


def main() -> None:
    """Run the full batch clustering flow from a main file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = AgenticClusteringSession(llm=model, embeddings=embeddings)
    session.initialize_from_excel(
        EXCEL_PATH,
        text_column=TEXT_COLUMN,
        id_column=ID_COLUMN,
        sheet_name=SHEET_NAME,
        limit=LIMIT,
    )

    orchestrator = LangChainAgentOrchestrator(
        llm=model,
        session=session,
        max_repairs=MAX_REPAIRS,
        recursion_limit=AGENT_RECURSION_LIMIT,
    )
    agent_result = orchestrator.run()

    output_records = [asdict(record) for record in session.export_output_records()]
    OUTPUT_JSON_PATH.write_text(dump_json(output_records), encoding="utf-8")
    DEBUG_JSON_PATH.write_text(dump_json(session.export_debug_snapshot()), encoding="utf-8")
    AGENT_TRACE_PATH.write_text(dump_json(agent_result), encoding="utf-8")

    print(f"Готово: {len(output_records)} комментариев обработано.")
    print(f"Результат сохранен в {OUTPUT_JSON_PATH}")
    print(f"Debug snapshot сохранен в {DEBUG_JSON_PATH}")
    print(f"Трасса агента сохранена в {AGENT_TRACE_PATH}")


if __name__ == "__main__":
    main()
