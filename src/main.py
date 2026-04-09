"""Application entrypoint for running the demo pipeline."""

import json

try:
    from .clustring_data import CommentGroupingPipeline
    from .model import embeddings, model as llm
except ImportError:
    from clustring_data import CommentGroupingPipeline
    from model import embeddings, model as llm


# Пример A: перефразы одного кейса
test_case_a = [
    {"comment_id": "a1", "text": "Не могу перевести деньги"},
    {"comment_id": "a2", "text": "Перевод денег не проходит"},
]

# Пример B: общий и частный кейс
test_case_b = [
    {"comment_id": "b1", "text": "Не могу перевести деньги"},
    {"comment_id": "b2", "text": "Не проходит перевод между своими счетами"},
]

# Пример C: разные сущности
test_case_c = [
    {"comment_id": "c1", "text": "Не могу перевести в компанию А"},
    {"comment_id": "c2", "text": "Не могу перевести в компанию Б"},
]

# Пример D: разная детализация
test_case_d = [
    {"comment_id": "d1", "text": "Проблема с переводами"},
    {"comment_id": "d2", "text": "Проблема с переводами по СБП"},
]

# Полный сводный тест из ТЗ
test_case_full = [
    {"comment_id": "c1", "text": "Не могу перевести деньги"},
    {"comment_id": "c2", "text": "Не проходит перевод между своими счетами"},
    {"comment_id": "c3", "text": "Перевод в компанию А блокируется"},
    {"comment_id": "c4", "text": "Перевод денег не проходит"},
    {"comment_id": "c5", "text": "Не могу перевести в компанию Б"},
    {"comment_id": "c6", "text": "Проблема с переводами по СБП"},
    {"comment_id": "c7", "text": "не могу войти в приложение"},
    {"comment_id": "c8", "text": "приложение не открывается"},
    {"comment_id": "c9", "text": "вход в приложение через Face ID не работает"},
]


def run_test(
        name: str,
        comments: list[dict],
        pipeline: CommentGroupingPipeline,
) -> None:
    """Run the pipeline on a test set and print the result."""
    print("\n" + "=" * 60)
    print(f"ТЕСТ: {name}")
    print("=" * 60)

    print("\nВХОД:")
    for comment in comments:
        print(f"  [{comment['comment_id']}] {comment['text']}")

    results = pipeline.run(comments)

    print("\nРЕЗУЛЬТАТ:")
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main() -> None:
    """Create the pipeline and run demo scenarios."""
    pipeline = CommentGroupingPipeline(
        llm=llm,
        embeddings=embeddings,
        candidate_top_k=10,
        extraction_batch_size=50,
    )

    # run_test("A — Перефразы одного кейса", test_case_a, pipeline)
    # run_test("B — Общий и частный кейс", test_case_b, pipeline)
    # run_test("C — Разные сущности", test_case_c, pipeline)
    # run_test("D — Разная детализация", test_case_d, pipeline)
    run_test("FULL — Полный сводный тест", test_case_full, pipeline)


if __name__ == "__main__":
    main()
