# clusteringtextdata

Библиотека для кластеризации текстовых комментариев через минимальную предобработку, embeddings, FAISS, BM25 и LLM.

Текущая рабочая версия не использует agentic-постобработку, LLM-нормализацию, проверку пустых комментариев и резервное присвоение по similarity. FAISS и BM25 используются для поиска групп-кандидатов, финальное решение принимает LLM. После основного прохода pipeline может выполнить дополнительное LLM-слияние маленьких групп с похожими существующими группами.

У каждой входной записи в успешном результате всегда есть группа. Если запись не подходит ни к одной существующей группе, LLM должна вернуть `new_group` и `group_name`, включая случай одиночной группы.

## Схема работы

```text
исходные строки
  -> минимальная предобработка текста
  -> embeddings
  -> FAISS top_k
  -> BM25 top_k
  -> объединение найденных групп-кандидатов
  -> LLM выбирает existing_group или new_group
  -> второй проход: LLM проверяет слияние маленьких групп
  -> результат: все исходные поля + group_name
```

Минимальная предобработка:

- приведение к нижнему регистру;
- замена `ё` на `е`;
- схлопывание пробелов;
- схлопывание повторяющейся пунктуации: `!!! -> !`, `??? -> ?`, `... -> .`;
- замена типографских кавычек и тире на простые символы.

## Использование

```python
from clusteringtextdata import VectorLLMClusteringPipeline

pipeline = VectorLLMClusteringPipeline(
    llm=llm,
    embeddings=embeddings,
    text_field="text",
    faiss_top_k=50,
    bm25_top_k=50,
    candidate_group_limit=15,
    max_examples_per_candidate_group=5,
    merge_small_groups=True,
    small_group_max_size=2,
    merge_candidate_group_limit=20,
    max_llm_retries=1,
    max_embedding_retries=1,
)

result = pipeline.run(rows)
```

`rows` — список словарей. Все исходные поля сохраняются:

```python
rows = [
    {"comment_id": "1", "text": "Не приходит код подтверждения", "source": "app"},
    {"comment_id": "2", "text": "Долго жду код для перевода", "source": "web"},
]
```

Результат:

```python
{
    "rows": [
        {
            "comment_id": "1",
            "text": "Не приходит код подтверждения",
            "source": "app",
            "group_name": "Проблемы с кодом подтверждения",
        }
    ],
    "groups": [
        {
            "group_id": "group_0001",
            "group_name": "Проблемы с кодом подтверждения",
            "comment_count": 2,
        }
    ],
}
```

## Jupyter Notebook

В ноутбуке можно использовать синхронный метод:

```python
result = pipeline.run(rows)
```

Метод безопасно работает в окружении с уже запущенным event loop. Если удобнее использовать async-стиль:

```python
result = await pipeline.arun(rows)
```

## DataFrame и Excel

```python
df = pipeline.to_dataframe(result)
pipeline.save_excel(result, "clusters.xlsx")
```

`to_dataframe` требует установленный `pandas`. Сохранение Excel использует `openpyxl`.

## Формат ответа LLM

Дефолтный system prompt находится в `src/prompts.py`. `src/config.py` содержит только dataclass-конфигурацию и не хранит текст запросов. Все данные для LLM передаются через system prompt, human message в цепочке всегда пустой.

LLM должна возвращать только JSON без markdown:

```json
{
  "decision_type": "existing_group",
  "group_id": "group_0001",
  "group_name": "",
  "reason": "Комментарий описывает ту же проблему."
}
```

Для новой группы:

```json
{
  "decision_type": "new_group",
  "group_id": "",
  "group_name": "Блокировка карты при оплате",
  "reason": "Среди кандидатов нет группы с этой проблемой."
}
```

Если LLM вернула несуществующий `group_id`, невалидный `decision_type` или не указала `group_name` для новой группы, pipeline завершает работу с ошибкой. Если LLM-вызов или batch embedding-вызов не сработал после retry-попыток, pipeline также завершает работу с ошибкой.

## Слияние маленьких групп

По умолчанию после первичной кластеризации pipeline проверяет группы размером до `small_group_max_size=2`. Для каждой такой группы он ищет похожие существующие группы через FAISS и BM25, затем отправляет LLM отдельное решение: оставить группу отдельной или присоединить ее к одному из кандидатов.

Чтобы отключить второй проход:

```python
pipeline = VectorLLMClusteringPipeline(
    llm=llm,
    embeddings=embeddings,
    merge_small_groups=False,
)
```
