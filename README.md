# clusteringtextdata

Библиотека для простой кластеризации текстовых комментариев через embeddings, FAISS/BM25 и LLM.

Текущая версия убирает дорогие лишние этапы:

- нет LLM-нормализации каждого комментария;
- нет agentic-постобработки;
- нет обязательного слияния групп после нейминга;
- есть один простой вызов для Jupyter Notebook, IDE, CSV/XLSX и pandas DataFrame.

Библиотека не создает LLM-клиент и не читает API-ключи. Вы передаете готовые LangChain-объекты `llm` и `embeddings` снаружи.

## Быстрый запуск в Jupyter

```python
from clusteringtextdata import cluster_text_data

result_df = cluster_text_data(
    "comments.xlsx",
    llm=llm,
    embeddings=embeddings,
    text_column="text",
    output_path="clusters.xlsx",
)
```

Если данные уже в pandas DataFrame:

```python
result_df = cluster_text_data(
    df,
    llm=llm,
    embeddings=embeddings,
    text_column="comment",
    id_column="id",
)
```

Для ускорения можно отключить LLM-нейминг групп. Тогда названием группы будет первый характерный комментарий:

```python
result_df = cluster_text_data(
    df,
    llm=llm,
    embeddings=embeddings,
    text_column="comment",
    generate_group_names=False,
)
```

## Что делает pipeline

1. Читает строки из DataFrame, списка словарей, CSV или XLSX.
2. Берет текст из `text_column`.
3. Делает только техническую очистку текста: пробелы, типографские кавычки, тире.
4. Строит embeddings для комментариев.
5. Обрабатывает комментарии последовательно.
6. Для каждого нового комментария ищет похожие уже обработанные комментарии через FAISS и BM25.
7. Передает LLM только текущий комментарий и несколько групп-кандидатов.
8. LLM выбирает `existing_group` или `new_group`.
9. В конце опционально генерирует короткие названия групп.
10. Возвращает исходные строки с добавленным `group_name`.

## Основные параметры

- `text_column` — колонка с текстом комментария.
- `id_column` — колонка с ID. Если не указана, используется `comment_id` или номер строки.
- `output_path` — путь для сохранения результата в `.xlsx` или `.csv`.
- `generate_group_names=True` — включить LLM-нейминг групп.
- `generate_group_names=False` — быстрее, без дополнительного LLM-вызова на каждую группу.
- `merge_same_name_groups=False` — по умолчанию группы не склеиваются после нейминга.
- `show_progress=False` — по умолчанию без консольного прогресса, удобно для notebook.

## Использование класса напрямую

```python
from clusteringtextdata import VectorLLMClusteringPipeline

pipeline = VectorLLMClusteringPipeline(
    llm=llm,
    embeddings=embeddings,
    retrieval_top_k=12,
    max_examples_per_candidate_group=3,
    primary_similarity_threshold=0.5,
)

result = pipeline.run([
    {"comment_id": "1", "text": "Не приходит код подтверждения перевода"},
    {"comment_id": "2", "text": "Долго приходит СМС для подтверждения"},
])
```

В Jupyter можно использовать обычный синхронный `run`, но для async-окружений также доступен:

```python
result = await pipeline.arun(comments)
```

## Формат результата

`cluster_text_data` возвращает pandas DataFrame, если pandas установлен. Если pandas недоступен, возвращается список словарей.

В результат добавляется поле:

- `group_name` — название найденной группы.

Если передать `include_details=True`, дополнительно добавятся:

- `normalized_text`;
- `decision_type`;
- `decision_reason`.
