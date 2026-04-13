# Stable Structured Clustering

Новый алгоритм для сравнения с текущим пайплайном.

Архитектура:

1. `text -> structured signal`
2. `exact specific_key -> prototype`
3. `prototype -> deterministic specific clusters`
4. `specific clusters -> deterministic parent clusters`
5. `cluster -> stable label from representative members`

Ключевая идея:

- LLM используется только для извлечения структуры комментария.
- Названия групп не генерируются заново для каждого комментария.
- Итоговые `specific_group` и `parent_group` выбираются детерминированно из представителей кластера.

Запуск:

```powershell
python -m src.stable_structured_clustering.main --excel data/comments_1000.xlsx --text-column comment --id-column comment_id --limit 100
```

Если `--excel` не передан, используется небольшой демо-набор.
