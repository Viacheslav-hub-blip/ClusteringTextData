# LangChain Agentic Clustering

Отдельный экспериментальный проект для батчевой кластеризации банковских комментариев.

Идея:
- первично извлечь из каждого текста структурный банковский сигнал
- собрать детерминированную кластеризацию без заранее заданного справочника категорий
- поверх результата запустить LangChain-агента, который ищет подозрительно похожие локальные neighborhoods
- перекластеризовать только эти neighborhoods, а не весь датасет
- перед применением локального patch отдельно проверить, что он реально улучшает структуру

Основные файлы:
- `main.py` — запуск из IDE или как обычный main-файл
- `orchestrator.py` — LangChain-агент и набор инструментов
- `services/session.py` — состояние текущей кластеризации, neighborhoods и patch-логика
- `services/structure_extractor.py` — первичный банковский semantic parsing
- `services/snapshot_builder.py` — детерминированная сборка snapshot
- `services/critic.py` — критик neighborhoods
- `services/local_reclusterer.py` — локальная перекластеризация
- `services/patch_evaluator.py` — оценка patch до применения

Результаты складываются в папку `output` рядом с этим README.
