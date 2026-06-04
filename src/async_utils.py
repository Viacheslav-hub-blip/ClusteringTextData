"""Утилиты для запуска асинхронного кода из синхронного API.

Файл содержит:
- ``run_coroutine_sync`` — безопасный синхронный запуск coroutine в обычном Python
  и в средах с уже запущенным event loop.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_coroutine_sync(coroutine: Coroutine[Any, Any, T]) -> T:
    """Синхронно выполняет coroutine и возвращает ее результат.

    Args:
        coroutine: Coroutine, которую нужно выполнить до завершения.

    Returns:
        Результат выполнения переданной coroutine.

    Raises:
        BaseException: Исключение, которое возникло внутри coroutine.

    Notes:
        В обычном скрипте используется ``asyncio.run``. Если текущий поток уже имеет
        запущенный event loop, как часто бывает в Jupyter Notebook, coroutine
        выполняется в отдельном потоке с собственным event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result: list[T] = []
    error: list[BaseException] = []

    def _runner() -> None:
        """Запускает coroutine в отдельном потоке и сохраняет результат или ошибку."""
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if error:
        raise error[0]
    return result[0]
