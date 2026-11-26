# Слепок портала

`BitrixReadOnlyClient` выполняет вызовы через общий лимитер портала.
`PortalSnapshotCollector` нормализует метаданные в DTO из `core/contracts.py`
и записывает наблюдения о неполноте. Сохранение слепка в SQLite и JSON
выполняет раннер. [Контракт](../../docs/contracts.md) · [Запуск](../../docs/operations.md)
