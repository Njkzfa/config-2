# Инструмент визуализации графа зависимостей для pip-пакетов

## Файлы

- `main.py` — основной скрипт CLI.
- `config.xml` — пример конфигурационного файла.
- `test_repo.txt` — пример тестового репозитория для режима `test_mode=true`.

## Примеры запуска

1. Вывести параметры конфигурации (Этап 1):

```bash
python main.py config.xml --mode print-config
```

2. Вывести прямые зависимости пакета (Этап 2):

```bash
python main.py config.xml --mode direct-deps
```

3. Построить и вывести граф зависимостей (Этап 3):

```bash
python main.py config.xml --mode graph
```

4. Вывести порядок загрузки зависимостей (Этап 4):

```bash
python main.py config.xml --mode load-order
```

5. Сгенерировать Mermaid и SVG (Этап 5):

```bash
python main.py config.xml --mode visualize
```

Для генерации SVG требуется установленный `mmdc` (mermaid-cli).
