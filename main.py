#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Set, Optional


# =======================
#  Общие исключения
# =======================

class ConfigError(Exception):
    pass


# =======================
#  Модель конфигурации
# =======================

@dataclass
class Config:
    package_name: str
    package_version: str
    repo_url: str
    test_mode: bool
    test_repo_path: str
    output_svg: str


def _parse_bool(text: str) -> bool:
    text = text.strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ConfigError(f"Некорректное булево значение: {text!r}")


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise ConfigError(f"Файл конфигурации не найден: {path}")

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ConfigError(f"Ошибка парсинга XML: {e}") from e

    def get(tag: str, required: bool = True, default: Optional[str] = None) -> str:
        el = root.find(tag)
        if el is None or not (el.text and el.text.strip()):
            if required:
                raise ConfigError(f"Отсутствует обязательный параметр <{tag}> в конфигурации")
            return default or ""
        return el.text.strip()

    package_name = get("package_name")
    if not re.match(r"^[A-Za-z0-9_.\-]+$", package_name):
        raise ConfigError(f"Некорректное имя пакета: {package_name!r}")

    package_version = get("package_version")
    if not re.match(r"^[A-Za-z0-9_.\-!+]+$", package_version):
        raise ConfigError(f"Некорректный формат версии пакета: {package_version!r}")

    repo_url = get("repo_url")
    if not (repo_url.startswith("http://") or repo_url.startswith("https://")):
        raise ConfigError("repo_url должен начинаться с http:// или https://")
    repo_url = repo_url.rstrip("/")

    test_mode_str = get("test_mode", required=True)
    test_mode = _parse_bool(test_mode_str)

    test_repo_path = get("test_repo_path", required=test_mode, default="")

    if test_mode:
        if not os.path.exists(test_repo_path):
            raise ConfigError(f"Файл тестового репозитория не найден: {test_repo_path}")
        if not os.path.isfile(test_repo_path):
            raise ConfigError(f"test_repo_path должен быть файлом: {test_repo_path}")

    output_svg = get("output_svg")
    if not output_svg.lower().endswith(".svg"):
        raise ConfigError(f"Имя выходного файла должно оканчиваться на .svg, сейчас: {output_svg!r}")

    return Config(
        package_name=package_name,
        package_version=package_version,
        repo_url=repo_url,
        test_mode=test_mode,
        test_repo_path=test_repo_path,
        output_svg=output_svg,
    )


def print_config(cfg: Config) -> None:
    """Этап 1: выводим все параметры в формате ключ=значение."""
    print(f"package_name={cfg.package_name}")
    print(f"package_version={cfg.package_version}")
    print(f"repo_url={cfg.repo_url}")
    print(f"test_mode={cfg.test_mode}")
    print(f"test_repo_path={cfg.test_repo_path}")
    print(f"output_svg={cfg.output_svg}")


# ============================
#  Работа с реальным pip-репой
# ============================

def fetch_requires_dist_from_pypi(package: str, version: str, base_url: str) -> List[str]:
    """
    Обращение к pip-репозиторию (PyPI) по HTTP:
    GET {base_url}/pypi/{package}/{version}/json

    Возвращает список строк требований (requires_dist) в формате pip.
    """
    url = f"{base_url}/pypi/{package}/{version}/json"
    try:
        with urllib.request.urlopen(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} при обращении к {url}")
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP ошибка {e.code} при обращении к {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ошибка сети при обращении к {url}: {e}") from e

    info = data.get("info") or {}
    requires = info.get("requires_dist")

    if requires is None:
        # у пакета нет зависимостей
        return []
    if not isinstance(requires, list):
        raise RuntimeError("Поле requires_dist имеет неожиданный тип")

    # всё ещё строки формата pip: 'idna>=2.5; python_version >= "3"'
    return [r for r in requires if isinstance(r, str)]


_req_name_re = re.compile(r"^[A-Za-z0-9_.\-]+")


def parse_requirement_name(requirement: str) -> Optional[str]:
    """
    Получить имя пакета из строки формата pip, без версий и маркеров окружения.

    Примеры:
      'idna>=2.5; python_version >= "3"' -> 'idna'
      'charset_normalizer>=2,<4'        -> 'charset_normalizer'
      'pkg[extra1,extra2]>=1.0'         -> 'pkg'
    """
    # Отбросить environment marker
    requirement = requirement.split(";", 1)[0].strip()
    if not requirement:
        return None

    # Отбросить extras: pkg[extra] -> pkg
    if "[" in requirement:
        requirement = requirement.split("[", 1)[0]

    m = _req_name_re.match(requirement)
    if not m:
        return None
    return m.group(0)


def get_direct_dependencies_real(cfg: Config) -> List[str]:
    """Этап 2: прямые зависимости для заданной версии из реального pip-репо."""
    raw_reqs = fetch_requires_dist_from_pypi(
        package=cfg.package_name,
        version=cfg.package_version,
        base_url=cfg.repo_url,
    )
    deps: List[str] = []
    for r in raw_reqs:
        name = parse_requirement_name(r)
        if name:
            deps.append(name)
    return deps


# ============================
#  Тестовый репозиторий (Этап 3)
# ============================

def load_test_repository(path: str) -> Dict[str, List[str]]:
    """
    Формат файла тестового репозитория, например:

      # комментарии
      A: B C
      B: C
      C:
      D: A C

    Имена пакетов — заглавные латинские буквы (по условию).
    """
    repo: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise RuntimeError(
                    f"Ожидался формат 'A: B C', строка {lineno}: {line!r}"
                )
            left, right = line.split(":", 1)
            pkg = left.strip()
            if not re.match(r"^[A-Z]+$", pkg):
                raise RuntimeError(
                    f"Имя тестового пакета должно быть заглавными латинскими буквами, строка {lineno}"
                )
            deps = right.strip().split() if right.strip() else []
            repo[pkg] = deps
    return repo


def get_direct_dependencies_test(pkg: str, repo: Dict[str, List[str]]) -> List[str]:
    if pkg not in repo:
        raise RuntimeError(f"Пакет {pkg!r} не найден в тестовом репозитории")
    return repo[pkg]


# ============================
#  Построение графа (Этап 3)
# ============================

def get_direct_deps_any_version(pkg: str, base_url: str) -> List[str]:
    """
    Для транзитивных зависимостей, где конкретная версия не указана,
    можно (упрощённо) взять метаданные "последней версии".

    GET {base_url}/pypi/{pkg}/json
    """
    url = f"{base_url}/pypi/{pkg}/json"
    try:
        with urllib.request.urlopen(url) as resp:
            if resp.status != 200:
                return []
            data = json.load(resp)
    except Exception:
        return []

    info = data.get("info") or {}
    requires = info.get("requires_dist") or []
    result: List[str] = []
    for r in requires:
        if not isinstance(r, str):
            continue
        name = parse_requirement_name(r)
        if name:
            result.append(name)
    return result


def build_graph(cfg: Config) -> Dict[str, List[str]]:
    """
    Построить граф зависимостей с учётом транзитивности (Этап 3).
    DFS с рекурсией, обработка циклов.
    """
    graph: Dict[str, List[str]] = {}
    visited: Set[str] = set()
    stack: Set[str] = set()

    test_repo: Optional[Dict[str, List[str]]] = None
    if cfg.test_mode:
        test_repo = load_test_repository(cfg.test_repo_path)

    def dfs(pkg: str, is_root: bool = False) -> None:
        if pkg in stack:
            print(f"[WARN] Обнаружен цикл зависимостей, узел {pkg} уже в стеке")
            return
        if pkg in visited:
            return

        stack.add(pkg)

        # берем прямые зависимости
        if cfg.test_mode:
            assert test_repo is not None
            deps = test_repo.get(pkg, [])
        else:
            if is_root:
                # для корневого пакета обязана использоваться версия из конфига
                deps = get_direct_dependencies_real(cfg)
            else:
                # упрощённо — берем зависимости "последней" версии
                deps = get_direct_deps_any_version(pkg, cfg.repo_url)

        graph[pkg] = deps

        for d in deps:
            dfs(d, is_root=False)

        stack.remove(pkg)
        visited.add(pkg)

    dfs(cfg.package_name, is_root=True)
    return graph


# ============================
#  Порядок загрузки (Этап 4)
# ============================

def topo_sort(graph: Dict[str, List[str]], root: str) -> List[str]:
    """
    Топологическая сортировка (DFS postorder).
    Если найдём цикл — просто предупредим, но что-то всё равно выведем.
    """
    visited: Set[str] = set()
    stack: Set[str] = set()
    order: List[str] = []

    def dfs(node: str) -> None:
        if node in stack:
            print(f"[WARN] Цикл в графе, порядок может быть неточным, узел {node}")
            return
        if node in visited:
            return
        stack.add(node)
        for dep in graph.get(node, []):
            dfs(dep)
        stack.remove(node)
        visited.add(node)
        order.append(node)

    dfs(root)
    order.reverse()
    return order


# ============================
#  Визуализация (Этап 5)
# ============================

def graph_to_mermaid(graph: Dict[str, List[str]]) -> str:
    """
    Генерация текста диаграммы на языке Mermaid (Этап 5.1).
    """
    lines = ["graph TD"]
    for src, dests in graph.items():
        if not dests:
            lines.append(f"    {src}")
        else:
            for dst in dests:
                lines.append(f"    {src} --> {dst}")
    return "\n".join(lines)


def save_svg_from_mermaid(mermaid_text: str, svg_path: str) -> None:
    """
    Сохранить SVG используя утилиту mermaid-cli (mmdc).
    Это не менеджер пакетов и не библиотека для зависимостей, так что по условию можно.

    Требуется:
      npm install -g @mermaid-js/mermaid-cli
    """
    mmd_path = svg_path + ".mmd"
    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(mermaid_text)

    try:
        subprocess.run(["mmdc", "-i", mmd_path, "-o", svg_path], check=True)
    except FileNotFoundError:
        print(
            "[WARN] Команда 'mmdc' (mermaid-cli) не найдена. "
            "SVG не сгенерирован, но .mmd-файл создан.",
            file=sys.stderr,
        )
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Ошибка при генерации SVG: {e}", file=sys.stderr)


# ============================
#  CLI-оболочка
# ============================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Инструмент визуализации графа зависимостей pip-пакетов"
    )
    parser.add_argument("config", help="Путь к XML-файлу конфигурации")
    parser.add_argument(
        "--mode",
        choices=[
            "print-config",   # Этап 1
            "direct-deps",    # Этап 2
            "graph",          # Этап 3
            "load-order",     # Этап 4
            "visualize",      # Этап 5
        ],
        default="print-config",
        help="Режим работы приложения",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"Ошибка конфигурации: {e}", file=sys.stderr)
        return 1

    if args.mode == "print-config":
        # Этап 1: просто вывести параметры
        print_config(cfg)
        return 0

    if args.mode == "direct-deps":
        # Этап 2: получить и вывести прямые зависимости корневого пакета
        if cfg.test_mode:
            repo = load_test_repository(cfg.test_repo_path)
            deps = get_direct_dependencies_test(cfg.package_name, repo)
        else:
            deps = get_direct_dependencies_real(cfg)
        print(f"Прямые зависимости для {cfg.package_name}=={cfg.package_version}:")
        for d in deps:
            print(f"  - {d}")
        return 0

    # Для остальных режимов сначала строим граф (Этап 3)
    graph = build_graph(cfg)

    if args.mode == "graph":
        print("Граф зависимостей:")
        for src, dests in graph.items():
            deps_str = ", ".join(dests) if dests else "—"
            print(f"  {src}: {deps_str}")
        return 0

    if args.mode == "load-order":
        # Этап 4: порядок загрузки / установки (топологическая сортировка)
        order = topo_sort(graph, cfg.package_name)
        print(f"Порядок загрузки зависимостей для {cfg.package_name}:")
        for name in order:
            print(f"  - {name}")
        return 0

    if args.mode == "visualize":
        # Этап 5: Mermaid + SVG
        mermaid_text = graph_to_mermaid(graph)
        print("Mermaid-представление графа:")
        print(mermaid_text)
        save_svg_from_mermaid(mermaid_text, cfg.output_svg)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())