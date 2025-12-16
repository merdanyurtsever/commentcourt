"""Small, readable I/O helpers.

Keep the common read/write helpers minimal so the codebase is easy
to understand and maintain. This module focuses on JSON/YAML/CSV/text
and JSONL helpers which are sufficient for the pipeline and GUI.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Iterator
import yaml


logger = logging.getLogger(__name__)


class FileHandler:
    """Small set of file helpers used across the project."""

    @staticmethod
    def read_json(path: Union[str, Path]) -> Any:
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def write_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def read_yaml(path: Union[str, Path]) -> Any:
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def write_yaml(path: Union[str, Path], data: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def read_csv(path: Union[str, Path]) -> List[Dict[str, str]]:
        rows = []
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    @staticmethod
    def write_csv(path: Union[str, Path], data: List[Dict[str, Any]]) -> None:
        if not data:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(data[0].keys())
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    @staticmethod
    def read_jsonl(path: Union[str, Path]) -> Iterator[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def write_jsonl(path: Union[str, Path], data: List[Dict]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


# Convenience helpers
def read_json(path: Union[str, Path]) -> Any:
    return FileHandler.read_json(path)


def write_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    FileHandler.write_json(path, data, indent)
