"""
I/O Utilities for CommentCourt (moved to `utils.io`).
"""

import json
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator
from datetime import datetime
import shutil
import gzip
import yaml


logger = logging.getLogger(__name__)


class FileHandler:
    SUPPORTED_FORMATS = {'.json', '.csv', '.yaml', '.yml', '.txt'}

    @staticmethod
    def read(path: Union[str, Path], encoding: str = 'utf-8') -> Any:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix == '.json':
            return FileHandler.read_json(path, encoding)
        elif suffix == '.csv':
            return FileHandler.read_csv(path, encoding)
        elif suffix in {'.yaml', '.yml'}:
            return FileHandler.read_yaml(path, encoding)
        else:
            return FileHandler.read_text(path, encoding)

    @staticmethod
    def write(path: Union[str, Path], data: Any, encoding: str = 'utf-8') -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == '.json':
            FileHandler.write_json(path, data, encoding)
        elif suffix == '.csv':
            FileHandler.write_csv(path, data, encoding)
        elif suffix in {'.yaml', '.yml'}:
            FileHandler.write_yaml(path, data, encoding)
        else:
            FileHandler.write_text(path, data, encoding)

    @staticmethod
    def read_json(path: Union[str, Path], encoding: str = 'utf-8') -> Any:
        with open(path, 'r', encoding=encoding) as f:
            return json.load(f)

    @staticmethod
    def write_json(path: Union[str, Path], data: Any, 
                   encoding: str = 'utf-8', indent: int = 2) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"Wrote JSON file: {path}")

    @staticmethod
    def read_csv(path: Union[str, Path], encoding: str = 'utf-8') -> List[Dict[str, str]]:
        rows = []
        with open(path, 'r', encoding=encoding, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    @staticmethod
    def write_csv(path: Union[str, Path], data: List[Dict[str, Any]], 
                  encoding: str = 'utf-8') -> None:
        if not data:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(data[0].keys())
        with open(path, 'w', encoding=encoding, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logger.debug(f"Wrote CSV file: {path}")

    @staticmethod
    def read_yaml(path: Union[str, Path], encoding: str = 'utf-8') -> Any:
        with open(path, 'r', encoding=encoding) as f:
            return yaml.safe_load(f)

    @staticmethod
    def write_yaml(path: Union[str, Path], data: Any, 
                   encoding: str = 'utf-8') -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding=encoding) as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        logger.debug(f"Wrote YAML file: {path}")

    @staticmethod
    def read_text(path: Union[str, Path], encoding: str = 'utf-8') -> str:
        with open(path, 'r', encoding=encoding) as f:
            return f.read()

    @staticmethod
    def write_text(path: Union[str, Path], data: str, 
                   encoding: str = 'utf-8') -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding=encoding) as f:
            f.write(data)
        logger.debug(f"Wrote text file: {path}")

    @staticmethod
    def read_jsonl(path: Union[str, Path], encoding: str = 'utf-8') -> Iterator[Dict]:
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @staticmethod
    def write_jsonl(path: Union[str, Path], data: List[Dict], 
                    encoding: str = 'utf-8') -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding=encoding) as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.debug(f"Wrote JSONL file: {path}")


class DataExporter:
    def __init__(self, output_dir: Union[str, Path] = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_influencers(self, influencers: List[Dict[str, Any]], format: str = 'json') -> Path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"influencers_{timestamp}.{format}"
        filepath = self.output_dir / filename
        FileHandler.write(filepath, influencers)
        logger.info(f"Exported {len(influencers)} influencers to {filepath}")
        return filepath

    def export_comments(self, comments: List[Dict[str, Any]], format: str = 'json') -> Path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"comments_{timestamp}.{format}"
        filepath = self.output_dir / filename
        FileHandler.write(filepath, comments)
        logger.info(f"Exported {len(comments)} comments to {filepath}")
        return filepath

    def export_analysis_report(self, report: Dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"analysis_report_{timestamp}.json"
        filepath = self.output_dir / filename
        FileHandler.write_json(filepath, report)
        logger.info(f"Exported analysis report to {filepath}")
        return filepath


class DataImporter:
    @staticmethod
    def import_scraped_data(path: Union[str, Path]) -> List[Dict[str, Any]]:
        path = Path(path)
        data = []
        if path.is_file():
            files = [path]
        else:
            files = list(path.glob('*.json'))
        for file in files:
            try:
                file_data = FileHandler.read_json(file)
                data.append(file_data)
                logger.debug(f"Imported: {file}")
            except Exception as e:
                logger.error(f"Error importing {file}: {e}")
        logger.info(f"Imported {len(data)} data files from {path}")
        return data

    @staticmethod
    def import_training_data(path: Union[str, Path]) -> List[Dict[str, str]]:
        path = Path(path)
        if path.suffix == '.csv':
            return FileHandler.read_csv(path)
        elif path.suffix == '.json':
            return FileHandler.read_json(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")


class BackupManager:
    def __init__(self, backup_dir: Union[str, Path] = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_database(self, db_path: Union[str, Path]) -> Path:
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"db_backup_{timestamp}.sqlite3.gz"
        backup_path = self.backup_dir / backup_name
        with open(db_path, 'rb') as f_in:
            with gzip.open(backup_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"Created database backup: {backup_path}")
        return backup_path

    def restore_database(self, backup_path: Union[str, Path], target_path: Union[str, Path]) -> None:
        backup_path = Path(backup_path)
        target_path = Path(target_path)
        if backup_path.suffix == '.gz':
            with gzip.open(backup_path, 'rb') as f_in:
                with open(target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
