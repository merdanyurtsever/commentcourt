"""
I/O Utilities for CommentCourt.

This module provides utilities for:
- File I/O operations (JSON, CSV, YAML)
- Data export/import
- Backup/restore functionality
- Safe file operations
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
    """
    Generic file handler with support for multiple formats.
    
    Supports JSON, CSV, YAML, and plain text files with
    automatic format detection.
    """
    
    SUPPORTED_FORMATS = {'.json', '.csv', '.yaml', '.yml', '.txt'}
    
    @staticmethod
    def read(path: Union[str, Path], encoding: str = 'utf-8') -> Any:
        """
        Read file with automatic format detection.
        
        Args:
            path: File path
            encoding: File encoding
            
        Returns:
            Parsed file content
        """
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
        """
        Write file with automatic format detection.
        
        Args:
            path: File path
            data: Data to write
            encoding: File encoding
        """
        path = Path(path)
        
        # Create parent directories
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
        """Read JSON file."""
        with open(path, 'r', encoding=encoding) as f:
            return json.load(f)
    
    @staticmethod
    def write_json(path: Union[str, Path], data: Any, 
                   encoding: str = 'utf-8', indent: int = 2) -> None:
        """Write JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.debug(f"Wrote JSON file: {path}")
    
    @staticmethod
    def read_csv(path: Union[str, Path], encoding: str = 'utf-8') -> List[Dict[str, str]]:
        """Read CSV file as list of dicts."""
        rows = []
        with open(path, 'r', encoding=encoding, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
    
    @staticmethod
    def write_csv(path: Union[str, Path], data: List[Dict[str, Any]], 
                  encoding: str = 'utf-8') -> None:
        """Write CSV file from list of dicts."""
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
        """Read YAML file."""
        with open(path, 'r', encoding=encoding) as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def write_yaml(path: Union[str, Path], data: Any, 
                   encoding: str = 'utf-8') -> None:
        """Write YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        logger.debug(f"Wrote YAML file: {path}")
    
    @staticmethod
    def read_text(path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """Read plain text file."""
        with open(path, 'r', encoding=encoding) as f:
            return f.read()
    
    @staticmethod
    def write_text(path: Union[str, Path], data: str, 
                   encoding: str = 'utf-8') -> None:
        """Write plain text file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            f.write(data)
        
        logger.debug(f"Wrote text file: {path}")
    
    @staticmethod
    def read_jsonl(path: Union[str, Path], encoding: str = 'utf-8') -> Iterator[Dict]:
        """Read JSON Lines file (one JSON object per line)."""
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    
    @staticmethod
    def write_jsonl(path: Union[str, Path], data: List[Dict], 
                    encoding: str = 'utf-8') -> None:
        """Write JSON Lines file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.debug(f"Wrote JSONL file: {path}")


class DataExporter:
    """
    Export data to various formats.
    
    Supports exporting influencer data, comments, and analysis results.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "exports"):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory for exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_influencers(self, influencers: List[Dict[str, Any]], 
                           format: str = 'json') -> Path:
        """
        Export influencer data.
        
        Args:
            influencers: List of influencer dicts
            format: Export format ('json', 'csv')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"influencers_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        FileHandler.write(filepath, influencers)
        logger.info(f"Exported {len(influencers)} influencers to {filepath}")
        
        return filepath
    
    def export_comments(self, comments: List[Dict[str, Any]], 
                        format: str = 'json') -> Path:
        """
        Export comment data.
        
        Args:
            comments: List of comment dicts
            format: Export format
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"comments_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        FileHandler.write(filepath, comments)
        logger.info(f"Exported {len(comments)} comments to {filepath}")
        
        return filepath
    
    def export_analysis_report(self, report: Dict[str, Any]) -> Path:
        """
        Export analysis report.
        
        Args:
            report: Analysis report dict
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"analysis_report_{timestamp}.json"
        filepath = self.output_dir / filename
        
        FileHandler.write_json(filepath, report)
        logger.info(f"Exported analysis report to {filepath}")
        
        return filepath


class DataImporter:
    """
    Import data from various sources.
    
    Handles importing scraped data, training data, and external datasets.
    """
    
    @staticmethod
    def import_scraped_data(path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Import scraped data from JSON files.
        
        Expected format per file:
        {
            "influencer": "Name",
            "platform": "trendyol",
            "comments": [...]
        }
        
        Args:
            path: File or directory path
            
        Returns:
            List of imported data dicts
        """
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
        """
        Import training data.
        
        Expected CSV format: text,label
        Expected JSON format: [{"text": "...", "label": "..."}]
        
        Args:
            path: Path to training data
            
        Returns:
            List of training samples
        """
        path = Path(path)
        
        if path.suffix == '.csv':
            return FileHandler.read_csv(path)
        elif path.suffix == '.json':
            return FileHandler.read_json(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")


class BackupManager:
    """
    Manages data backups.
    
    Provides backup and restore functionality for database and configuration.
    """
    
    def __init__(self, backup_dir: Union[str, Path] = "backups"):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Directory for backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def backup_database(self, db_path: Union[str, Path]) -> Path:
        """
        Create a backup of the database.
        
        Args:
            db_path: Path to database file
            
        Returns:
            Path to backup file
        """
        db_path = Path(db_path)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"db_backup_{timestamp}.sqlite3.gz"
        backup_path = self.backup_dir / backup_name
        
        # Compress backup
        with open(db_path, 'rb') as f_in:
            with gzip.open(backup_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        logger.info(f"Created database backup: {backup_path}")
        return backup_path
    
    def restore_database(self, backup_path: Union[str, Path], 
                         target_path: Union[str, Path]) -> None:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            target_path: Path to restore to
        """
        backup_path = Path(backup_path)
        target_path = Path(target_path)
        
        if backup_path.suffix == '.gz':
            with gzip.open(backup_path, 'rb') as f_in:
                with open(target_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(backup_path, target_path)
        
        logger.info(f"Restored database to: {target_path}")
    
    def backup_config(self, config_path: Union[str, Path]) -> Path:
        """
        Backup configuration file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Path to backup
        """
        config_path = Path(config_path)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"config_backup_{timestamp}{config_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(config_path, backup_path)
        
        logger.info(f"Created config backup: {backup_path}")
        return backup_path
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup info dicts
        """
        backups = []
        
        for file in self.backup_dir.iterdir():
            if file.is_file():
                stat = file.stat()
                backups.append({
                    'name': file.name,
                    'path': str(file),
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'type': 'database' if 'db_' in file.name else 'config'
                })
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x['created'], reverse=True)
        
        return backups
    
    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """
        Remove old backups, keeping only the most recent.
        
        Args:
            keep_count: Number of backups to keep per type
            
        Returns:
            Number of removed backups
        """
        backups = self.list_backups()
        
        db_backups = [b for b in backups if b['type'] == 'database']
        config_backups = [b for b in backups if b['type'] == 'config']
        
        removed = 0
        
        for backup_list in [db_backups, config_backups]:
            for backup in backup_list[keep_count:]:
                Path(backup['path']).unlink()
                removed += 1
                logger.debug(f"Removed old backup: {backup['name']}")
        
        if removed:
            logger.info(f"Cleaned up {removed} old backups")
        
        return removed


# Convenience functions

def read_file(path: Union[str, Path]) -> Any:
    """Read file with automatic format detection."""
    return FileHandler.read(path)


def write_file(path: Union[str, Path], data: Any) -> None:
    """Write file with automatic format detection."""
    FileHandler.write(path, data)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
