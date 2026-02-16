# logger.py

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ExperimentLogger:
    def __init__(self, base_runs_dir: Path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = base_runs_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.metrics_path = self.run_dir / "metrics.json"
        self.dataset_stats_path = self.run_dir / "dataset_stats.json"

        # Initialize empty files
        self._write_json(self.metrics_path, {})
        self._write_json(self.dataset_stats_path, {})

    # -------------------------
    # Utility
    # -------------------------

    def _write_json(self, path: Path, content: Dict[str, Any]):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=4)

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------
    # Config snapshot
    # -------------------------

    def save_config(self, config_path: Path):
        shutil.copy(config_path, self.run_dir / "config_snapshot.yaml")

    # -------------------------
    # Dataset stats
    # -------------------------

    def log_dataset_stats(self, concept_name: str, stats: Dict[str, Any]):
        data = self._read_json(self.dataset_stats_path)
        data[concept_name] = stats
        self._write_json(self.dataset_stats_path, data)

    # -------------------------
    # Metrics logging
    # -------------------------

    def log_metrics(self, concept_name: str, metrics: Dict[str, Any]):
        data = self._read_json(self.metrics_path)
        data[concept_name] = metrics
        self._write_json(self.metrics_path, data)

    # -------------------------
    # Save artifacts
    # -------------------------

    def save_numpy(self, array, filename: str):
        import numpy as np
        np.save(self.run_dir / filename, array)

    def get_run_dir(self) -> Path:
        return self.run_dir
