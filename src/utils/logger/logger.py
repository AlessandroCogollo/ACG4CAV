from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union, Tuple

REASON_CODES = {
    "ACCEPT",
    "LOW_BOX_SCORE",
    "LOW_MASK_SCORE",
    "BOX_TOO_SMALL",
    "BOX_TOO_LARGE",
    "BAD_ASPECT_RATIO",
    "TOUCHING_BORDER",
    "MASK_TOO_SMALL",
    "MASK_TOO_LARGE",
    "MASK_FRAGMENTED",
    "MASK_HOLES",
    "CLIP_LOW",
    "VLM_NO",
    "DUPLICATE_NEAR",
    "DUPLICATE_CLUSTER",
    "CAP_PER_IMAGE",
    "CAP_PER_PROMPT",
    "CAP_PER_CLUSTER",
    "UNDERSUPPLY_RELAXED",
    "ERROR",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_config(config: Dict[str, Any]) -> str:
    blob = _stable_json_dumps(config).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _open_append(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8")


@dataclass
class RunMeta:
    project: str
    run_id: str
    config_hash: str
    started_utc: str
    concept: str
    python: str = field(default_factory=lambda: sys.version.split()[0])

@dataclass
class ImageGroundingRecord:
    run_id: str
    concept: str
    prompt: str
    image_id: str
    box: List[int]  # [x1,y1,x2,y2]
    box_score: float

@dataclass
class GrounderRecord:
    run_id: str
    processor: str
    model: str
    prompt: str
    threshold = float
    text_threshold = float
    target_sizes = Tuple[int, int]
    image_grounding_results = List[ImageGroundingRecord]  # e.g., [{"box": [x1,y1,x2,y2], "score": float, "label": int}, ...]

# TODO: implement filtering record if needed (e.g., with mask info, reason codes, etc.)

class ExperimentLogger:
    """
    Usage:
        logger = ExperimentLogger(
            root_dir="runs",
            project="acg4cav",
            config=config_dict,
        )
        logger.event("INFO", "run_started", {"notes": "..."})
        logger.log_candidate(CandidateRecord(...))
        logger.flush()
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        project: str,
        config: Dict[str, Any],
        run_name: Optional[str] = None,
        also_print: bool = True,
    ) -> None:
        self.root_dir = _ensure_dir(root_dir)
        self.project = project
        self.config = config
        self.config_hash = _hash_config(config)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = run_name or f"{project}"
        self.run_id = f"{base}-{ts}-{self.config_hash}"

        self.run_dir = _ensure_dir(self.root_dir / self.run_id)
        self.paths = {
            "events": self.run_dir / "events.jsonl",
            "candidates": self.run_dir / "candidates.jsonl",
            "metrics": self.run_dir / "metrics.json",
            "config": self.run_dir / "config.json",
            "run_meta": self.run_dir / "run_meta.json",
        }

        self.also_print = also_print
        self._events_f = _open_append(self.paths["events"])
        self._cands_f = _open_append(self.paths["candidates"])

        self.meta = RunMeta(
            project=project,
            run_id=self.run_id,
            config_hash=self.config_hash,
            started_utc=_utc_now_iso(),
            extra={},
        )

        # Snapshot config + run meta (atomic)
        _atomic_write_text(self.paths["config"], _stable_json_dumps(config) + "\n")
        _atomic_write_text(self.paths["run_meta"], _stable_json_dumps(dataclasses.asdict(self.meta)) + "\n")

        self.event("INFO", "run_started", {"run_id": self.run_id, "config_hash": self.config_hash})

        # In-memory counters for quick summaries
        self._reason_counts: Dict[str, int] = {}
        self._accepted_count = 0
        self._total_candidates = 0

    def event(self, level: str, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        rec = {
            "ts_utc": _utc_now_iso(),
            "level": level,
            "event": name,
            "run_id": self.run_id,
            "payload": payload or {},
        }
        line = _stable_json_dumps(rec)
        self._events_f.write(line + "\n")
        if self.also_print:
            # human-friendly single line
            msg = f"[{rec['ts_utc']}] {level} {name} {payload or {}}"
            print(msg)

    def log_candidate(self, cand: CandidateRecord) -> None:
        # Validate reason codes (soft; don't crash pipeline)
        for rc in cand.reason_codes:
            if rc not in REASON_CODES:
                self.event("WARN", "unknown_reason_code", {"reason_code": rc, "known": sorted(REASON_CODES)[:8]})

        self._total_candidates += 1
        if cand.accepted:
            self._accepted_count += 1

        for rc in cand.reason_codes:
            self._reason_counts[rc] = self._reason_counts.get(rc, 0) + 1

        # Ensure run_id is consistent
        cand.run_id = self.run_id
        line = _stable_json_dumps(cand.to_dict())
        self._cands_f.write(line + "\n")

    def metric(self, key: str, value: Any, step: Optional[int] = None) -> None:
        """
        Writes/updates metrics.json (atomic rewrite), good for end-of-stage snapshots.
        """
        cur = {}
        if self.paths["metrics"].exists():
            try:
                cur = json.loads(self.paths["metrics"].read_text(encoding="utf-8"))
            except Exception:
                cur = {}

        entry = {"value": value, "ts_utc": _utc_now_iso()}
        if step is not None:
            entry["step"] = step
        cur[key] = entry
        _atomic_write_text(self.paths["metrics"], _stable_json_dumps(cur) + "\n")

    @contextmanager
    def timer(self, name: str, payload: Optional[Dict[str, Any]] = None):
        start = time.time()
        self.event("INFO", f"{name}_start", payload or {})
        try:
            yield
        except Exception as e:
            self.event("ERROR", f"{name}_error", {"error": repr(e)})
            raise
        finally:
            dur_s = time.time() - start
            self.event("INFO", f"{name}_end", {"duration_s": round(dur_s, 6), **(payload or {})})
            self.metric(f"time/{name}_s", dur_s)

    def summary(self) -> Dict[str, Any]:
        acceptance_rate = (self._accepted_count / self._total_candidates) if self._total_candidates else 0.0
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "total_candidates": self._total_candidates,
            "accepted": self._accepted_count,
            "acceptance_rate": acceptance_rate,
            "reason_counts": dict(sorted(self._reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        }

    def flush(self) -> None:
        self._events_f.flush()
        self._cands_f.flush()
        # also store a snapshot summary
        self.metric("summary", self.summary())

    def close(self) -> None:
        self.event("INFO", "run_finished", self.summary())
        self.flush()
        self._events_f.close()
        self._cands_f.close()


# -----------------------------
# Helper: build records from pipeline outputs
# -----------------------------

def make_run_meta(project: str, config: Dict[str, Any], concept: str) -> RunMeta:
    return RunMeta(
        project=project,
        run_id="",  # to be filled by logger
        config_hash=_hash_config(config),
        started_utc=_utc_now_iso(),
        concept=concept,
    )

def make_grounding_record(
    processor: str,
    model: str,
    prompt: str,
    threshold: float,
    text_threshold: float,
    target_sizes: Tuple[int, int],
    image_grounding_results: List[Dict[str, Any]],
) -> GrounderRecord:
    return GrounderRecord(
        run_id="",
        processor=processor,
        model=model,
        prompt=prompt,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
        image_grounding_results=[
            ImageGroundingRecord(
                run_id="",
                concept="",  # to be filled by logger
                prompt=prompt,
                image_id=res.get("image_id", ""),
                box=res["box"],
                box_score=res["score"],
            )
            for res in image_grounding_results
        ],
    )

# -----------------------------
# Example usage (remove in production)
# -----------------------------
if __name__ == "__main__":
    cfg = {
        "model": {"name": "vit-base", "layer": "block11"},
        "grounding": {"K": 2, "tau_box": 0.30},
        "segmentation": {"tau_mask": 0.85},
        "filtering": {"a_box": [0.01, 0.60], "n_cc_max": 2},
        "dedup": {"per_image_cap": 2, "cluster_cap": 3},
    }

    logger = ExperimentLogger(root_dir="runs", project="acg4cav", config=cfg, run_name="mvp", also_print=True)

    try:
        with logger.timer("grounding_stage", {"concepts": 12}):
            # pretend we found one candidate, rejected
            rec1 = make_candidate_record(
                concept="wheel",
                prompt="a wheel",
                image_id="img_0001",
                box=[10, 20, 200, 220],
                box_score=0.22,
                accepted=False,
                reason_codes=["LOW_BOX_SCORE"],
                stage="grounding",
                metrics={"img_w": 640, "img_h": 480, "box_area_ratio": 0.12, "aspect_ratio": 1.05},
            )
            logger.log_candidate(rec1)

        with logger.timer("filtering_stage"):
            # accepted candidate
            rec2 = make_candidate_record(
                concept="wheel",
                prompt="a wheel",
                image_id="img_0002",
                box=[30, 40, 180, 210],
                box_score=0.61,
                mask_score=0.92,
                crop_path="datasets/Dc_auto/wheel/img_0002_0.png",
                accepted=True,
                reason_codes=["ACCEPT"],
                stage="filtering",
                metrics={"img_w": 640, "img_h": 480, "box_area_ratio": 0.08, "mask_area_ratio_in_box": 0.74},
                score_composite=0.83,
            )
            logger.log_candidate(rec2)

        logger.flush()
        print("Summary:", logger.summary())

    finally:
        logger.close()
