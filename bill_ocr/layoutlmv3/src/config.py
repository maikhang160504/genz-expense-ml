import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class PathsConfig:
    checkpoint_dir: Path
    volume_checkpoint: Path
    evaluation_metrics: Path
    test_result_csv: Path


@dataclass
class SplitConfig:
    images_dir: Path
    csv_path: Path
    output_dir: Path


@dataclass
class LayoutLMv3Config:
    epochs: int = 5
    learning_rate: float = 5e-5
    seed: int = 42
    batch_size: int = 4
    max_seq_length: int = 512
    early_stop_patience: int = 3
    train: SplitConfig = None
    val: SplitConfig = None
    test: SplitConfig = None
    paths: PathsConfig = None

    @staticmethod
    def load_from_yaml(yaml_path: Path) -> "LayoutLMv3Config":
        with yaml_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # Helper to convert string paths to Path objects
        def to_path(p: Any) -> Path:
            return Path(p) if isinstance(p, str) else p

        split_cfg = lambda s: SplitConfig(
            images_dir=to_path(s["images_dir"]),
            csv_path=to_path(s["csv_path"]),
            output_dir=to_path(s["output_dir"]),
        )
        paths_cfg = PathsConfig(
            checkpoint_dir=to_path(cfg["paths"]["checkpoint_dir"]),
            volume_checkpoint=to_path(cfg["paths"]["volume_checkpoint"]),
            evaluation_metrics=to_path(cfg["paths"]["evaluation_metrics"]),
            test_result_csv=to_path(cfg["paths"]["test_result_csv"]),
        )
        return LayoutLMv3Config(
            epochs=cfg.get("epochs", 5),
            learning_rate=cfg.get("learning_rate", 5e-5),
            seed=cfg.get("seed", 42),
            batch_size=cfg.get("batch_size", 4),
            max_seq_length=cfg.get("max_seq_length", 512),
            num_workers=cfg.get("num_workers", 2),
            train=split_cfg(cfg["train"]),
            val=split_cfg(cfg["val"]),
            test=split_cfg(cfg["test"]),
            paths=paths_cfg,
        )
