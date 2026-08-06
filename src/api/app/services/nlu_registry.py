"""NLU model registry: version string, pending accepts, meta for admin UI."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGISTRY_FILENAME = "nlu_model_registry.json"
DEFAULT_VERSION = "v1.1-global"


def _paths(nlu_root: Path) -> dict[str, Path]:
    use_storage = Path("/storage").is_dir()
    models = Path("/storage/nlu_models") if use_storage else (nlu_root / "text_nlu" / "models")
    return {
        "registry": models / REGISTRY_FILENAME,
        "history": models / "nlu_training_history.json",
        "metrics": models / "retrain_all_metrics.json",
        "intent_model": models / "intent_model.joblib",
        "datasets": nlu_root / "text_nlu" / "datasets",
    }


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_registry(nlu_root: Path) -> dict[str, Any]:
    p = _paths(nlu_root)["registry"]
    reg = _load_json(p, None)
    if not reg:
        reg = {
            "version": DEFAULT_VERSION,
            "major": 1,
            "minor": 1,
            "last_accepted_run_index": 0,
            "pending_run_index": None,
            "accepted_at": None,
            "intent_backend": "encoder",
            "category_backend": "llm_v2",
        }
        _save_json(p, reg)
    if "intent_backend" not in reg:
        reg["intent_backend"] = reg.get("inference_backend", "encoder")
    if "category_backend" not in reg:
        reg["category_backend"] = reg.get("inference_backend", "llm_v2")
    return reg


def get_intent_backend(nlu_root: Path) -> str:
    reg = load_registry(nlu_root)
    backend = str(reg.get("intent_backend", "encoder")).strip().lower()
    if backend in {"llm", "llm_v2"}:
        return backend
    return "encoder" if backend in {"encoder", "phobert"} else "tfidf"


def get_category_backend(nlu_root: Path) -> str:
    reg = load_registry(nlu_root)
    backend = str(reg.get("category_backend", "llm_v2")).strip().lower()
    if backend in {"llm", "llm_v2"}:
        return backend
    return "encoder" if backend in {"encoder", "phobert"} else "tfidf"


def set_inference_backends(nlu_root: Path, intent_backend: str, category_backend: str) -> dict[str, Any]:
    def normalize(b: str) -> str:
        b = str(b).strip().lower()
        if b not in {"tfidf", "encoder", "phobert", "llm", "llm_v2"}:
            raise ValueError(f"invalid backend '{b}', must be 'tfidf', 'encoder', 'llm', or 'llm_v2'")
        if b in {"llm", "llm_v2"}:
            return b
        return "encoder" if b in {"encoder", "phobert"} else "tfidf"

    reg = load_registry(nlu_root)
    reg["intent_backend"] = normalize(intent_backend)
    reg["category_backend"] = normalize(category_backend)
    save_registry(nlu_root, reg)
    return reg


def save_registry(nlu_root: Path, reg: dict[str, Any]) -> None:
    _save_json(_paths(nlu_root)["registry"], reg)


def _format_f1(value: float | int | str | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        if value.endswith("%"):
            return value
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, (int, float)):
        pct = value * 100 if value <= 1 else value
        return f"{pct:.1f}%"
    return str(value)


def _headline_f1_from_metrics(metrics: dict | None) -> str:
    if not metrics:
        return "N/A"
    cat = metrics.get("category") or {}
    raw = cat.get("weighted_f1", cat.get("accuracy"))
    return _format_f1(raw)


def mark_train_success(nlu_root: Path, run_index: int) -> dict[str, Any]:
    """Mark a training run as pending acceptance (reload to deploy version bump)."""
    reg = load_registry(nlu_root)
    if run_index > reg.get("last_accepted_run_index", 0):
        reg["pending_run_index"] = run_index
    save_registry(nlu_root, reg)
    return reg


def accept_pending_version(nlu_root: Path) -> dict[str, Any]:
    """Accept pending model on disk — bump v1.x-global and mark run accepted."""
    import shutil
    reg = load_registry(nlu_root)
    pending = reg.get("pending_run_index")
    last = reg.get("last_accepted_run_index", 0)

    # Physical directory promote if models_new exists
    models_dir = nlu_root / "text_nlu" / "models"
    models_new = nlu_root / "text_nlu" / "models_new"
    models_old = nlu_root / "text_nlu" / "models_old"

    if models_new.exists():
        try:
            if models_dir.exists():
                if models_old.exists():
                    shutil.rmtree(models_old, ignore_errors=True)
                shutil.copytree(models_dir, models_old)
                shutil.rmtree(models_dir, ignore_errors=True)
            shutil.copytree(models_new, models_dir)
            
            # Also copy to Modal persistent storage if mounted
            storage_models = Path("/storage/nlu_models")
            if Path("/storage").is_dir():
                print(f"[NLU Registry] Promoting to persistent storage {storage_models}...", flush=True)
                if storage_models.exists():
                    # Keep old storage just in case
                    storage_old = Path("/storage/nlu_models_old")
                    if storage_old.exists():
                        shutil.rmtree(storage_old, ignore_errors=True)
                    shutil.copytree(storage_models, storage_old)
                    shutil.rmtree(storage_models, ignore_errors=True)
                shutil.copytree(models_new, storage_models)

        except Exception as e:
            print(f"[NLU Registry] Warning while copying candidate model folder: {e}", flush=True)

    if pending and pending > last:
        minor = int(reg.get("minor", 1)) + 1
        major = int(reg.get("major", 1))
        reg["minor"] = minor
        reg["major"] = major
        reg["version"] = f"v{major}.{minor}-global"
        reg["last_accepted_run_index"] = pending
        reg["pending_run_index"] = None
        reg["accepted_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    elif reg.get("version") is None:
        reg["version"] = DEFAULT_VERSION
    else:
        minor = int(reg.get("minor", 1)) + 1
        major = int(reg.get("major", 1))
        reg["minor"] = minor
        reg["major"] = major
        reg["version"] = f"v{major}.{minor}-global"
        reg["accepted_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    save_registry(nlu_root, reg)
    return reg


def get_model_meta(nlu_root: Path, *, nlu_real: bool = True, nlu_loaded: bool = False) -> dict[str, Any]:
    paths = _paths(nlu_root)
    reg = load_registry(nlu_root)
    history = _load_json(paths["history"], [])
    metrics = _load_json(paths["metrics"], None)
    action_slots = None
    if isinstance(metrics, dict):
        action_slots = metrics.get("action_slots")
    if not action_slots:
        action_slots = _load_json(paths["metrics"].parent / "action_slots_metrics.json", None)

    trained_at = "Never"
    if paths["intent_model"].is_file():
        mtime = datetime.fromtimestamp(paths["intent_model"].stat().st_mtime, tz=timezone.utc)
        trained_at = mtime.strftime("%Y-%m-%d %H:%M:%S")

    training_rows = 0
    for name in ("intent_record.csv", "intent_action.csv", "intent_chitchat.csv"):
        csv_path = paths["datasets"] / name
        if csv_path.is_file():
            lines = [ln for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            training_rows += max(0, len(lines) - 1)

    latest = history[-1] if history else None
    f1 = _headline_f1_from_metrics(latest.get("metrics") if latest else metrics)

    version = reg.get("version", DEFAULT_VERSION)
    if not nlu_real:
        version = f"v{reg.get('major', 1)}.{reg.get('minor', 1)}-fallback-mock"
        f1 = "N/A (Mock)"

    pending = reg.get("pending_run_index")
    pending_note = None
    if pending and pending > reg.get("last_accepted_run_index", 0):
        pending_note = f"Run #{pending} chờ chấp nhận — bấm Tải lại model NLU"

    return {
        "version": version,
        "trainedAt": trained_at,
        "f1Score": f1,
        "trainingRows": training_rows,
        "loaded": nlu_loaded,
        "pendingRunIndex": pending,
        "pendingNote": pending_note,
        "lastAcceptedRunIndex": reg.get("last_accepted_run_index", 0),
        "actionSlots": action_slots,
        "intent_backend": get_intent_backend(nlu_root),
        "category_backend": get_category_backend(nlu_root),
    }


def bump_version_label(raw: str) -> str:
    m = re.match(r"^v(\d+)\.(\d+)-global$", raw or "")
    if not m:
        return DEFAULT_VERSION
    return f"v{m.group(1)}.{int(m.group(2))}-global"
