from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path)


def _range_ok(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    problems: list[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        if not ((values >= 0) & (values <= 1)).all():
            problems.append(f"{col} has values outside [0, 1].")
    return problems


def _epoch_ok(df: pd.DataFrame, best_col: str = "best_epoch", trained_col: str = "trained_epochs") -> list[str]:
    if best_col not in df.columns or trained_col not in df.columns:
        return []
    rows = df[pd.to_numeric(df[best_col], errors="coerce").notna()].copy()
    if rows.empty:
        return []
    best = pd.to_numeric(rows[best_col], errors="coerce")
    trained = pd.to_numeric(rows[trained_col], errors="coerce")
    if ((best >= 1) & (best <= trained)).all():
        return []
    return [f"{best_col} must satisfy 1 <= best_epoch <= trained_epochs for all DL rows."]


def _checkpoint_ok(checkpoint_dir: Path) -> list[str]:
    if not checkpoint_dir.exists():
        return [f"Checkpoint dir missing: {checkpoint_dir}"]
    ckpts = list(checkpoint_dir.glob("*.pt"))
    if not ckpts:
        return [f"No .pt checkpoint files found in: {checkpoint_dir}"]
    empty = [p.name for p in ckpts if p.stat().st_size <= 0]
    if empty:
        return [f"Empty checkpoint files found: {empty}"]
    return []


def _find_task_dir(root: Path, script_name: str) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / script_name).exists()]
    if not candidates:
        raise FileNotFoundError(f"Cannot find task directory for script: {script_name}")
    return candidates[0]


def main() -> None:
    root = Path(__file__).resolve().parent
    issues: list[str] = []
    lines: list[str] = []

    sentiment_dir = _find_task_dir(root, "sentiment_analysis.py")
    reuters_dir = _find_task_dir(root, "reuters_multiclass.py")
    mt_dir = _find_task_dir(root, "machine_translation.py")

    sentiment_cfg = _read_json(sentiment_dir / "outputs" / "sentiment_config.json")
    sentiment_df = _read_csv(sentiment_dir / "outputs" / "sentiment_results.csv")
    issues += _range_ok(sentiment_df, ["accuracy", "precision", "recall", "f1", "best_val_f1"])
    issues += _epoch_ok(sentiment_df)
    issues += _checkpoint_ok(sentiment_dir / "outputs" / "checkpoints")
    lines.append(f"Sentiment quick={sentiment_cfg.get('quick', None)}")

    reuters_cfg = _read_json(reuters_dir / "outputs" / "reuters_config.json")
    reuters_df = _read_csv(reuters_dir / "outputs" / "reuters_results.csv")
    issues += _range_ok(reuters_df, ["accuracy", "f1_macro", "f1_weighted", "best_val_f1_macro"])
    issues += _epoch_ok(reuters_df)
    issues += _checkpoint_ok(reuters_dir / "outputs" / "checkpoints")
    lines.append(f"Reuters quick={reuters_cfg.get('quick', None)}")

    mt_cfg = _read_json(mt_dir / "outputs" / "translation_config.json")
    mt_df = _read_csv(mt_dir / "outputs" / "translation_results.csv")
    issues += _range_ok(mt_df, ["BLEU-1", "BLEU-2", "BLEU-4", "best_val_BLEU-4"])
    issues += _epoch_ok(mt_df)
    issues += _checkpoint_ok(mt_dir / "outputs" / "checkpoints")
    lines.append(f"MT quick={mt_cfg.get('quick', None)}")

    report = ["Output Audit Report", "==================", *lines, ""]
    if issues:
        report.append("Status: WARN")
        report.append("Issues:")
        report.extend(f"- {item}" for item in issues)
    else:
        report.append("Status: OK")
        report.append("No structural issues found.")

    out_report = root / "outputs_audit_report.txt"
    out_report.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print()
    print(f"Saved: {out_report}")


if __name__ == "__main__":
    main()
