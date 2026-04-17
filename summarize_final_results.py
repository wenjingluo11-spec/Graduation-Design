from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path)


def _best_row(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in columns: {list(df.columns)}")
    return df.sort_values(metric, ascending=False).iloc[0]


def _find_task_dir(root: Path, script_name: str) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / script_name).exists()]
    if not candidates:
        raise FileNotFoundError(f"Cannot find task directory for script: {script_name}")
    return candidates[0]


def main() -> None:
    root = Path(__file__).resolve().parent

    sentiment_dir = _find_task_dir(root, "sentiment_analysis.py")
    reuters_dir = _find_task_dir(root, "reuters_multiclass.py")
    mt_dir = _find_task_dir(root, "machine_translation.py")

    sentiment_cfg = _read_json(sentiment_dir / "outputs" / "sentiment_config.json")
    reuters_cfg = _read_json(reuters_dir / "outputs" / "reuters_config.json")
    mt_cfg = _read_json(mt_dir / "outputs" / "translation_config.json")

    sentiment_df = _read_csv(sentiment_dir / "outputs" / "sentiment_results.csv")
    reuters_df = _read_csv(reuters_dir / "outputs" / "reuters_results.csv")
    mt_df = _read_csv(mt_dir / "outputs" / "translation_results.csv")

    best_sentiment = _best_row(sentiment_df, "f1")
    best_reuters = _best_row(reuters_df, "f1_macro")
    best_mt = _best_row(mt_df, "BLEU-4")

    summary = pd.DataFrame(
        [
            {
                "task": "Sentiment (IMDB)",
                "is_full_run": bool(not sentiment_cfg.get("quick", False)),
                "primary_metric": "F1",
                "best_model": best_sentiment["model"],
                "score": float(best_sentiment["f1"]),
            },
            {
                "task": "News Multi-class (Reuters-46)",
                "is_full_run": bool(not reuters_cfg.get("quick", False)),
                "primary_metric": "Macro-F1",
                "best_model": best_reuters["model"],
                "score": float(best_reuters["f1_macro"]),
            },
            {
                "task": "Machine Translation (Es->En)",
                "is_full_run": bool(not mt_cfg.get("quick", False)),
                "primary_metric": "BLEU-4",
                "best_model": best_mt["model"],
                "score": float(best_mt["BLEU-4"]),
            },
        ]
    )

    out_csv = root / "final_results_summary.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False, float_format="%.6f"))
    print()
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
