from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TaskSpec:
    name: str
    script: str
    result_file: str
    config_file: str
    primary_metric: str
    seed_arg: str
    fast_args: Sequence[str]


TASK_SPECS: Dict[str, TaskSpec] = {
    "sentiment": TaskSpec(
        name="sentiment",
        script="sentiment_analysis.py",
        result_file="sentiment_results.csv",
        config_file="sentiment_config.json",
        primary_metric="f1",
        seed_arg="--seed",
        fast_args=("--epochs", "4", "--max-train-samples", "4000", "--max-test-samples", "2000"),
    ),
    "reuters": TaskSpec(
        name="reuters",
        script="reuters_multiclass.py",
        result_file="reuters_results.csv",
        config_file="reuters_config.json",
        primary_metric="f1_macro",
        seed_arg="--seed",
        fast_args=("--epochs", "5", "--max-samples", "4500"),
    ),
    "translation": TaskSpec(
        name="translation",
        script="machine_translation.py",
        result_file="translation_results.csv",
        config_file="translation_config.json",
        primary_metric="BLEU-4",
        seed_arg="--seed",
        fast_args=("--epochs-seq2seq", "6", "--epochs-transformer", "6", "--max-samples", "9000"),
    ),
}


def _find_task_dir(root: Path, script_name: str) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / script_name).exists()]
    if not candidates:
        raise FileNotFoundError(f"Cannot find task dir for script: {script_name}")
    return candidates[0]


def _task_dirs(root: Path) -> Dict[str, Path]:
    return {name: _find_task_dir(root, spec.script) for name, spec in TASK_SPECS.items()}


def _run_task(
    *,
    root: Path,
    task_name: str,
    output_dir: Path,
    python_exe: str,
    device: str,
    extra_args: Sequence[str],
) -> Dict[str, float | str]:
    spec = TASK_SPECS[task_name]
    task_dir = _find_task_dir(root, spec.script)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        spec.script,
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        *[str(a) for a in extra_args],
    ]
    print(f"\n[RUN] {task_name}: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, cwd=task_dir, check=True)
    run_seconds = time.time() - t0

    results_path = output_dir / spec.result_file
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    df = pd.read_csv(results_path)
    best = df.sort_values(spec.primary_metric, ascending=False).iloc[0]
    return {
        "task": task_name,
        "metric": spec.primary_metric,
        "best_model": str(best["model"]),
        "score": float(best[spec.primary_metric]),
        "run_seconds": float(run_seconds),
        "output_dir": str(output_dir),
    }


def _save_line_plot(df: pd.DataFrame, x_col: str, y_col: str, group_col: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for g, part in df.groupby(group_col):
        part = part.sort_values(x_col)
        plt.plot(part[x_col], part[y_col], marker="o", label=str(g))
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_stability(
    *,
    root: Path,
    out_root: Path,
    python_exe: str,
    device: str,
    seeds: Sequence[int],
    fast: bool,
) -> None:
    rows: List[Dict[str, float | str]] = []
    base_dir = out_root / "stability"
    for seed in seeds:
        for task_name, spec in TASK_SPECS.items():
            extra = [spec.seed_arg, str(seed)]
            if fast:
                extra.extend(spec.fast_args)
            out_dir = base_dir / task_name / f"seed_{seed}"
            rec = _run_task(
                root=root,
                task_name=task_name,
                output_dir=out_dir,
                python_exe=python_exe,
                device=device,
                extra_args=extra,
            )
            rec["seed"] = int(seed)
            rows.append(rec)

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(base_dir / "stability_runs.csv", index=False, encoding="utf-8-sig")

    summary_df = (
        runs_df.groupby(["task", "metric"], as_index=False)["score"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary_df.to_csv(base_dir / "stability_summary.csv", index=False, encoding="utf-8-sig")

    _save_line_plot(
        runs_df,
        x_col="seed",
        y_col="score",
        group_col="task",
        title="Seed Stability (Best Score)",
        out_path=base_dir / "stability_curve.png",
    )


def run_ablation(
    *,
    root: Path,
    out_root: Path,
    python_exe: str,
    device: str,
    fast: bool,
) -> None:
    base_dir = out_root / "ablation"
    rows: List[Dict[str, float | str]] = []

    cls_variants = [
        ("baseline", []),
        ("no_early_stop", ["--patience", "999", "--min-delta", "0"]),
        ("strict_early_stop", ["--patience", "2", "--min-delta", "0.001"]),
    ]

    for task_name in ("sentiment", "reuters"):
        spec = TASK_SPECS[task_name]
        for variant_name, var_args in cls_variants:
            extra = [spec.seed_arg, "42", *var_args]
            if fast:
                extra.extend(spec.fast_args)
            out_dir = base_dir / task_name / variant_name
            rec = _run_task(
                root=root,
                task_name=task_name,
                output_dir=out_dir,
                python_exe=python_exe,
                device=device,
                extra_args=extra,
            )
            rows.append(
                {
                    "task": task_name,
                    "variant": variant_name,
                    "metric": rec["metric"],
                    "best_model": rec["best_model"],
                    "score": rec["score"],
                    "run_seconds": rec["run_seconds"],
                }
            )

    mt_variants = [
        ("baseline", []),
        (
            "no_early_stop",
            ["--patience-seq2seq", "999", "--patience-transformer", "999", "--min-delta-bleu", "0"],
        ),
    ]
    mt_spec = TASK_SPECS["translation"]
    for variant_name, var_args in mt_variants:
        extra = [mt_spec.seed_arg, "42", *var_args]
        if fast:
            extra.extend(mt_spec.fast_args)
        out_dir = base_dir / "translation" / variant_name
        rec = _run_task(
            root=root,
            task_name="translation",
            output_dir=out_dir,
            python_exe=python_exe,
            device=device,
            extra_args=extra,
        )
        rows.append(
            {
                "task": "translation",
                "variant": f"{variant_name}_best_beam",
                "metric": rec["metric"],
                "best_model": rec["best_model"],
                "score": rec["score"],
                "run_seconds": rec["run_seconds"],
            }
        )
        decode_path = out_dir / "translation_decode_ablation.csv"
        if decode_path.exists():
            decode_df = pd.read_csv(decode_path)
            for _, row in decode_df.iterrows():
                rows.append(
                    {
                        "task": "translation",
                        "variant": f"{variant_name}_{row['model']}_{row['decode_method']}",
                        "metric": "BLEU-4",
                        "best_model": row["model"],
                        "score": float(row["BLEU-4"]),
                        "run_seconds": np.nan,
                    }
                )

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(base_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")


def _read_base_config_value(config_path: Path, key: str, default: int) -> int:
    if not config_path.exists():
        return default
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return int(cfg.get(key, default))


def run_data_scale(
    *,
    root: Path,
    out_root: Path,
    python_exe: str,
    device: str,
    scales: Sequence[float],
    fast: bool,
) -> None:
    task_dirs = _task_dirs(root)
    base_dir = out_root / "data_scale"
    rows: List[Dict[str, float | str]] = []

    sent_cfg = task_dirs["sentiment"] / "outputs" / "sentiment_config.json"
    reu_cfg = task_dirs["reuters"] / "outputs" / "reuters_config.json"
    mt_cfg = task_dirs["translation"] / "outputs" / "translation_config.json"
    sent_train_base = _read_base_config_value(sent_cfg, "max_train_samples", 12000)
    sent_test_base = _read_base_config_value(sent_cfg, "max_test_samples", 5000)
    reuters_base = _read_base_config_value(reu_cfg, "max_samples", 11228)
    mt_base = _read_base_config_value(mt_cfg, "max_samples", 32000)

    for scale in scales:
        scale_tag = f"scale_{scale:.2f}".replace(".", "_")

        sent_train = max(800, int(sent_train_base * scale))
        sent_test = max(400, int(sent_test_base * scale))
        sent_extra: List[str] = [
            "--seed",
            "42",
            "--max-train-samples",
            str(sent_train),
            "--max-test-samples",
            str(sent_test),
        ]
        if fast:
            sent_extra.extend(["--epochs", "4"])
        rec = _run_task(
            root=root,
            task_name="sentiment",
            output_dir=base_dir / "sentiment" / scale_tag,
            python_exe=python_exe,
            device=device,
            extra_args=sent_extra,
        )
        rec["scale"] = float(scale)
        rows.append(rec)

        reu_samples = max(1500, int(reuters_base * scale))
        reu_extra: List[str] = ["--seed", "42", "--max-samples", str(reu_samples)]
        if fast:
            reu_extra.extend(["--epochs", "5"])
        rec = _run_task(
            root=root,
            task_name="reuters",
            output_dir=base_dir / "reuters" / scale_tag,
            python_exe=python_exe,
            device=device,
            extra_args=reu_extra,
        )
        rec["scale"] = float(scale)
        rows.append(rec)

        mt_samples = max(5000, int(mt_base * scale))
        mt_extra: List[str] = ["--seed", "42", "--max-samples", str(mt_samples)]
        if fast:
            mt_extra.extend(["--epochs-seq2seq", "6", "--epochs-transformer", "6"])
        rec = _run_task(
            root=root,
            task_name="translation",
            output_dir=base_dir / "translation" / scale_tag,
            python_exe=python_exe,
            device=device,
            extra_args=mt_extra,
        )
        rec["scale"] = float(scale)
        rows.append(rec)

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(base_dir / "data_scale_curve.csv", index=False, encoding="utf-8-sig")
    _save_line_plot(
        curve_df,
        x_col="scale",
        y_col="score",
        group_col="task",
        title="Data Scale Curve (Best Score)",
        out_path=base_dir / "data_scale_curve.png",
    )


def run_efficiency(*, root: Path, out_root: Path) -> None:
    task_dirs = _task_dirs(root)
    base_dir = out_root / "efficiency"
    base_dir.mkdir(parents=True, exist_ok=True)

    tables: List[pd.DataFrame] = []
    files = [
        (
            "sentiment",
            [
                task_dirs["sentiment"] / "outputs" / "sentiment_efficiency.csv",
                task_dirs["sentiment"] / "outputs_smoke" / "sentiment_efficiency.csv",
            ],
        ),
        (
            "reuters",
            [
                task_dirs["reuters"] / "outputs" / "reuters_efficiency.csv",
                task_dirs["reuters"] / "outputs_smoke" / "reuters_efficiency.csv",
            ],
        ),
        (
            "translation",
            [
                task_dirs["translation"] / "outputs" / "translation_efficiency.csv",
                task_dirs["translation"] / "outputs_smoke" / "translation_efficiency.csv",
            ],
        ),
    ]
    for task, candidates in files:
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            print(f"[WARN] Missing efficiency file for {task}: {candidates[0]}")
            continue
        df = pd.read_csv(path)
        df.insert(0, "task", task)
        tables.append(df)
        print(f"[INFO] Use efficiency file for {task}: {path}")

    if not tables:
        print("[WARN] No efficiency files found. Skip efficiency summary.")
        return

    out_df = pd.concat(tables, ignore_index=True)
    out_df.to_csv(base_dir / "efficiency_summary.csv", index=False, encoding="utf-8-sig")


def run_error_analysis(*, root: Path, out_root: Path) -> None:
    task_dirs = _task_dirs(root)
    base_dir = out_root / "error_analysis"
    base_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "sentiment_error_summary": [
            task_dirs["sentiment"] / "outputs" / "sentiment_error_summary.csv",
            task_dirs["sentiment"] / "outputs_smoke" / "sentiment_error_summary.csv",
        ],
        "reuters_top_confusions": [
            task_dirs["reuters"] / "outputs" / "reuters_top_confusions.csv",
            task_dirs["reuters"] / "outputs_smoke" / "reuters_top_confusions.csv",
        ],
        "translation_error_summary": [
            task_dirs["translation"] / "outputs" / "translation_error_summary.csv",
            task_dirs["translation"] / "outputs_smoke" / "translation_error_summary.csv",
        ],
    }

    lines: List[str] = ["# Error Analysis Report", ""]
    for key, candidates in paths.items():
        path = next((p for p in candidates if p.exists()), None)
        lines.append(f"## {key}")
        if path is None:
            lines.append(f"- Missing: `{candidates[0]}`")
            lines.append("")
            continue
        df = pd.read_csv(path)
        preview = df.head(12)
        lines.append(f"- Source: `{path}`")
        lines.append("")
        lines.append("```text")
        lines.append(preview.to_csv(index=False).strip())
        lines.append("```")
        lines.append("")

    report_path = base_dir / "error_analysis_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5 supplementary experiment packages for graduation project.")
    parser.add_argument("--python-exe", type=str, default="python")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407])
    parser.add_argument("--scales", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--fast", action="store_true", help="Use reduced epochs for quicker supplementary runs.")
    parser.add_argument("--all", action="store_true", help="Run all supplementary packages.")
    parser.add_argument("--run-stability", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--run-error-analysis", action="store_true")
    parser.add_argument("--run-efficiency", action="store_true")
    parser.add_argument("--run-data-scale", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out_root = root / "supplementary_outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    if not any(
        [
            args.all,
            args.run_stability,
            args.run_ablation,
            args.run_error_analysis,
            args.run_efficiency,
            args.run_data_scale,
        ]
    ):
        raise SystemExit("Please specify at least one section flag, e.g. --all or --run-stability.")

    if args.all or args.run_stability:
        run_stability(
            root=root,
            out_root=out_root,
            python_exe=args.python_exe,
            device=args.device,
            seeds=args.seeds,
            fast=args.fast,
        )

    if args.all or args.run_ablation:
        run_ablation(
            root=root,
            out_root=out_root,
            python_exe=args.python_exe,
            device=args.device,
            fast=args.fast,
        )

    if args.all or args.run_data_scale:
        run_data_scale(
            root=root,
            out_root=out_root,
            python_exe=args.python_exe,
            device=args.device,
            scales=args.scales,
            fast=args.fast,
        )

    if args.all or args.run_error_analysis:
        run_error_analysis(root=root, out_root=out_root)

    if args.all or args.run_efficiency:
        run_efficiency(root=root, out_root=out_root)

    print("\nDone. Supplementary outputs saved to:")
    print(out_root)


if __name__ == "__main__":
    main()
