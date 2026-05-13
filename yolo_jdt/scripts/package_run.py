"""Package a Lightning training run into a self-contained, archival folder.

Each training run lives at `runs/<group>/<run_name>/` and after this script
runs, the folder contains EVERYTHING needed to understand what happened
without re-querying WandB or hunting in /tmp:

    runs/<group>/<run_name>/
    +-- RUN_INFO.md                  # human-readable summary (this script)
    +-- config.yaml                  # full Hydra/wandb config snapshot
    +-- train.log                    # full stdout from training (if --log-source given)
    +-- detection_map.json           # eval results (if --eval-json given)
    +-- weights/
    |   +-- promoted.pt              # symlink to weights/ours/<...>.pt
    +-- epoch=NNN-stepstep=SSS.ckpt  # Lightning best ckpt (already there)
    +-- last.ckpt                    # Lightning latest ckpt (already there)
    +-- wandb/                       # WandB local dir (already there)

Idempotent: re-run safely overwrites RUN_INFO.md / config.yaml / weights/promoted.pt
but never touches Lightning ckpts or wandb/.

Usage:
    # Backfill an existing run
    python -m yolo_jdt.scripts.package_run \\
        --run-dir runs/baselines/step3a_yolo11s_70ep \\
        --log-source /tmp/step3a.log \\
        --eval-json runs/baselines/detection_map.json \\
        --promoted weights/ours/yolo11s_det.pt \\
        --notes "Step 3.A baseline 70ep single-GPU"

    # Called automatically at end of run_step3a_overnight.sh for new runs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml


def _wandb_paths(run_dir: Path) -> tuple[Path | None, Path | None, str | None, str | None]:
    """Return (config.yaml, wandb-summary.json, run_id, run_url) if available."""
    wandb_dir = run_dir / "wandb" / "latest-run" / "files"
    cfg = wandb_dir / "config.yaml"
    summary = wandb_dir / "wandb-summary.json"
    metadata = wandb_dir / "wandb-metadata.json"

    run_id, run_url = None, None
    if metadata.exists():
        try:
            md = json.loads(metadata.read_text())
            entity = md.get("entity") or md.get("username")
            project = md.get("project")
            run_id_md = md.get("id") or md.get("runId")
            if entity and project and run_id_md:
                run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id_md}"
                run_id = run_id_md
        except (json.JSONDecodeError, OSError):
            pass

    if run_id is None:
        # fallback: parse from dir name run-YYYYMMDD_HHMMSS-<id>
        latest = (run_dir / "wandb" / "latest-run").resolve()
        if latest.exists():
            tail = latest.name.rsplit("-", 1)
            if len(tail) == 2:
                run_id = tail[1]

    return (cfg if cfg.exists() else None,
            summary if summary.exists() else None,
            run_id, run_url)


def _extract_summary_metrics(summary_path: Path) -> dict:
    try:
        d = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    keys_of_interest = sorted(k for k in d
        if k.startswith("val/") and not k.startswith("val/")
        or k.startswith("val/") and k.endswith(("/mAP", "/mAP50", "/mAP75",
                                                  "/mAR100", "/mAP_small",
                                                  "/mAP_medium", "/mAP_large")))
    return {k: d[k] for k in keys_of_interest}


def _format_metrics_table(metrics: dict) -> str:
    """Render val/<set>/<metric> dict as a markdown table grouped by set."""
    by_set: dict[str, dict[str, float]] = {}
    for k, v in metrics.items():
        if not k.startswith("val/"):
            continue
        parts = k.split("/")
        if len(parts) != 3:
            continue
        _, set_name, metric = parts
        by_set.setdefault(set_name, {})[metric] = v
    if not by_set:
        return "_(no val metrics found in WandB summary)_"

    cols = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large", "mAR100"]
    lines = ["| Val set | " + " | ".join(cols) + " |",
             "|---" + "|---" * len(cols) + "|"]
    for set_name, m in sorted(by_set.items()):
        row = [set_name] + [f"{m[c]:.4f}" if c in m and isinstance(m[c], (int, float)) else "—" for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _hparams_from_wandb_config(cfg_path: Path) -> dict:
    """Parse wandb config.yaml -> {section: {key: value}}.
    WandB stores values under {section: {value: <value>}}.
    """
    try:
        raw = yaml.safe_load(cfg_path.read_text())
    except (yaml.YAMLError, OSError):
        return {}
    out = {}
    for k, v in raw.items():
        if k == "_wandb":
            continue
        if isinstance(v, dict) and "value" in v:
            out[k] = v["value"]
        else:
            out[k] = v
    return out


def _format_hparams(hp: dict) -> str:
    """Render flat dict as `## Hyperparameters` section."""
    if not hp:
        return "_(no hyperparameters captured)_"
    lines = []
    for section, val in sorted(hp.items()):
        if isinstance(val, dict):
            lines.append(f"\n### {section}\n")
            lines.append("| Param | Value |")
            lines.append("|---|---|")
            for k, v in val.items():
                lines.append(f"| `{k}` | `{v}` |")
        else:
            lines.append(f"- `{section}`: `{val}`")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="The Lightning run dir (contains epoch=*.ckpt + wandb/)")
    ap.add_argument("--log-source", type=Path, default=None,
                    help="Path to training log file (e.g. /tmp/step3a.log) — copied in")
    ap.add_argument("--eval-json", type=Path, default=None,
                    help="Path to detection_map.json — copied in")
    ap.add_argument("--promoted", type=Path, default=None,
                    help="Path to promoted standalone weights — symlinked into weights/")
    ap.add_argument("--notes", type=str, default="",
                    help="Free-text notes appended to RUN_INFO.md")
    ap.add_argument("--phase", type=str, default="",
                    help="Phase ID for this run (e.g. '3.A', '4', '5.ABC')")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        sys.exit(f"[package_run] --run-dir is not a directory: {run_dir}")

    # ---- 1. WandB metadata ---------------------------------------------------
    wandb_cfg, wandb_summary, run_id, run_url = _wandb_paths(run_dir)
    metrics = _extract_summary_metrics(wandb_summary) if wandb_summary else {}
    hparams = _hparams_from_wandb_config(wandb_cfg) if wandb_cfg else {}

    # ---- 2. Copy artifacts ---------------------------------------------------
    if wandb_cfg:
        shutil.copy2(wandb_cfg, run_dir / "config.yaml")
        print(f"[package_run] copied config.yaml from wandb")

    if args.log_source:
        if args.log_source.exists():
            shutil.copy2(args.log_source, run_dir / "train.log")
            print(f"[package_run] copied train.log from {args.log_source}")
        else:
            print(f"[package_run] WARN: --log-source not found: {args.log_source}")

    if args.eval_json:
        if args.eval_json.exists():
            shutil.copy2(args.eval_json, run_dir / "detection_map.json")
            print(f"[package_run] copied detection_map.json from {args.eval_json}")
        else:
            print(f"[package_run] WARN: --eval-json not found: {args.eval_json}")

    if args.promoted:
        promoted_src = args.promoted.resolve()
        if promoted_src.exists():
            wdir = run_dir / "weights"
            wdir.mkdir(exist_ok=True)
            link = wdir / "promoted.pt"
            if link.exists() or link.is_symlink():
                link.unlink()
            # Relative symlink so the run dir is portable across project moves
            rel = os.path.relpath(promoted_src, link.parent)
            link.symlink_to(rel)
            print(f"[package_run] symlinked weights/promoted.pt -> {rel}")
        else:
            print(f"[package_run] WARN: --promoted not found: {args.promoted}")

    # ---- 3. Detect Lightning ckpts -------------------------------------------
    epoch_ckpts = sorted(p for p in run_dir.glob("epoch=*.ckpt") if p.name != "last.ckpt")
    last_ckpt = run_dir / "last.ckpt"
    best_ckpt = epoch_ckpts[-1] if epoch_ckpts else None

    # ---- 4. Read detection_map.json (if present) for the metrics block ------
    eval_metrics_str = ""
    eval_path = run_dir / "detection_map.json"
    if eval_path.exists():
        try:
            ev = json.loads(eval_path.read_text())
            eval_metrics_str = "\n## Final eval (FP32, from `eval_det.py`)\n\n"
            eval_metrics_str += "```json\n" + json.dumps(ev, indent=2) + "\n```\n"
        except json.JSONDecodeError:
            pass

    # ---- 5. Render RUN_INFO.md -----------------------------------------------
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_lines = [
        f"# Run: `{run_dir.name}`",
        "",
        f"_Packaged: {now}_  ",
    ]
    if args.phase:
        info_lines.append(f"_Phase: {args.phase}_  ")
    info_lines.append(f"_Path: `{run_dir.relative_to(run_dir.parents[2]) if len(run_dir.parents) >= 3 else run_dir}`_")
    info_lines.append("")

    # WandB block
    info_lines.append("## WandB")
    if run_url:
        info_lines.append(f"- **URL:** [{run_url}]({run_url})")
    if run_id:
        info_lines.append(f"- **Run ID:** `{run_id}`")
    if not run_url and not run_id:
        info_lines.append("_(no WandB run found in this dir)_")
    info_lines.append("")

    # Metrics
    info_lines.append("## Final WandB val metrics (last logged step)")
    info_lines.append(_format_metrics_table(metrics))
    info_lines.append("")

    if eval_metrics_str:
        info_lines.append(eval_metrics_str)

    # Notes
    if args.notes:
        info_lines.append("## Notes")
        info_lines.append(args.notes)
        info_lines.append("")

    # Hparams
    info_lines.append("## Hyperparameters (from WandB config snapshot)")
    info_lines.append(_format_hparams(hparams))
    info_lines.append("")

    # Files
    info_lines.append("## Files in this dir")
    info_lines.append("")
    info_lines.append("| File | Size | Purpose |")
    info_lines.append("|---|---|---|")
    artifact_rows = [
        ("RUN_INFO.md", "human-readable summary (this file)"),
        ("config.yaml", "Hydra/WandB config snapshot used for this run"),
        ("train.log", "full stdout from training (snapshot of /tmp log)"),
        ("detection_map.json", "FP32 EMA eval mAP (`eval_det.py` output)"),
        ("weights/promoted.pt", "symlink to standalone state_dict (consumed by downstream phases)"),
    ]
    if best_ckpt:
        artifact_rows.append((best_ckpt.name, "Lightning best ckpt (highest val mAP)"))
    if last_ckpt.exists():
        artifact_rows.append(("last.ckpt", "Lightning last ckpt (final epoch state)"))
    artifact_rows.append(("wandb/", "WandB local cache (config, summary, system metrics)"))

    for name, purpose in artifact_rows:
        p = run_dir / name
        if p.exists() or p.is_symlink():
            try:
                size = p.stat().st_size
                size_str = f"{size/1e6:.1f} MB" if size > 1e6 else f"{size/1e3:.1f} KB" if size > 1e3 else f"{size} B"
            except OSError:
                size_str = "—"
            info_lines.append(f"| `{name}` | {size_str} | {purpose} |")
        else:
            info_lines.append(f"| ~~`{name}`~~ | — | _(absent)_ |")

    info_lines.append("")

    # Reproducibility hint
    info_lines.append("## Reproduce")
    info_lines.append("")
    info_lines.append("Re-eval mAP from the promoted ckpt:")
    info_lines.append("```bash")
    info_lines.append("python -m yolo_jdt.scripts.eval_det \\")
    info_lines.append(f"    --output {run_dir.name}_reeval.json \\")
    info_lines.append(f"    --weights {run_dir}/weights/promoted.pt --scale s")
    info_lines.append("```")
    info_lines.append("")
    info_lines.append("Resume training from `last.ckpt` (advanced; check Lightning resume rules):")
    info_lines.append("```bash")
    info_lines.append(f"python -m yolo_jdt.scripts.train -cn base \\")
    info_lines.append(f"    +trainer.ckpt_path={run_dir}/last.ckpt \\")
    info_lines.append(f"    run_name={run_dir.name}_resume")
    info_lines.append("```")

    info_md = "\n".join(info_lines) + "\n"
    (run_dir / "RUN_INFO.md").write_text(info_md)
    print(f"[package_run] wrote RUN_INFO.md ({len(info_md)} bytes)")

    print(f"[package_run] DONE — see {run_dir}/RUN_INFO.md")


if __name__ == "__main__":
    main()
