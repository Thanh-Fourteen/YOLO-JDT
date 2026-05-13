"""Step 3.BCD baseline orchestrator: 2 trackers × 3 datasets.

For each cell (tracker, dataset, split):
    1. run inference (via infer_tracking) → tracker_outputs/<seq>.txt
    2. run TrackEval                       → trackeval_aggregate.json + per_seq.json
    3. package the run dir                 → RUN_INFO.md + tracker_config.yaml

After all cells, aggregate the 6 trackeval_aggregate.json files into a single
`runs/baselines/results.json` for the paper main table.

Skip-if-exists: if a cell's `trackeval_aggregate.json` is already present,
the cell is skipped (use --force to rerun).

Usage:
    python -m yolo_jdt.scripts.run_baselines \\
        --weights weights/ours/yolo11s_det.pt --scale s \\
        [--trackers bytetrack botsort] \\
        [--datasets mot17:val_half mot20:val_half dancetrack:val]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def cell_run_dir(tracker: str, dataset: str, split: str) -> Path:
    short = "mot17" if dataset == "mot17" else dataset
    return PROJECT_ROOT / "runs" / "baselines" / f"step3bcd_yolo11s_{tracker}_{short}"


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
    if rc != 0:
        print(f"[run_baselines] command exited rc={rc}")
    return rc


def run_cell(weights: Path, scale: str, tracker: str, dataset: str, split: str,
             conf: float, gt_cache: Path, force: bool) -> dict | None:
    """Returns the aggregate metrics dict for this cell (or None on failure)."""
    run_dir = cell_run_dir(tracker, dataset, split)
    tracker_out_dir = run_dir / "tracker_outputs"
    agg_path = run_dir / "trackeval_aggregate.json"

    if agg_path.exists() and not force:
        print(f"[run_baselines] SKIP {tracker} on {dataset}/{split} "
              f"(already done — use --force to rerun)")
        return json.loads(agg_path.read_text())

    print(f"\n{'=' * 70}")
    print(f"[run_baselines] cell: tracker={tracker} dataset={dataset}/{split}")
    print(f"{'=' * 70}")
    t0 = time.perf_counter()

    # 1. Inference
    rc = _run([
        sys.executable, "-m", "yolo_jdt.scripts.infer_tracking",
        "--weights", str(weights), "--scale", scale,
        "--dataset", dataset, "--split", split,
        "--tracker", tracker,
        "--output-dir", str(tracker_out_dir),
        "--conf", str(conf),
    ])
    if rc != 0:
        print(f"[run_baselines] inference FAILED for {tracker}/{dataset}")
        return None

    # 2. TrackEval
    rc = _run([
        sys.executable, "-m", "yolo_jdt.eval.trackeval_runner",
        "--tracker-outputs", str(tracker_out_dir),
        "--gt-cache", str(gt_cache),
        "--dataset", dataset, "--split", split,
        "--tracker-name", f"yolo11s_{tracker}",
        "--out-dir", str(run_dir),
    ])
    if rc != 0:
        print(f"[run_baselines] trackeval FAILED for {tracker}/{dataset}")
        return None

    # 3. Save tracker config snapshot for reproducibility
    if tracker == "bytetrack":
        from yolo_jdt.tracker.bytetrack import ByteTrackConfig
        cfg_dict = vars(ByteTrackConfig())
    else:
        from yolo_jdt.tracker.botsort import BoTSORTConfig
        cfg_dict = vars(BoTSORTConfig())
    cfg_dict["nms_conf_thresh"] = conf
    (run_dir / "tracker_config.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in sorted(cfg_dict.items())) + "\n"
    )

    # 4. RUN_INFO.md (simpler than package_run.py — there's no Lightning ckpt
    #    or wandb dir for tracker runs)
    elapsed = time.perf_counter() - t0
    agg = json.loads(agg_path.read_text())
    info_lines = [
        f"# Run: `{run_dir.name}`",
        "",
        f"_Packaged: {datetime.now():%Y-%m-%d %H:%M:%S}_  ",
        f"_Phase: 3.BCD_  ",
        "",
        "## Cell",
        f"- **Tracker:** `{tracker}`",
        f"- **Dataset:** `{dataset}` / `{split}`",
        f"- **Detector:** `{weights}` (scale=`{scale}`)",
        f"- **NMS conf threshold:** `{conf}`",
        f"- **Wall clock:** {elapsed:.1f} s",
        "",
        "## Aggregate metrics (TrackEval, mean over sequences)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in agg.items():
        if isinstance(v, float):
            info_lines.append(f"| {k} | {v:.4f} |")
        else:
            info_lines.append(f"| {k} | {v} |")
    info_lines.extend([
        "",
        "## Files",
        "",
        "| File | Purpose |",
        "|---|---|",
        "| `tracker_outputs/*.txt` | per-seq MOT-format predictions |",
        "| `trackeval_aggregate.json` | aggregate metrics (this table) |",
        "| `trackeval_per_seq.json` | per-sequence metrics |",
        "| `fps_breakdown.json` | per-stage timing (load / preproc / forward / nms / postproc / track) |",
        "| `tracker_config.yaml` | tracker hyperparameter snapshot |",
        "| `_trackeval_workspace/` | TrackEval temp working dir (safe to delete) |",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m yolo_jdt.scripts.infer_tracking \\",
        f"    --weights {weights} --scale {scale} \\",
        f"    --dataset {dataset} --split {split} --tracker {tracker} \\",
        f"    --output-dir {tracker_out_dir} --conf {conf}",
        "",
        "python -m yolo_jdt.eval.trackeval_runner \\",
        f"    --tracker-outputs {tracker_out_dir} \\",
        f"    --gt-cache {gt_cache} --dataset {dataset} --split {split} \\",
        f"    --tracker-name yolo11s_{tracker} --out-dir {run_dir}",
        "```",
    ])
    (run_dir / "RUN_INFO.md").write_text("\n".join(info_lines) + "\n")

    print(f"[run_baselines] cell done in {elapsed:.1f}s — see {run_dir}/RUN_INFO.md")
    return agg


def aggregate_results(cells: list[tuple[str, str, str]],
                      detector_meta: dict,
                      out_path: Path) -> None:
    """Build the immutable runs/baselines/results.json aggregating all cells."""
    results = {}
    for tracker, dataset, split in cells:
        agg_path = cell_run_dir(tracker, dataset, split) / "trackeval_aggregate.json"
        if not agg_path.exists():
            print(f"[run_baselines] WARN: missing {agg_path} — skipping in aggregate")
            continue
        agg = json.loads(agg_path.read_text())
        # Average per-stage FPS for the cell
        fps_path = cell_run_dir(tracker, dataset, split) / "fps_breakdown.json"
        overall_fps = None
        if fps_path.exists():
            fps = json.loads(fps_path.read_text())
            fps_vals = [v["overall_fps"] for v in fps.values()
                        if isinstance(v, dict) and "overall_fps" in v]
            if fps_vals:
                overall_fps = sum(fps_vals) / len(fps_vals)

        tracker_label = f"yolo11s_{tracker}"
        # Use a readable benchmark key
        bench_label = {
            ("mot17", "val_half"): "MOT17_val_half",
            ("mot20", "val_half"): "MOT20_val_half",
            ("dancetrack", "val"): "DanceTrack_val",
        }.get((dataset, split), f"{dataset}_{split}")
        cell_block = dict(agg)
        if overall_fps is not None:
            cell_block["FPS_overall"] = overall_fps
        results.setdefault(tracker_label, {})[bench_label] = cell_block

    # Meta — try to capture git commit (read-only call, ok per project rules)
    git_sha = None
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    results["_meta"] = {
        **detector_meta,
        "trackeval_version": "1.3.0",
        "git_commit": git_sha,
        "date": datetime.now().isoformat(timespec="seconds"),
        "headline_metrics_explained": {
            "HOTA": "Higher Order Tracking Accuracy (Luiten et al. 2021); paper-standard.",
            "DetA": "Detection accuracy component of HOTA.",
            "AssA": "Association accuracy component of HOTA.",
            "MOTA": "Multi-Object Tracking Accuracy (Bernardin/Stiefelhagen 2008).",
            "IDF1": "ID F1 score (Ristani et al. 2016).",
            "IDs":  "ID switches (CLEAR-MOT).",
            "FP":   "False positives.",
            "FN":   "False negatives.",
            "FPS_overall": "Mean per-sequence end-to-end FPS (load+preproc+forward+nms+postproc+track).",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[run_baselines] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--scale", required=True, choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--trackers", nargs="+",
                    default=["bytetrack", "botsort"],
                    choices=["bytetrack", "botsort"])
    ap.add_argument("--datasets", nargs="+",
                    default=["mot17:val_half", "mot20:val_half", "dancetrack:val"],
                    help="dataset:split pairs")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--gt-cache", type=Path,
                    default=Path("runs/baselines/_gt_mot"))
    ap.add_argument("--out-results-json", type=Path,
                    default=Path("runs/baselines/results.json"))
    ap.add_argument("--force", action="store_true",
                    help="rerun cells even if results already exist")
    args = ap.parse_args()

    # Parse dataset:split pairs
    cells = []
    for tracker in args.trackers:
        for ds_str in args.datasets:
            try:
                ds, sp = ds_str.split(":")
            except ValueError:
                sys.exit(f"--datasets entries must be 'dataset:split', got {ds_str!r}")
            cells.append((tracker, ds, sp))

    print(f"[run_baselines] running {len(cells)} cells:")
    for c in cells:
        print(f"  - tracker={c[0]} dataset={c[1]}/{c[2]}")
    print()

    # Cache GT for any datasets we need
    from yolo_jdt.eval.mot_format import cache_gt_dataset
    seen_gt = set()
    for tracker, dataset, split in cells:
        key = (dataset, split)
        if key in seen_gt:
            continue
        seen_gt.add(key)
        bench_token = f"{dataset}_{split}"
        if not (args.gt_cache / bench_token).is_dir() or args.force:
            print(f"[run_baselines] caching GT for {dataset}/{split}...")
            counts = cache_gt_dataset(
                Path("datasets/standard"), dataset, split, args.gt_cache)
            print(f"  -> {len(counts)} seqs, {sum(counts.values())} GT rows")

    # Run cells
    for tracker, dataset, split in cells:
        run_cell(args.weights, args.scale, tracker, dataset, split,
                 args.conf, args.gt_cache, args.force)

    # Aggregate
    weights_payload = None
    try:
        import torch
        weights_payload = torch.load(args.weights, map_location="cpu", weights_only=False)
    except Exception:
        pass
    detector_meta = {
        "detector_ckpt": str(args.weights),
        "detector_scale": args.scale,
    }
    if isinstance(weights_payload, dict):
        if "val_metrics" in weights_payload and weights_payload["val_metrics"]:
            detector_meta["detector_val_metric"] = weights_payload["val_metrics"]
        detector_meta["detector_source_ckpt"] = weights_payload.get("source_ckpt")
        detector_meta["detector_source_weights"] = weights_payload.get("source_weights")

    aggregate_results(cells, detector_meta, args.out_results_json)


if __name__ == "__main__":
    main()
