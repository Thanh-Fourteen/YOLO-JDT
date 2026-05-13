"""Thin wrapper around the TrackEval library for our run-dir layout.

TrackEval expects a strict folder hierarchy:
    GT_FOLDER/<BENCHMARK>-<SPLIT>/<seq>/gt/gt.txt
    GT_FOLDER/<BENCHMARK>-<SPLIT>/<seq>/seqinfo.ini
    TRACKERS_FOLDER/<BENCHMARK>-<SPLIT>/<tracker_name>/data/<seq>.txt

Our `cache_gt_dataset` already builds the GT layout under
`runs/baselines/_gt_mot/<dataset>_<split>/`. This wrapper:
1. Symlinks tracker output into the TrackEval-expected `data/` subfolder.
2. Configures `MotChallenge2DBox` with our paths + a custom seqmap.
3. Runs HOTA + CLEAR + Identity metrics.
4. Returns aggregated + per-seq results as plain Python dicts.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from trackeval import Evaluator
from trackeval.datasets import MotChallenge2DBox
from trackeval.metrics import HOTA, CLEAR, Identity


# Metric keys we expose externally — TrackEval emits many but we keep
# the headline numbers for results.json + RUN_INFO.md.
HEADLINE_KEYS = {
    "HOTA":    ("HOTA", "HOTA"),       # (TrackEval metric class, sub-key) → produced as float
    "DetA":    ("HOTA", "DetA"),
    "AssA":    ("HOTA", "AssA"),
    "MOTA":    ("CLEAR", "MOTA"),
    "MOTP":    ("CLEAR", "MOTP"),
    "IDF1":    ("Identity", "IDF1"),
    "IDP":     ("Identity", "IDP"),
    "IDR":     ("Identity", "IDR"),
    "FP":      ("CLEAR", "CLR_FP"),
    "FN":      ("CLEAR", "CLR_FN"),
    "IDs":     ("CLEAR", "IDSW"),
}


def evaluate(tracker_outputs_dir: Path, gt_cache_root: Path,
             dataset_name: str, split: str, tracker_name: str
             ) -> tuple[dict, dict[str, dict]]:
    """Run TrackEval over the given tracker outputs.

    Args:
        tracker_outputs_dir: dir containing `<seq>.txt` files written by
            `infer_tracking.py`.
        gt_cache_root: dir from `cache_gt_dataset(...)`.
        dataset_name: e.g. "mot17".
        split: e.g. "val_half".
        tracker_name: arbitrary label, becomes a folder name.

    Returns:
        (aggregate_metrics, per_seq_metrics) — both flat dicts of the keys
        listed in `HEADLINE_KEYS`.
    """
    # ---- Build TrackEval-expected layout ----
    bench_token = f"{dataset_name}_{split}"          # e.g. mot17_val_half
    gt_bench_dir = gt_cache_root / bench_token
    if not gt_bench_dir.is_dir():
        raise FileNotFoundError(f"GT cache missing: {gt_bench_dir}")
    gt_seqs = sorted(p.name for p in gt_bench_dir.iterdir() if p.is_dir())
    # Only evaluate seqs for which we have tracker output (intersection)
    pred_seqs = sorted(p.stem for p in tracker_outputs_dir.glob("*.txt"))
    seqs = [s for s in gt_seqs if s in set(pred_seqs)]
    missing = sorted(set(gt_seqs) - set(pred_seqs))
    if missing:
        print(f"[trackeval_runner] skipping {len(missing)} seqs without tracker "
              f"output: {missing}")
    if not seqs:
        raise RuntimeError("No overlap between GT cache and tracker outputs")

    trackers_root = tracker_outputs_dir.parent / "_trackeval_workspace"
    trackers_root.mkdir(exist_ok=True)
    tracker_data_dir = trackers_root / bench_token / tracker_name / "data"
    if tracker_data_dir.exists():
        shutil.rmtree(tracker_data_dir)
    tracker_data_dir.mkdir(parents=True)
    for seq in seqs:
        src = tracker_outputs_dir / f"{seq}.txt"
        dst = tracker_data_dir / f"{seq}.txt"
        dst.symlink_to(os.path.relpath(src, dst.parent))

    # Custom seqmap file — TrackEval reads to know which seqs to evaluate
    seqmap_dir = trackers_root / bench_token / "_seqmaps"
    seqmap_dir.mkdir(exist_ok=True)
    seqmap_file = seqmap_dir / f"{bench_token}.txt"
    seqmap_file.write_text("name\n" + "\n".join(seqs) + "\n")

    # ---- Configure TrackEval ----
    eval_cfg = Evaluator.get_default_eval_config()
    eval_cfg.update({
        "USE_PARALLEL": False,
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": False,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_EMPTY_CLASSES": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
        "DISPLAY_LESS_PROGRESS": True,
        "TIME_PROGRESS": False,
    })

    # With SKIP_SPLIT_FOL=True, GT_FOLDER is treated as the dir directly
    # containing per-seq folders. So pass `<gt_cache_root>/<bench_token>` as
    # GT_FOLDER and `<trackers_root>/<bench_token>` as TRACKERS_FOLDER.
    ds_cfg = MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({
        "GT_FOLDER": str(gt_bench_dir),
        "TRACKERS_FOLDER": str(trackers_root / bench_token),
        "OUTPUT_FOLDER": str(trackers_root / "output"),
        "TRACKERS_TO_EVAL": [tracker_name],
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": dataset_name,
        "SPLIT_TO_EVAL": split,
        "INPUT_AS_ZIP": False,
        "DO_PREPROC": dataset_name in ("mot17", "mot20"),
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "SEQMAP_FILE": str(seqmap_file),
        "SEQ_INFO": None,
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "SKIP_SPLIT_FOL": True,
        "PRINT_CONFIG": False,
    })

    evaluator = Evaluator(eval_cfg)
    dataset_list = [MotChallenge2DBox(ds_cfg)]
    metrics_list = [HOTA(), CLEAR(), Identity()]

    raw_results, _ = evaluator.evaluate(dataset_list, metrics_list)

    # raw_results structure (TrackEval 1.3):
    #   {dataset_class_name: {tracker_name: {seq_name | "COMBINED_SEQ": {class: {metric: {key: val}}}}}}
    ds_key = "MotChallenge2DBox"
    tracker_results = raw_results[ds_key][tracker_name]

    def _pull(seq_block: dict) -> dict:
        cls_block = seq_block.get("pedestrian", {})
        out = {}
        for headline_name, (metric_name, sub_key) in HEADLINE_KEYS.items():
            arr = cls_block.get(metric_name, {}).get(sub_key)
            if arr is None:
                out[headline_name] = None
                continue
            try:
                # HOTA returns arrays (one per IoU); take mean
                if hasattr(arr, "mean"):
                    out[headline_name] = float(arr.mean())
                else:
                    out[headline_name] = float(arr)
            except (TypeError, ValueError):
                out[headline_name] = None
        return out

    per_seq: dict[str, dict] = {}
    for k, v in tracker_results.items():
        if k == "COMBINED_SEQ":
            continue
        per_seq[k] = _pull(v)

    aggregate = _pull(tracker_results["COMBINED_SEQ"])
    return aggregate, per_seq


def write_results_json(aggregate: dict, per_seq: dict[str, dict],
                       out_dir: Path) -> None:
    """Write trackeval_aggregate.json + trackeval_per_seq.json side-by-side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trackeval_aggregate.json").write_text(
        json.dumps(aggregate, indent=2))
    (out_dir / "trackeval_per_seq.json").write_text(
        json.dumps(per_seq, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker-outputs", required=True, type=Path)
    ap.add_argument("--gt-cache", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tracker-name", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    agg, per_seq = evaluate(args.tracker_outputs, args.gt_cache,
                             args.dataset, args.split, args.tracker_name)
    write_results_json(agg, per_seq, args.out_dir)
    print("\nAGGREGATE:")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
