# Dataset Splits — YOLO-JDT

Standard splits for `datasets/standard/<name>/` after running converters
(Step 1.EF). All annotations and images point at these split names.

## Schema

```
datasets/standard/<name>/
├── images/<split>/<seq>/<frame_id>.jpg     # symlinks to raw
└── annotations/<split>/<seq>.json
```

Per-seq JSON:

```json
{
  "name": "MOT17-02-SDP",
  "frame_rate": 30,                       // 0 for static datasets
  "image_size": [W, H],
  "num_frames": N,
  "frames": [
    {
      "frame_id": 1,                      // 1-indexed (MOT convention)
      "image": "MOT17-02-SDP/000001.jpg", // relative to images/<split>/
      "objects": [
        {"track_id": 1, "class_id": 0,
         "bbox_xywh": [x_tl, y_tl, w, h], // PIXELS (top-left + size)
         "visibility": 1.0,
         "iscrowd": 0}
      ]
    }
  ]
}
```

Bboxes stored in **pixels** (top-left x/y + width/height). Dataset
classes normalize to YOLO `(cx, cy, w, h) ∈ [0, 1]` on `__getitem__`.

`class_id == 0` is the only class in this project (pedestrian / person);
multi-class will be added later if scope expands.

## ByteTrack half-split convention

For MOT17/MOT20 ablation work and method development, we follow the
ByteTrack convention: split each train sequence in half; first half
becomes `train_half`, second half becomes `val_half`. This lets us
evaluate locally without holding back any test-set submission budget,
and matches numbers reported in ByteTrack, BoT-SORT, FairMOT.

Specifically: for a sequence with `N` frames (1-indexed),

- `train_half` = frames `1 .. N // 2`
- `val_half`   = frames `N // 2 + 1 .. N`

For MOT17 the raw mirror already ships pre-split `images/half/<seq>-half/`
(second-half images only). We use this directly for `val_half`, and
derive `train_half` as the complement. For MOT20 we compute both halves
ourselves from `seqinfo.ini::seqLength`.

## Splits per dataset

### MOT17 (`datasets/standard/mot17/`)

| Split        | Source                                         | # Seq | # Frames |
|--------------|------------------------------------------------|-------|----------|
| `train`      | `MOT17/images/train/<seq>` (full sequences)    | 7     | 5,316    |
| `train_half` | first half of each train sequence              | 7     | 2,664    |
| `val_half`   | second half of each train sequence (pre-split) | 7     | 2,652    |
| `test`       | `MOT17/images/test/<seq>` (no GT)              | 7     | 5,919    |

GT filter: `mark == 1` AND `class == 1` (pedestrian). All other classes
in MOT17 GT (static person, distractor, occluder, vehicles) are dropped.

### MOT20 (`datasets/standard/mot20/`)

| Split        | Source                            | # Seq | # Frames (approx) |
|--------------|-----------------------------------|-------|-------------------|
| `train`      | `MOT20/train/<seq>` (full)        | 4     | 8,931             |
| `train_half` | derived: frames `1..N//2`         | 4     | 4,463             |
| `val_half`   | derived: frames `N//2+1..N`       | 4     | 4,468             |
| `test`       | `MOT20/test/<seq>` (no GT)        | 4     | 4,479             |

GT filter: same as MOT17 (`mark == 1`, `class == 1`).

### DanceTrack (`datasets/standard/dancetrack/`)

| Split   | Source                                  | # Seq | # Frames (approx) |
|---------|-----------------------------------------|-------|-------------------|
| `train` | `dancetrack/train/dancetrackXXXX`       | 40    | ~40,000           |
| `val`   | `dancetrack/val/dancetrackXXXX`         | 25    | ~25,000           |
| `test`  | `dancetrack/test/dancetrackXXXX` no GT  | 35    | ~35,000           |

DanceTrack uses native `train/val/test` splits — no half-split needed,
training set is large enough.

GT filter: keep all (`class == 1` already; `mark == 1` always).
Track IDs in DanceTrack GT start at `0`.

### CrowdHuman (`datasets/standard/crowdhuman/`)

CrowdHuman is a static-image dataset, not a video. Each split is encoded
as a single "sequence" of unrelated images for schema uniformity.

| Split   | Source                                          | "# Seq" | # Frames |
|---------|-------------------------------------------------|---------|----------|
| `train` | `images/train/*.jpg` + `annotation_train.odgt`  | 1       | 15,000   |
| `val`   | `images/val/*.jpg` + `annotation_val.odgt`      | 1       | 4,370    |

All boxes have `track_id = -1` (static; no temporal continuity). Pairs
loader (Phase 5) MUST skip CrowdHuman or yield `(img, img)` self-pairs.

GT filter: `tag == 'person'` (drop `mask` ignore regions). `bbox_xywh`
uses CrowdHuman's `fbox` (full body, including occluded parts). `hbox`
and `vbox` are not exported — add later if head detection is needed.

## Why not symlink the raw splits as-is

Each raw dataset has its own quirks: MOT17 wraps in `images/{train,test}/`,
MOT20 uses standard layout, DanceTrack uses zero-padded 8-digit frames,
CrowdHuman is single-image with `<id>,<hash>.jpg` filenames. Standard
format normalizes naming, splits, and annotation schema so dataloaders
and converters in Phases 3-7 see one schema across the project.

Symlinking (rather than copying) image files keeps disk usage flat —
standard format adds < 5 MB of JSON on top of ~70 GB raw datasets.
