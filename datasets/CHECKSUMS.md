# Dataset Checksums

Track download sources + SHA256 cho integrity check. **Mỗi lần re-download dataset, so sánh SHA256 với row tương ứng — bất kỳ mismatch nghĩa là dataset đã đổi version hoặc bị corrupt.**

## Sources

| Dataset | Primary source | Mirror used | Why mirror |
|---------|----------------|-------------|------------|
| MOT17 | https://motchallenge.net/data/MOT17.zip | https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip (PaddleDetection) | motchallenge.net không truy cập được từ máy này (TCP timeout) |
| MOT20 | https://motchallenge.net/data/MOT20.zip | https://dataset.bj.bcebos.com/mot/MOT20.zip (Baidu BCE) | motchallenge.net không truy cập được |
| DanceTrack | https://github.com/DanceTrack/DanceTrack | https://huggingface.co/datasets/noahcao/dancetrack (5 zips, 17.7 GB total) | Official Google Drive deprecated; HF là recommended source. Tải từ xethub.hf.co CDN (302 từ HF). |
| CrowdHuman | https://www.crowdhuman.org/download.html | https://huggingface.co/datasets/sshao0516/CrowdHuman (4 zip + 2 odgt, 10.96 GB total) | Official chỉ Google Drive + Baidu Pan; HF mirror scriptable, files identical schema. Skipped `CrowdHuman_test.zip` (3.26 GB) — không cần (val đủ, test ID labels không public). |
| COCO val2017 | http://images.cocodataset.org/zips/val2017.zip + http://images.cocodataset.org/annotations/annotations_trainval2017.zip | direct (CC BY 4.0) | Phase 2 sanity check chỉ cần val split + `instances_val2017.json`. Skip train2017/test2017. |

## Checksums (SHA256)

| File | Size | SHA256 | Downloaded at | Mirror |
|------|------|--------|----------------|--------|
| `MOT17.zip` | 2.22 GB (2,388,186,946 B) | `4253cf596550847a74f58859fee6a1263a03c5bd946ec9545c0119e8e5e5e800` | 2026-05-11 | Baidu BCE (PaddleDetection) |
| `MOT20.zip` | 4.68 GB (5,028,926,248 B) | `ebcf0e3d44e4f50b5357d24817e5db485d777633d1b8ca9e8380d1c8437dbdd7` | 2026-05-11 | Baidu BCE (dataset.bj.bcebos.com) |
| `train1.zip` (DanceTrack) | 3.61 GB (3,606,300,312 B) | `70a66f10d8d71df94d03059fc4f966cb719e61e8e7136d2886fd6345ed1ff6dd` | 2026-05-11 | HF noahcao/dancetrack |
| `train2.zip` (DanceTrack) | 3.30 GB (3,299,320,948 B) | `f676e8c6ac2a2d7566f1702d588311eafd4c90cc5416bb70f36a192d1e365af4` | 2026-05-11 | HF noahcao/dancetrack |
| `val.zip` (DanceTrack) | 4.21 GB (4,209,785,614 B) | `90ba30973761ce0b81a9654c11086d87537392475ac8bc666d842e645641277c` | 2026-05-11 | HF noahcao/dancetrack |
| `test1.zip` (DanceTrack) | 3.68 GB (3,677,760,950 B) | `10e71d2b1c81fb954d7076863d299944b3b15951c4a4b033a454a94569a34a71` | 2026-05-11 (re-DL) | HF noahcao/dancetrack |
| `test2.zip` (DanceTrack) | 2.88 GB (2,882,754,122 B) | `f1d5eb5ffac33ee0c2f184264cb625cd05efe246afcd898cb28db1eb4e3c9167` | 2026-05-11 | HF noahcao/dancetrack |
| `annotation_val.odgt` (CrowdHuman) | 22.2 MB (23,323,139 B) | `be422c79a190ff7e30fe5cbd74cbf45a2dadda6c5af58c6ec11a038ba2993c04` | 2026-05-12 | HF sshao0516/CrowdHuman |
| `annotation_train.odgt` (CrowdHuman) | 76.3 MB (80,017,502 B) | `6bf241a79f19e30cf52681eab3392368bd5a534164be9272e7a808cb284d9f77` | 2026-05-12 | HF sshao0516/CrowdHuman |
| `CrowdHuman_val.zip` | 2.32 GB (2,488,658,160 B) | `c0ab99bb80ac162cd3efdf94a1a6100c4f059a61d69596412c6b44ebc20d1363` | 2026-05-12 | HF sshao0516/CrowdHuman |
| `CrowdHuman_train01.zip` | 2.77 GB (2,970,597,373 B) | `7ba340163cff0f2446027af95dc96bcb9c66be18506eabb57822c400a7efd3b8` | 2026-05-12 | HF sshao0516/CrowdHuman |
| `CrowdHuman_train02.zip` | 2.88 GB (3,092,749,718 B) | `d9ecfb43eaf8381ddd4d1ff4e9a0877b694dbfba7a7a33ed3966ee2c2a628663` | 2026-05-12 | HF sshao0516/CrowdHuman |
| `CrowdHuman_train03.zip` | 2.15 GB (2,306,357,030 B) | `9cac171914c4f5c7371e3d42d20b50858caa1602811ae573aa35bbfe672dc29f` | 2026-05-12 | HF sshao0516/CrowdHuman |
| `val2017.zip` (COCO) | 778 MB (815,585,330 B) | `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05` | 2026-05-12 | images.cocodataset.org |
| `annotations_trainval2017.zip` (COCO) | 241 MB (252,907,541 B) | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` | 2026-05-12 | images.cocodataset.org (only `instances_val2017.json` extracted) |

## MOT17 layout notes

PaddleDetection mirror layout khác standard MOT17.zip (motchallenge.net) — đây là điểm cần lưu ý cho Step 1.E converter:

```
datasets/raw/mot17/MOT17/
├── annotations/
│   ├── train.json          # COCO format + MOT extensions (track_id, prev_image_id, next_image_id)
│   ├── train_half.json     # First half of each train seq (ByteTrack convention)
│   └── val_half.json       # Second half — ready for validation, không phải tự split
├── images/
│   ├── train/MOT17-{02,04,05,09,10,11,13}-SDP/    # 7 train seq (SDP variant only)
│   │   ├── img1/<6-digit>.jpg
│   │   ├── gt/gt.txt        # Standard MOT format
│   │   ├── det/det.txt      # SDP detector predictions
│   │   └── seqinfo.ini
│   ├── test/MOT17-{01,03,06,07,08,12,14}-SDP/     # 7 test seq, no gt/
│   └── half/MOT17-XX-SDP-half/                    # Pre-split val halves
└── labels_with_ids/train/MOT17-XX-SDP/img1/<frame>.txt   # YOLO-format JDE labels
```

**Khác standard MOT17:**
- Standard MOT17.zip có **3 detector variants** per sequence: DPM, FRCNN, SDP (chia sẻ images, khác `det/`). PaddleDetection mirror chỉ có **SDP variant**.
- Standard layout: `MOT17/{train,test}/<seq>/...`. PaddleDetection: `MOT17/images/{train,test}/<seq>/...`.

**Tác động cho project:**
- ✅ SDP-only OK vì pipeline dùng YOLO detector riêng (Phase 2-3), không phụ thuộc pre-computed `det/`.
- ✅ Pre-split `half/` + `train_half.json` / `val_half.json` — không cần tự split ở Step 1.E.
- ✅ COCO JSON với `track_id` + `prev_image_id` / `next_image_id` — bonus cho Phase 5 paired-frame loader.
- ⚠ Cho test submission Step 11.A: chỉ submit SDP variant; document trong paper là single-variant submission.

**Frame counts (train + test):**
- Train: 02=600, 04=1050, 05=837, 09=525, 10=654, 11=900, 13=750 → tổng 5316 frames.
- Test: 01=450, 03=1500, 06=1194, 07=500, 08=625, 12=900, 14=750 → tổng 5919 frames.

## MOT20 layout notes

MOT20.zip từ Baidu BCE = STANDARD MOT layout (khác MOT17 PaddleDetection wrapped):

```
datasets/raw/mot20/MOT20/
├── train/MOT20-{01,02,03,05}/
│   ├── img1/<6-digit>.jpg
│   ├── gt/gt.txt
│   ├── det/det.txt
│   └── seqinfo.ini
└── test/MOT20-{04,06,07,08}/
    ├── img1/...
    ├── det/...
    └── seqinfo.ini  (no gt — test set private)
```

**Khác MOT17:** không có pre-computed COCO JSON annotations, không có pre-split half-train/val, không có labels_with_ids. Step 1.E converter sẽ tự sinh từ gt.txt.

**Frame counts:**
- Train: 01=429, 02=2782, 03=2405, 05=3315 → tổng 8931 frames (rất dense, 246 ped/frame avg).
- Test: 04=2080, 06=1008, 07=585, 08=806 → tổng 4479 frames.

Extracted size: 4.8 GB.

## DanceTrack layout notes

HuggingFace mirror `noahcao/dancetrack` chia thành 5 zip:
- `train1.zip` + `train2.zip` → mỗi cái 20 seq, merge → `train/` (40 total)
- `val.zip` → 25 seq, đã sẵn standalone
- `test1.zip` + `test2.zip` → 20 + 15 = 35 seq, merge → `test/`

**Sau merge:**
```
datasets/raw/dancetrack/
├── train/dancetrack{0001..0099}/     # 40 seq, có gt
├── val/dancetrack0xxx/                # 25 seq, có gt
└── test/dancetrack0xxx/                # 35 seq, gt private
    └── <seq>/{img1/<6-digit>.jpg, gt/gt.txt, seqinfo.ini}
```

**Lưu ý:** zip có `.DS_Store` macOS metadata trong train1/train2/test1/test2 — đã xóa khi merge.

**Frame counts:**
- Train: min=403 max=2163, total 40 seq
- Val: min=183 max=2402, total 25 seq
- Test: min=403 max=1601, total 35 seq

**Truncate accident**: lần download đầu, `test1.zip` chỉ về 2.23 GB (vs expected 3.68 GB) — có thể network drop hoặc HF rate limit khi 5 parallel. Re-DL với `--retry 3 --retry-delay 5 -C -` đã fix. SHA256 đã ghi version đúng. Bài học: với multi-part zip qua HF, set retry hoặc download tuần tự.

Extracted size: ~31 GB (sau merge, vẫn giữ 5 zip ~17.7 GB cho re-extract nếu corrupt).

## CrowdHuman layout notes

HF mirror `sshao0516/CrowdHuman` chia thành 4 zip + 2 odgt. Mỗi zip có prefix `Images/` (extract junk-paths để flatten):

```
datasets/raw/crowdhuman/
├── images/
│   ├── train/<id>,<hash>.jpg     # 15,000 jpg (từ train01+02+03 merged, mỗi zip 5000)
│   └── val/<id>,<hash>.jpg        # 4,370 jpg
├── annotations/
│   ├── annotation_train.odgt      # 15,000 records
│   └── annotation_val.odgt        # 4,370 records
└── CrowdHuman_*.zip               # giữ 4 zip (10.96 GB) cho re-extract nếu corrupt
```

**odgt format:** one JSON dict per line. Schema:
```json
{"ID": "284193,faa9000f2678b5e",
 "gtboxes": [
   {"tag": "person",
    "fbox": [x, y, w, h],          // full body box
    "hbox": [x, y, w, h],          // head box
    "vbox": [x, y, w, h],          // visible body box
    "head_attr": {...},
    "extra": {"box_id": ..., "occ": 0/1, ...}}
 ]}
```
- `tag`: `person` (positive) hoặc `mask` (ignore region — không train detection trên).
- `fbox` = full body (kể cả phần bị che). `vbox` = visible only. JDT detection thường train trên `fbox` để model học predict cả phần occluded.
- `extra.occ` = occlusion flag, dùng để filter heavy occlusion sample khi train ReID.

**Cross-check (Step 1.D verified):** train 15000 anno IDs == 15000 img IDs (0 missing); val 4370 == 4370 (0 missing). Average 22.6 GT boxes per image — đặc biệt dense, lý tưởng cho pre-train detection trước MOT fine-tune.

**Note ID prefix collision:** train03 và val có thể chia sẻ prefix range `284193,...` (khác hash phần sau). Vì vậy extract train/val vào dir riêng, không dùng chung `images/` flat.

**License:** Non-commercial research/educational only. KHÔNG redistribute. README + future MODEL_CARD đã ghi rõ; mọi production weight train trên CrowdHuman cần re-train không có CrowdHuman hoặc xin commercial license trực tiếp từ tác giả.

Extracted size: 11 GB images + 99 MB annotations = 11.1 GB. Tổng raw bao gồm zips: ~22 GB.
