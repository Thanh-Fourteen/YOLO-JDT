# YOLO-JDT: Joint Detection-Tracking với Temporal Feature Recycling

## Tài liệu Nghiên cứu Tổng hợp — Research & Project Overview

**Tác giả:** [Your Name]
**Ngày khởi tạo:** 09/05/2026
**Phiên bản:** 1.0
**Lĩnh vực:** AI Computer Vision — Multi-Object Tracking (MOT)

---

## Mục lục

1. [Tổng quan vấn đề nghiên cứu](#1-tổng-quan-vấn-đề-nghiên-cứu)
2. [Khảo sát các Paradigm trong MOT](#2-khảo-sát-các-paradigm-trong-mot)
3. [Phân tích chi tiết các Model hiện có](#3-phân-tích-chi-tiết-các-model-hiện-có)
4. [Bảng Benchmark tổng hợp](#4-bảng-benchmark-tổng-hợp)
5. [Phân tích YOLO Versions cho JDT](#5-phân-tích-yolo-versions-cho-jdt)
6. [Các hướng tiếp cận Novel đã khảo sát](#6-các-hướng-tiếp-cận-novel-đã-khảo-sát)
7. [Hướng đi được chọn: YOLO-JDT với TAGate](#7-hướng-đi-được-chọn-yolo-jdt-với-tagate)
8. [Chiến lược thực hiện — Strategy C](#8-chiến-lược-thực-hiện--strategy-c)
9. [Thiết kế kiến trúc đề xuất](#9-thiết-kế-kiến-trúc-đề-xuất)
10. [Kế hoạch thí nghiệm & Đánh giá](#10-kế-hoạch-thí-nghiệm--đánh-giá)
11. [Rủi ro & Giải pháp](#11-rủi-ro--giải-pháp)
12. [Tài liệu tham khảo](#12-tài-liệu-tham-khảo)

---

## 1. Tổng quan vấn đề nghiên cứu

### 1.1 Bối cảnh

Multi-Object Tracking (MOT) là bài toán cốt lõi trong computer vision, yêu cầu đồng thời phát hiện (detect) tất cả đối tượng quan tâm trong mỗi frame video và liên kết (associate/track) chúng qua các frame liên tiếp để tạo thành các quỹ đạo (trajectories). MOT được ứng dụng rộng rãi trong autonomous driving, video surveillance, human-computer interaction, sports analytics và robotics.

Hai thách thức cơ bản của MOT hiện tại:

- **Tốc độ vs. Độ chính xác:** Các phương pháp end-to-end (MOTR, MOTRv2) đạt association quality cao nhưng chạy chậm (7–10 FPS). Các phương pháp tracking-by-detection (ByteTrack, BoT-SORT) nhanh (30+ FPS) nhưng cần chạy detector và tracker riêng biệt, tốn compute gấp đôi khi thêm ReID.
- **Mâu thuẫn gradient khi train joint:** Khi huấn luyện detection và tracking embedding chung backbone, hai task cạnh tranh gradient, dẫn đến detection hoặc ReID embedding bị suy giảm chất lượng. FairMOT giải quyết phần nào bằng anchor-free design, nhưng backbone (DLA-34) đã lỗi thời.

### 1.2 Mục tiêu dự án

Xây dựng một model **Joint Detection-Tracking (JDT)** đáp ứng đồng thời các tiêu chí:

| Tiêu chí | Yêu cầu | Đo bằng |
|-----------|----------|---------|
| Tốc độ | Nhanh hơn pipeline TBD tách rời (detector + separate tracker) | FPS, latency (ms) |
| Độ chính xác | Không giảm so với TBD SOTA (ByteTrack/BoT-SORT) | HOTA, MOTA, IDF1 |
| Model nhẹ | Ít parameters, phù hợp deploy edge/production | Params (M), FLOPs (G) |
| Tính mới (Novelty) | Hướng tiếp cận chưa ai thực hiện hoặc thực hiện chưa tối ưu | Literature gap analysis |

### 1.3 Phạm vi nghiên cứu

- Tập trung vào **2D Multi-Object Tracking** trên video RGB.
- Benchmark chính: MOT17, MOT20, DanceTrack.
- Đối tượng chính: Pedestrian tracking (mở rộng sang general objects nếu thời gian cho phép).
- Không bao gồm: 3D MOT, Multi-camera tracking, Single Object Tracking (SOT).

---

## 2. Khảo sát các Paradigm trong MOT

### 2.1 Tracking-by-Detection (TBD)

**Mô tả:** Chạy object detector trên từng frame độc lập, sau đó dùng thuật toán data association (Kalman Filter, Hungarian Algorithm, IoU matching, ReID matching) để liên kết detection boxes qua các frame.

**Đại diện:** SORT, DeepSORT, ByteTrack, BoT-SORT, OC-SORT, StrongSORT.

**Ưu điểm:**
- Linh hoạt: có thể thay đổi detector hoặc tracker độc lập.
- Heuristic-based association (Kalman + IoU) hoạt động rất tốt trên datasets có chuyển động tuyến tính (MOT17).
- Tốc độ cao khi detector nhanh và tracker không cần ReID (~30 FPS).

**Nhược điểm:**
- Compute không tối ưu: backbone inference chạy 2 lần nếu cần ReID (1 cho detector, 1 cho ReID network).
- Detector và tracker không share gradient — tracker không giúp detector tốt hơn và ngược lại.
- Kalman Filter giả định linear motion — yếu trên datasets có non-linear motion (DanceTrack).

**Benchmark tiêu biểu (MOT17 test, private detection):**

| Method | MOTA | IDF1 | HOTA | IDs↓ | FPS |
|--------|------|------|------|------|-----|
| ByteTrack | 80.3 | 77.3 | 63.1 | 159 | ~30 |
| BoT-SORT | 80.5 | 80.2 | 65.0 | 116 | ~30 |
| BoT-SORT-ReID | 80.5 | 80.2 | 65.0 | 116 | ~20 |
| StrongSORT++ | 79.6 | 79.5 | 64.4 | 143 | ~15 |
| OC-SORT | 78.0 | 77.5 | 63.2 | 942 | ~30 |

### 2.2 Joint Detection and Embedding (JDE)

**Mô tả:** Detection và ReID embedding được tích hợp trong cùng một network, chia sẻ backbone. Network xuất ra đồng thời bounding boxes, class scores, và ReID feature vectors.

**Đại diện:** JDE (2019), FairMOT (2020), CSTrack (2021), RetinaTrack (2020), YOLO11-JDE (WACV 2025).

**Ưu điểm:**
- Chỉ cần 1 lần backbone inference cho cả detection và ReID → tiết kiệm compute.
- Tích hợp chặt hơn TBD — features được chia sẻ tốt hơn giữa detection và ReID.
- Phù hợp real-time deployment.

**Nhược điểm:**
- **Anchor bias (JDE gốc):** Khi dùng anchor-based detector, ReID features bị ảnh hưởng bởi anchor positions, dẫn đến nhiều ID switches. FairMOT giải quyết bằng anchor-free (CenterNet).
- **Detection accuracy thấp hơn TBD:** Backbone chia sẻ cho 2 tasks thường đạt detection accuracy thấp hơn detector chuyên biệt. FairMOT đạt MOTA 73.7 vs ByteTrack 80.3 — gap 6.6 điểm.
- **Không exploit temporal information:** Mỗi frame được xử lý độc lập, không tận dụng thông tin temporal giữa các frames.

**Phát hiện quan trọng — YOLO11-JDE (WACV 2025):**
Paper gần đây nhất trong dòng JDE, tích hợp ReID branch vào YOLO11's decoupled head, dùng Mosaic augmentation như self-supervised ReID signal. Kết quả outperform FairMOT trên cả MOT17 và MOT20. Đây là proof-of-concept rằng modern YOLO architecture có thể support JDE paradigm hiệu quả.

### 2.3 End-to-End Transformer-based (E2E)

**Mô tả:** Dùng Transformer architecture với concept "track queries" — mỗi track query đại diện cho một object trajectory, được truyền và cập nhật qua các frames. Detection và association diễn ra hoàn toàn trong một forward pass, không cần NMS hay Hungarian matching post-processing.

**Đại diện:** MOTR (ECCV 2022), TrackFormer (2022), MOTRv2 (CVPR 2023), TransTrack (2021), CO-MOT (ICLR 2024), MO-YOLO (2024), DecoderTracker (2026).

**Ưu điểm:**
- Association thông qua attention mechanism — mạnh ở non-linear motion, complex scenarios (DanceTrack).
- Không cần hand-crafted heuristics (NMS, Kalman Filter, Hungarian).
- Tiềm năng learn complex association patterns end-to-end.

**Nhược điểm:**
- **Rất chậm:** MOTR ~9.5 FPS, MOTRv2 ~7–10 FPS. MO-YOLO cải thiện lên ~19.6 FPS nhưng vẫn chậm hơn TBD.
- **Model nặng:** Transformer encoder-decoder tốn params và FLOPs đáng kể.
- **Conflict detect vs track:** MOTR gặp vấn đề detection quality kém do gradient interference giữa detect queries và track queries. MOTRv2 giải quyết bằng cách dùng pretrained YOLOX detector bên ngoài — nhưng điều này mâu thuẫn với triết lý "end-to-end".
- **Scarce positive samples cho detection queries** (CO-MOT phân tích): trong enclosed scenes, phần lớn objects xuất hiện ngay đầu video → detection queries thiếu training signal.

**Benchmark tiêu biểu:**

| Method | MOT17 MOTA | DanceTrack HOTA | FPS | Backbone |
|--------|-----------|-----------------|-----|----------|
| MOTR | 73.4 | — | ~9.5 | ResNet-50 + DETR |
| MOTRv2 | 78.6 | 73.4 | ~7–10 | Deformable DETR + YOLOX |
| MO-YOLO | ≈MOTR | ≈MOTR | ~19.6 | YOLOv8 + RT-DETR decoder |
| DecoderTracker | — | — | >MO-YOLO | YOLOv8 + lightweight decoder |

### 2.4 Hybrid — LITE Paradigm

**LITE (Lightweight Integrated Tracking-by-detection with ReID)** là một paradigm mới đáng chú ý: tích hợp ReID feature extraction trực tiếp vào tracking pipeline sử dụng standard CNN-based detector (YOLOv8m). Đạt HOTA 43.03% tại 28.3 FPS trên MOT17, nhanh gấp 2x DeepSORT trên MOT17 và 4x trên MOT20 crowded dataset, trong khi giữ nguyên accuracy. LITE demonstrate rằng ReID integration không nhất thiết phải thay đổi detector architecture mà có thể thực hiện ở pipeline level.

### 2.5 Tổng hợp vấn đề cốt lõi

| Vấn đề | TBD | JDE | E2E |
|--------|-----|-----|-----|
| Compute efficiency | ✗ (2x backbone) | ✓ (1x backbone) | ✗ (heavy decoder) |
| Detection accuracy | ✓ (dedicated detector) | ✗ (shared backbone, conflict) | ✗ (conflict queries) |
| Association quality | ✗ (heuristic-based) | ○ (learned embedding) | ✓ (attention-based) |
| Temporal reasoning | ✗ (Kalman = linear) | ✗ (no temporal info) | ✓ (track queries) |
| Real-time | ✓ (~30 FPS) | ✓ (~30 FPS) | ✗ (10–20 FPS) |
| Model size | ○ (detector + tracker) | ✓ (single model) | ✗ (transformer heavy) |

**Gap xác định:** Chưa có phương pháp nào đồng thời đạt: (1) 1 lần backbone inference, (2) temporal reasoning tốt hơn Kalman Filter, (3) real-time 30+ FPS, (4) model nhẹ.

---

## 3. Phân tích chi tiết các Model hiện có

### 3.1 ByteTrack (ECCV 2022)

- **Kiến trúc:** YOLOX detector + BYTE association (two-stage matching: high-confidence → low-confidence).
- **Đặc điểm nổi bật:** Associate mọi detection box kể cả low-score — recover missed detections do occlusion.
- **Điểm mạnh:** Đơn giản, hiệu quả, không cần ReID → nhanh. BYTE strategy có thể plug vào bất kỳ detector nào.
- **Điểm yếu:** Không có appearance model → yếu long-term re-identification sau occlusion dài. Phụ thuộc hoàn toàn vào detector quality.
- **MOT17:** MOTA 80.3, IDF1 77.3, HOTA 63.1, IDs 159, ~30 FPS.

### 3.2 BoT-SORT / BoT-SORT-ReID (2022)

- **Kiến trúc:** YOLOX detector + cải tiến SORT với Camera Motion Compensation (CMC), Kalman Filter cải tiến, optional ReID (BoT features + SBS-S50 ReID model).
- **Đặc điểm nổi bật:** CMC xử lý tốt camera motion. ReID version thêm appearance matching nhưng tốn thêm latency.
- **Điểm mạnh:** SOTA IDF1/MOTA/HOTA trên MOT17. CMC giúp robust trong camera chuyển động.
- **Điểm yếu:** ReID cần model riêng → thêm ~10ms latency. Nhiều hyperparameter cần tune (IoU threshold, ReID weight, CMC params).
- **MOT17:** MOTA 80.5, IDF1 80.2, HOTA 65.0, IDs 116, ~30 FPS (without ReID), ~20 FPS (with ReID).

### 3.3 FairMOT (IJCV 2021)

- **Kiến trúc:** CenterNet (DLA-34 backbone, anchor-free) + 2 homogeneous branches (detection + ReID).
- **Đặc điểm nổi bật:** Anchor-free design giải quyết "anchor bias" mà JDE gốc gặp phải — ReID features được extract tại object center thay vì anchor positions.
- **Điểm mạnh:** 1 backbone cho cả detect + ReID, real-time, balance speed/accuracy tốt cho thời điểm đó.
- **Điểm yếu:** DLA-34 backbone đã cũ so với YOLOX/YOLOv8. Detection accuracy thấp hơn SOTA ~6 MOTA. Cần nhiều data có ID annotation để train.
- **MOT17:** MOTA 73.7, IDF1 72.3, HOTA 59.3, IDs 3303, ~30 FPS.

### 3.4 YOLO11-JDE (WACV 2025)

- **Kiến trúc:** YOLO11s backbone + ReID branch tích hợp vào decoupled head + self-supervised ReID qua Mosaic augmentation.
- **Đặc điểm nổi bật:** Không cần ID annotation cho ReID training — dùng Mosaic data augmentation tạo ra cùng identity dưới nhiều transformations.
- **Điểm mạnh:** YOLO11 backbone hiện đại. Self-supervised ReID mở rộng được sang datasets không có ID labels. Outperform FairMOT trên MOT17 và MOT20. Robust trong crowded scenes nhờ CrowdHuman training data.
- **Điểm yếu:** Tracker vẫn dùng FairMOT-style heuristic association. Không exploit temporal information giữa frames. Kết quả trên MOT17/20 test set chưa đầy đủ dưới private detection protocol.
- **Ý nghĩa cho dự án:** Chứng minh ReID branch hoạt động tốt trên YOLO11 decoupled head → baseline đáng tin cậy.

### 3.5 MOTRv2 (CVPR 2023)

- **Kiến trúc:** Deformable DETR + YOLOX proposals làm anchor queries cho MOTR.
- **Đặc điểm nổi bật:** Giải quyết conflict giữa detection và tracking trong MOTR bằng cách inject detection priors từ pretrained YOLOX.
- **Điểm mạnh:** SOTA DanceTrack (73.4 HOTA) — vượt trội ở complex non-linear motion. Association qua attention mechanism rất robust.
- **Điểm yếu:** Rất chậm (~7–10 FPS). Model nặng (Deformable DETR). Vẫn cần YOLOX bên ngoài → không truly end-to-end. Không phù hợp real-time hay edge deployment.

### 3.6 MO-YOLO / DecoderTracker (2024/2026)

- **Kiến trúc:** YOLOv8 backbone + neck + RT-DETR decoder (decoder-only, bỏ Transformer encoder).
- **Đặc điểm nổi bật:** Kết hợp YOLO efficiency với MOTR-style track queries qua decoder. DecoderTracker (2026) là phiên bản cải tiến với lightweight FENet.
- **Điểm mạnh:** 2x nhanh hơn MOTR (~19.6 FPS). Training nhanh hơn, ít GPU hơn. YOLO backbone dễ optimize.
- **Điểm yếu:** Vẫn chậm hơn TBD (19 vs 30+ FPS). DanceTrack OK nhưng MOT17 chưa SOTA. Decoder vẫn thêm latency đáng kể so với pure YOLO.
- **Ý nghĩa cho dự án:** Chứng minh YOLO + temporal decoder là hướng khả thi, nhưng full decoder quá nặng → cần giải pháp nhẹ hơn.

---

## 4. Bảng Benchmark tổng hợp

### 4.1 MOT17 Test Set — So sánh toàn diện

| Method | Paradigm | MOTA↑ | IDF1↑ | HOTA↑ | IDs↓ | FPS↑ | Backbone | Year |
|--------|----------|-------|-------|-------|------|------|----------|------|
| ByteTrack | TBD | 80.3 | 77.3 | 63.1 | 159 | ~30 | YOLOX-X | 2022 |
| BoT-SORT-ReID | TBD | 80.5 | 80.2 | 65.0 | 116 | ~20 | YOLOX-X + ReID | 2022 |
| StrongSORT++ | TBD | 79.6 | 79.5 | 64.4 | 143 | ~15 | YOLOX-X + OSNet | 2022 |
| OC-SORT | TBD | 78.0 | 77.5 | 63.2 | 942 | ~30 | YOLOX-X | 2022 |
| FairMOT | JDE | 73.7 | 72.3 | 59.3 | 3303 | ~30 | DLA-34 | 2021 |
| YOLO11-JDE | JDE | — | — | — | — | RT | YOLO11s | 2025 |
| MOTR | E2E | 73.4 | 68.6 | 57.8 | 2439 | ~9.5 | ResNet-50+DETR | 2022 |
| MOTRv2 | E2E | 78.6 | 75.0 | 62.0 | — | ~7–10 | Def.DETR+YOLOX | 2023 |
| MO-YOLO | E2E | ≈MOTR | — | ≈MOTR | — | ~19.6 | YOLOv8+decoder | 2024 |
| LITE(DeepSORT) | Hybrid | — | — | 43.0 | — | 28.3 | YOLOv8m | 2024 |

### 4.2 Nhận xét từ benchmark

1. **TBD vẫn thống trị MOT17** — ByteTrack/BoT-SORT với YOLOX đứng đầu. Heuristic-based association (Kalman + IoU) bất ngờ outperform learned association trên datasets có chuyển động tuyến tính.
2. **E2E Transformer mạnh ở complex motion** — MOTRv2 dẫn đầu DanceTrack (73.4 HOTA) nhờ attention-based association. TBD methods yếu ở non-linear motion.
3. **JDE gap lớn vs TBD** — FairMOT MOTA 73.7 vs ByteTrack 80.3 (gap 6.6). Lý do: backbone yếu (DLA-34 vs YOLOX-X) + conflict detect/ReID. YOLO11-JDE bắt đầu thu hẹp gap nhưng chưa đủ data trên test set.
4. **Speed gap rõ ràng** — TBD ~30 FPS, JDE ~30 FPS, E2E ~10–20 FPS. JDE nhanh nhưng accuracy thấp → gap này là cơ hội cho research.

---

## 5. Phân tích YOLO Versions cho JDT

### 5.1 Lý do cần phân tích

Dự án YOLO-JDT cần chọn base detector (YOLO variant) phù hợp nhất. Các tiêu chí đánh giá: detection accuracy (mAP), training stability, memory footprint, khả năng modify head, ecosystem support, compatibility với TAGate module, và tính mới cho publication.

### 5.2 YOLOv8 (Jan 2023, Ultralytics)

**Kiến trúc:** C2f backbone, PANet neck, Decoupled head (anchor-free), SiLU activation.

**Đánh giá cho JDT:**
- **Ưu:** Ecosystem lớn nhất. Training cực kỳ stable. Decoupled head dễ extend (thêm branch). Pure CNN backbone → TAGate attention tạo kết hợp complementary. Rất nhiều papers dùng làm baseline (fair comparison). MO-YOLO, DecoderTracker, nhiều tracking papers base trên v8.
- **Nhược:** Không phải mới nhất — mAP thấp hơn v11/v12/v26. C2f block kém hiệu quả hơn C3k2 (YOLO11).
- **mAP COCO (val):** Nano 37.3%, Small 44.9%, Medium 50.2%, Large 52.9%.
- **Phù hợp JDT: 8.5/10** — Rất tốt cho baseline và ablation study.

### 5.3 YOLOv9 (Feb 2024, Chien-Yao Wang, ECCV 2024)

**Kiến trúc:** GELAN backbone + PGI (Programmable Gradient Information). PGI dùng auxiliary reversible branch để giải quyết information bottleneck.

**Đánh giá cho JDT:**
- **Ưu:** PGI giải quyết gradient loss qua deep layers — lý thuyết có thể giúp joint training. GELAN giảm 49% params, 43% FLOPs so với v8 mà tăng 0.6% AP. Có paper ECCV → uy tín học thuật.
- **Nhược:** **Chỉ hỗ trợ detection** — không multi-task (segmentation, pose, tracking). PGI auxiliary branch chỉ dùng khi training, nhưng phức tạp hóa head modification. Ecosystem nhỏ, ít community support. Benchmark cho thấy **yếu ở small-sized models và small objects**. Không phải Ultralytics ecosystem → integration khó hơn.
- **mAP COCO (val):** v9-S ~46.8%, v9-C 51.4%, v9-E 55.6%.
- **Phù hợp JDT: 4.5/10** — Không khuyến nghị. PGI auxiliary branch phức tạp hóa head modification không cần thiết.

### 5.4 YOLO11 (Oct 2024, Ultralytics)

**Kiến trúc:** C3k2 backbone (cải tiến từ C2f), SPPF enhanced, Decoupled head, multi-task support đầy đủ.

**Đánh giá cho JDT:**
- **Ưu:** Cải tiến trực tiếp từ v8 — tương thích code. mAP cao hơn v8 ở mọi scale. Ít parameters hơn v8 (YOLO11n 2.6M vs v8n 3.2M). Training cực kỳ stable. Pure CNN backbone → TAGate complementary. Ultralytics chính thức recommend v11 hoặc v26 cho production. **YOLO11-JDE (WACV 2025) đã chứng minh JDE paradigm hoạt động tốt trên v11 → proof-of-concept mạnh.**
- **Nhược:** Cải tiến so với v8 chủ yếu ở backbone efficiency, không có breakthrough architecture mới.
- **mAP COCO (val):** Nano 39.5%, Small 47.0%, Medium 51.5%, X-Large 53.6%.
- **Phù hợp JDT: 9.0/10** — Lựa chọn tối ưu cho development chính. Stable, fast, proven for JDE, easy to extend.

### 5.5 YOLOv12 (Feb 2025, Community, NeurIPS 2025)

**Kiến trúc:** R-ELAN backbone + Area Attention (A2) module + FlashAttention + 7×7 separable convolutions. Đây là YOLO đầu tiên đưa attention mechanism vào backbone.

**Đánh giá cho JDT:**
- **Ưu:** mAP cao nhất trong nano-small range (v12-N 40.6%, v12-S 48.0%). Area Attention giúp detect small/occluded objects tốt hơn. NeurIPS publication → uy tín học thuật.
- **Nhược NGHIÊM TRỌNG:**
  - **Training KHÔNG ổn định** — Ultralytics chính thức thừa nhận. Tác giả gốc cũng phát hiện Ultralytics implementation bị lỗi, khuyến cáo dùng repo gốc thay vì Ultralytics library.
  - **Memory consumption cao hơn v11** — attention blocks tốn VRAM. Khi thêm TAGate (cũng là attention) → budget memory bị ép.
  - **CPU throughput chậm** — attention-heavy architecture không phù hợp non-GPU deployment.
  - **Community model, không phải Ultralytics official** → ít hỗ trợ long-term, risk maintenance.
  - **QUAN TRỌNG NHẤT — Attention redundancy:** YOLOv12 đã có attention (Area Attention) trong backbone. TAGate module đề xuất cũng là cross-attention. Khi chồng 2 tầng attention → diminishing returns + latency tăng không cần thiết. Ngược lại, v8/v11 dùng pure CNN backbone → TAGate là nguồn attention DUY NHẤT, tạo ra kết hợp CNN + Attention có tính **novel và complementary** cao hơn.
- **mAP COCO (val):** Nano 40.6%, Small 48.0%, Medium 51.9%, Large 53.2%.
- **Phù hợp JDT: 4.0/10** — Không khuyến nghị cho development chính. Có thể dùng trong ablation study để chứng minh "TAGate on CNN backbone > TAGate on attention backbone".

### 5.6 YOLO26 (Sep 2025, Ultralytics)

**Kiến trúc:** NMS-free end-to-end inference + ProgLoss + STAL (Small-Target-Aware Label Assignment) + MuSGD optimizer. Bỏ hoàn toàn NMS post-processing và Distribution Focal Loss (DFL).

**Đánh giá cho JDT:**
- **Ưu:**
  - **NMS-free** → end-to-end, giảm latency variance, đơn giản hóa deployment. Đây là tính năng rất quan trọng cho JDT: khi kết hợp v26 + TAGate = **"first truly end-to-end lightweight JDT pipeline"** — không NMS, không Kalman Filter, không separate tracker.
  - STAL giúp detect small targets tốt → quan trọng cho crowded tracking scenes.
  - Pareto front mới trên COCO: v26-m >53%, v26-l >55%, v26-x 56.3% mAP.
  - CPU inference giảm 43% so với v11-nano → phù hợp edge.
  - Ultralytics official → long-term support.
- **Nhược:**
  - **Quá mới** (Sep 2025) — chưa có nhiều papers dùng làm baseline.
  - NMS-free design thay đổi cách detection head hoạt động → modify head phức tạp hơn v8/v11.
  - STAL + ProgLoss chưa rõ tương tác thế nào khi thêm tracking loss (chưa ai thử).
  - Community chưa validate đủ trong diverse scenarios.
- **mAP COCO (val):** Medium >53%, Large >55%, X-Large 56.3%.
- **Phù hợp JDT: 7.5/10** — Tiềm năng rất cao cho novelty claim. Ideal cho phase mở rộng sau khi TAGate ổn định trên v11.

### 5.7 Bảng tổng hợp so sánh YOLO versions

| Tiêu chí | YOLOv8 | YOLOv9 | YOLO11 | YOLOv12 | YOLO26 |
|----------|--------|--------|--------|---------|--------|
| Nhà phát triển | Ultralytics | Chien-Yao Wang | Ultralytics | Community | Ultralytics |
| Backbone type | Pure CNN (C2f) | CNN (GELAN+PGI) | Pure CNN (C3k2) | CNN+Attention | Pure CNN (NMS-free) |
| Post-processing | NMS | NMS | NMS | NMS | **No NMS** |
| Multi-task | ✓ Full | ✗ Detect only | ✓ Full | ○ Partial | ✓ Full |
| Small mAP COCO | 44.9% | — | 47.0% | 48.0% | ~47–48% |
| Training stability | ✓ Rất ổn định | ✓ Ổn định | ✓ Rất ổn định | ✗ KHÔNG ổn định | ✓ Ổn định |
| Memory footprint | ✓ Thấp | ○ TB | ✓ Thấp | ✗ Cao | ✓ Thấp |
| Head modifiability | ✓ Rất dễ | ✗ Khó (PGI) | ✓ Rất dễ | ○ TB | ○ TB |
| MOT/Tracking papers | ✓ Nhiều | ✗ Hầu như không | ✓ YOLO11-JDE | ○ 1 paper | ✗ Chưa có |
| Attention redundancy với TAGate | ✓ Không (CNN only) | ✓ Không | ✓ Không (CNN only) | ✗ CÓ (A2) | ✓ Không |
| Phù hợp JDT | 8.5/10 | 4.5/10 | **9.0/10** | 4.0/10 | 7.5/10 |

### 5.8 Kết luận chọn backbone

**Primary backbone: YOLO11s/m** — Tối ưu cho development, training stability, và compatibility với TAGate.

**Extension backbone: YOLO26** — Port sau khi TAGate ổn định, tạo novelty claim "first NMS-free JDT".

**Ablation backbones: YOLOv8 (so sánh thế hệ), YOLOv12 (chứng minh CNN+TAGate > Attention+TAGate).**

---

## 6. Các hướng tiếp cận Novel đã khảo sát

Quá trình research đã xác định 5 hướng tiếp cận novel, đánh giá dựa trên novelty, feasibility, và alignment với mục tiêu dự án.

### 6.1 Hướng 1: YOLO-JDT với Temporal Feature Recycling (TAGate) ★ CHỌN

**Ý tưởng:** Lấy YOLO làm base detector, thêm một lightweight Temporal Attention Gate (TAGate) — chỉ 2–3 cross-attention layers — tái sử dụng features từ frame trước. TAGate tạo ra "temporal tokens" encode motion + appearance thay đổi, inject vào detection head để vừa detect vừa predict track association trong 1 pass.

**Tại sao novel:**
- MO-YOLO/DecoderTracker dùng full RT-DETR decoder → nặng. Chưa ai thử "micro-decoder" chỉ vài layers với feature recycling.
- YOLO11-JDE có ReID branch nhưng KHÔNG dùng temporal info giữa frames.
- ByteTrack/BoT-SORT dùng Kalman Filter ngoài model → không learn được complex motion patterns.
- Kết hợp CNN backbone + lightweight temporal attention = unique architectural contribution.

**Đánh giá: Novelty CAO, Feasibility CAO, Alignment HOÀN HẢO.**

### 6.2 Hướng 2: Distilled Joint Tracker (KD từ E2E Teacher → JDE Student)

**Ý tưởng:** Train model E2E nặng (MOTRv2) làm Teacher, distill knowledge sang model JDE nhẹ (YOLO + ReID branch). Teacher dạy Student cả detection quality lẫn association quality thông qua soft tracking labels.

**Tại sao novel:** KD đã dùng rộng rãi cho detection, nhưng chưa ai distill tracking knowledge (association patterns, trajectory prediction) từ E2E tracker sang JDE tracker. AttTrack (2022) chỉ distill feature representation, không distill association logic.

**Đánh giá: Novelty CAO, Feasibility TRUNG BÌNH (define "tracking knowledge" khó), Alignment TỐT.**

### 6.3 Hướng 3: Adaptive Motion-Appearance Fusion trong YOLO Head

**Ý tưởng:** Integrate learned motion predictor trực tiếp vào detection head. Model predict bounding box hiện tại + dự đoán vị trí frame tiếp theo. Kết hợp với lightweight appearance embedding qua gating mechanism (tự learn khi nào dùng motion cue, khi nào dùng appearance cue).

**Tại sao novel:** CenterTrack predict offset giữa 2 frames nhưng backbone yếu + không có appearance branch. Chưa ai kết hợp predicted next-frame offset + appearance embedding trong modern YOLO architecture với adaptive gating.

**Đánh giá: Novelty CAO, Feasibility CAO, Alignment TỐT.**

### 6.4 Hướng 4: Sparse Temporal Convolution thay thế Transformer Decoder

**Ý tưởng:** Dùng sparse temporal 1D convolution trên track features thay vì Transformer decoder (nặng) hay Kalman Filter (đơn giản quá). Track mỗi object bằng 1D conv kernel dọc theo time axis.

**Tại sao novel:** Sparse convolution cho 3D tracking đã có (Minkowski Tracker), nhưng chưa ai apply 1D temporal sparse conv cho 2D MOT.

**Đánh giá: Novelty TRUNG BÌNH, Feasibility CAO, Alignment TỐT.**

### 6.5 Hướng 5: YOLO + Mamba (State Space Model) cho Temporal Modeling

**Ý tưởng:** Mamba/S4 model long-range temporal dependencies với O(n) complexity thay vì O(n²) của Transformer. Dùng Mamba block nhỏ sau YOLO backbone để encode temporal context cho tracking.

**Tại sao novel:** Mamba cho vision đang hot (VMamba, Vim) nhưng chưa ai apply cụ thể cho joint detection-tracking MOT pipeline.

**Đánh giá: Novelty TRUNG BÌNH, Feasibility TRUNG BÌNH (Mamba integration chưa mature), Alignment TỐT.**

---

## 7. Hướng đi được chọn: YOLO-JDT với TAGate

### 7.1 Tên dự án

**YOLO-JDT: Joint Detection-Tracking via Temporal Attention Gate**

Tên thay thế khả dụng cho publication: TAGate-Track, TemporalYOLO, YOLO-TAT.

### 7.2 Core Contribution

1. **TAGate Module:** Lightweight Temporal Attention Gate (2–3 cross-attention layers) tái sử dụng cached features từ frame trước, inject temporal context vào detection pipeline mà không cần full Transformer decoder. Overhead tối thiểu so với base YOLO.
2. **Joint Detect-Track Head:** Mở rộng YOLO's decoupled head với 3 outputs đồng thời: (a) Detection (box + class), (b) ReID embedding, (c) Track offset prediction (dự đoán displacement giữa current và previous frame).
3. **First NMS-free JDT (khi port sang YOLO26):** Pipeline hoàn toàn end-to-end — không NMS, không Kalman Filter, không separate tracker, không Hungarian matching.

### 7.3 Tại sao hướng này đáp ứng mọi mục tiêu

| Mục tiêu | Giải pháp | Lý do |
|-----------|-----------|-------|
| Nhanh hơn TBD tách rời | 1 backbone inference + micro TAGate (vs TBD: 1 detector + 1 ReID network) | TAGate chỉ 2–3 layers, overhead ~1–2ms |
| Accuracy không giảm | Temporal features giúp detect tốt hơn (occluded objects), learned association tốt hơn Kalman | Temporal context cải thiện cả detection lẫn tracking |
| Model nhẹ | Không cần full Transformer decoder, không cần separate ReID network | TAGate < 1M params additional |
| Novelty | Chưa ai kết hợp micro temporal attention + CNN YOLO backbone cho JDT | Unique contribution, gap rõ ràng trong literature |

### 7.4 So sánh với các phương pháp liên quan

| Phương pháp | Temporal module | Backbone overhead | FPS dự kiến | Temporal reasoning |
|-------------|----------------|-------------------|-------------|-------------------|
| ByteTrack | Kalman (external) | Không | ~30 | Linear only |
| FairMOT | Không có | Không | ~30 | Không |
| YOLO11-JDE | Không có | Không | ~30+ | Không |
| MO-YOLO | Full RT-DETR decoder | Nặng (+50% latency) | ~19.6 | Track queries |
| MOTRv2 | Full Deformable DETR | Rất nặng | ~7–10 | Full attention |
| **YOLO-JDT (ours)** | **TAGate (2–3 layers)** | **Minimal (~5–10%)** | **~35–45 target** | **Learned cross-frame** |

---

## 8. Chiến lược thực hiện — Strategy C

### 8.1 Tổng quan chiến lược

**Strategy C: YOLO11 (chính) + YOLO26 (mở rộng) + v8/v12 (ablation)**

- Develop và validate toàn bộ kiến trúc trên **YOLO11s/m** — stable, fast iteration.
- Khi TAGate module hoạt động tốt, port sang **YOLO26** để tạo novelty claim "first NMS-free JDT".
- Ablation trên **YOLOv8** (so sánh thế hệ backbone) và **YOLOv12** (chứng minh CNN+TAGate > Attention+TAGate).
- Paper có 2 contributions rõ ràng: (1) TAGate module cho JDT, (2) First NMS-free JDT với YOLO26.

### 8.2 Phân chia các Phase

#### Phase 1: Baseline Setup & Measurement (Tuần 1–2)

**Mục tiêu:** Thiết lập baseline measurements để so sánh sau này.

- Setup YOLO11s + ByteTrack trên MOT17 train/val split.
- Đo: MOTA, IDF1, HOTA, IDs, FPS, latency breakdown (detect time + track time).
- Setup YOLO11s + BoT-SORT-ReID → đo metrics tương tự.
- Xác định target metrics: TAGate phải đạt ≥ baseline accuracy, FPS > baseline.

#### Phase 2: JDE Integration — ReID Branch (Tuần 3–4)

**Mục tiêu:** Thêm ReID branch vào YOLO11 head, tái tạo YOLO11-JDE.

- Fork YOLO11-JDE paper implementation hoặc tự implement ReID branch vào YOLO11's decoupled head.
- ReID branch: 2 Conv3×3 layers + BN + SiLU → output ReID embedding vector (128-dim hoặc 256-dim).
- Training: Dùng CrowdHuman + MOT17 train split. Self-supervised ReID qua Mosaic augmentation.
- Validation: So sánh với FairMOT baseline trên MOT17 val.

#### Phase 3: TAGate Module Design & Integration (Tuần 5–8)

**Mục tiêu:** Core contribution — design, implement, và integrate TAGate.

**TAGate Architecture (đề xuất ban đầu):**
- Input: Feature map hiện tại F_t (từ YOLO FPN P5 level) + Cached feature map F_{t-1} (frame trước).
- Module: 2–3 Cross-Attention layers. Query = F_t, Key/Value = F_{t-1}.
- Output: Temporal-enhanced feature map F'_t = F_t + α × CrossAttn(F_t, F_{t-1}).
- α: Learnable gating parameter, khởi tạo = 0 (residual learning).
- F'_t được feed vào Detection head + ReID head + Track Offset head.

**Feature Caching Strategy:**
- Chỉ cache FPN P5 level (lowest resolution, richest semantic) → giảm memory.
- Cache size = 1 frame (chỉ frame ngay trước). Có thể mở rộng sang N frames sau.
- Cache update: overwrite sau mỗi frame.

**Training Strategy:**
- Stage 1: Freeze YOLO11 backbone pretrained, chỉ train TAGate + ReID head + Track Offset head (Tuần 5–6).
- Stage 2: Unfreeze backbone, fine-tune toàn bộ end-to-end với lower learning rate (Tuần 7–8).
- Loss function: L_total = λ_det × L_detect + λ_reid × L_ReID + λ_offset × L_offset.
- Dùng uncertainty weighting (Kendall et al., 2018) để tự động balance λ values → tránh gradient conflict.

#### Phase 4: Track Offset Prediction Head (Tuần 7–9, overlap Phase 3)

**Mục tiêu:** Thêm head predict displacement giữa frame hiện tại và frame trước.

- Track Offset Head: Predict (Δx, Δy) cho mỗi detected object — displacement từ position ở frame t–1 sang frame t.
- Loss: Smooth L1 loss trên ground truth offsets (từ GT bounding box annotations across frames).
- Khi inference: Offset prediction + ReID embedding + IoU → multi-cue association.
- Association algorithm: Lightweight, combine 3 cues với learned/fixed weights.

#### Phase 5: Ablation Studies & Benchmark (Tuần 10–13)

**Mục tiêu:** Validate TAGate effectiveness, benchmark trên MOT17/MOT20/DanceTrack.

**Ablation experiments:**
1. TAGate vs No TAGate (pure JDE) → chứng minh temporal module có ích.
2. TAGate vs Kalman Filter → chứng minh learned temporal > handcrafted linear.
3. TAGate vs LSTM → chứng minh cross-attention > recurrent.
4. TAGate vs Full Transformer Decoder (MO-YOLO style) → chứng minh micro-decoder đủ tốt.
5. TAGate layers: 1 vs 2 vs 3 vs 4 → tìm sweet spot.
6. Cache levels: P5 only vs P4+P5 vs P3+P4+P5 → trade-off memory vs accuracy.
7. Association cues: offset only vs ReID only vs offset+ReID vs offset+ReID+IoU.

**Backbone ablation:**
8. YOLO11 vs YOLOv8 backbone → thế hệ comparison.
9. YOLO11 + TAGate vs YOLOv12 + TAGate → CNN+Attention vs Attention+Attention.

#### Phase 6: YOLO26 Port & NMS-free JDT (Tuần 14–17)

**Mục tiêu:** Port TAGate sang YOLO26, tạo second contribution.

- Adapt TAGate module cho YOLO26's NMS-free head.
- Xử lý integration giữa YOLO26's end-to-end assignment và tracking assignment.
- Benchmark trên MOT17/MOT20/DanceTrack.
- So sánh: YOLO11-JDT (with NMS) vs YOLO26-JDT (NMS-free) → analyze latency benefit.

#### Phase 7: Paper Writing & Final Experiments (Tuần 18–20)

**Mục tiêu:** Viết paper, chạy final experiments, prepare submission.

### 8.3 Timeline tổng quan

```
Tuần  1──2──3──4──5──6──7──8──9──10──11──12──13──14──15──16──17──18──19──20
      ├──────┤                                                              Phase 1: Baseline
            ├──────┤                                                        Phase 2: JDE/ReID
                  ├──────────────┤                                          Phase 3: TAGate
                        ├──────────┤                                        Phase 4: Track Offset
                                    ├──────────────┤                        Phase 5: Ablation
                                                    ├──────────────┤        Phase 6: YOLO26 Port
                                                                    ├──────┤Phase 7: Paper
```

---

## 9. Thiết kế kiến trúc đề xuất

### 9.1 Pipeline tổng quan

```
Video Frame (t)
       │
       ▼
┌──────────────────┐
│  YOLO11 Backbone │──── Feature Maps (P3, P4, P5)
│  (C3k2 + SPPF)  │
└──────────────────┘
       │
       ▼
┌──────────────────┐     ┌─────────────────────┐
│   YOLO11 Neck    │     │   Feature Cache     │
│   (PANet FPN)    │────▶│   (P5 from t-1)     │
└──────────────────┘     └─────────────────────┘
       │                          │
       ▼                          │
┌──────────────────────────────────────────┐
│           TAGate Module                   │
│  ┌────────────────────────────────────┐  │
│  │  Cross-Attention Layer 1           │  │
│  │  Q = F_t(P5), K/V = F_{t-1}(P5)   │  │
│  ├────────────────────────────────────┤  │
│  │  Cross-Attention Layer 2           │  │
│  ├────────────────────────────────────┤  │
│  │  FFN + Gated Residual              │  │
│  │  F'_t = F_t + α·Attn(F_t, F_{t-1})│  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│        Joint Detect-Track Head            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Detect   │ │ ReID     │ │ Track    │ │
│  │ Head     │ │ Head     │ │ Offset   │ │
│  │ (Box+Cls)│ │ (Embed)  │ │ Head     │ │
│  │          │ │ (128-d)  │ │ (Δx, Δy) │ │
│  └──────────┘ └──────────┘ └──────────┘ │
└──────────────────────────────────────────┘
       │               │              │
       ▼               ▼              ▼
┌──────────────────────────────────────────┐
│     Lightweight Association Module        │
│  IoU matching + ReID cosine similarity    │
│  + Track offset prediction matching       │
│  → Hungarian Algorithm → Track IDs        │
└──────────────────────────────────────────┘
       │
       ▼
   Output: Detected + Tracked Objects with IDs
```

### 9.2 TAGate Module — Chi tiết

**Input dimensions (ví dụ với YOLO11s, input 640×640):**
- P5 feature map: 20×20×512
- Cached P5 (t-1): 20×20×512

**Cross-Attention Layer:**
- Flatten spatial: 400 tokens × 512 dim
- Multi-head attention: 8 heads, head_dim = 64
- Complexity: O(400 × 400 × 512) — rất nhỏ so với full image attention

**Gated Residual Connection:**
```
α = sigmoid(learnable_param)  # Khởi tạo param = -2 → α ≈ 0.12
F'_t = F_t + α × CrossAttn(F_t, F_{t-1})
```
α khởi tạo gần 0 để ban đầu model hoạt động gần giống base YOLO, dần dần learn temporal contribution.

**Estimated overhead:**
- Parameters: ~0.5–1M additional (2 cross-attention layers + FFN)
- Latency: ~1–2ms additional trên T4 GPU
- Memory: +20×20×512×4 bytes ≈ 0.8MB cho feature cache

### 9.3 Joint Detect-Track Head

**Detect Head (giữ nguyên YOLO11):**
- Box regression: 4 values (x, y, w, h)
- Classification: C classes

**ReID Head (thêm mới):**
- Input: Feature map từ TAGate output
- Architecture: Conv3×3(512→256) → BN → SiLU → Conv3×3(256→128) → L2 Normalize
- Output: 128-dim embedding vector cho mỗi detection
- Loss: Cross-entropy loss với identity labels (hoặc triplet loss)

**Track Offset Head (thêm mới):**
- Input: Feature map từ TAGate output
- Architecture: Conv3×3(512→256) → BN → SiLU → Conv1×1(256→2)
- Output: (Δx, Δy) displacement prediction
- Loss: Smooth L1 loss với ground truth offsets

### 9.4 Association Algorithm

```python
# Pseudocode cho association
def associate(detections_t, tracks_t_minus_1):
    # Stage 1: High-confidence detections
    high_conf = [d for d in detections_t if d.score > τ_high]
    
    # Cost matrix = weighted combination
    C_iou = 1 - iou_matrix(high_conf, tracks_t_minus_1)
    C_reid = 1 - cosine_similarity(high_conf.embeddings, tracks_t_minus_1.embeddings)
    C_offset = l2_distance(high_conf.positions, tracks_t_minus_1.positions + tracks_t_minus_1.predicted_offsets)
    
    C = w_iou * C_iou + w_reid * C_reid + w_offset * C_offset
    
    matched, unmatched_dets, unmatched_tracks = hungarian(C)
    
    # Stage 2: Low-confidence detections (ByteTrack style)
    low_conf = [d for d in detections_t if τ_low < d.score <= τ_high]
    C_iou_2 = 1 - iou_matrix(low_conf, unmatched_tracks)
    matched_2 = hungarian(C_iou_2)
    
    return matched + matched_2
```

### 9.5 Loss Function

```
L_total = λ_det × L_detect + λ_reid × L_ReID + λ_offset × L_offset

L_detect = L_box (CIoU) + L_cls (BCE) + L_dfl (Distribution Focal Loss)
L_ReID = CrossEntropy(predicted_id, ground_truth_id)  # hoặc Triplet Loss
L_offset = SmoothL1(predicted_offset, gt_offset)

# Uncertainty weighting (tự động balance):
L_total = (1/2σ₁²) × L_detect + (1/2σ₂²) × L_ReID + (1/2σ₃²) × L_offset + log(σ₁σ₂σ₃)
# σ₁, σ₂, σ₃ là learnable parameters
```

---

## 10. Kế hoạch thí nghiệm & Đánh giá

### 10.1 Datasets

| Dataset | Đặc điểm | Mục đích | Sequences |
|---------|----------|----------|-----------|
| MOT17 | Standard benchmark, linear motion, pedestrian | Primary benchmark, fair comparison | 7 train + 7 test |
| MOT20 | Extremely crowded (246 ped/frame) | Test dense scene handling | 4 train + 4 test |
| DanceTrack | Non-linear complex motion, similar appearance | Test temporal reasoning quality | 40 train + 25 val + 35 test |
| CrowdHuman | Static images, dense pedestrians | Pre-training detection + ReID | ~15K train + 4.4K val |

### 10.2 Metrics

| Metric | Ý nghĩa | Thiên về | Mục tiêu |
|--------|---------|----------|----------|
| HOTA | Higher Order Tracking Accuracy — metric tổng hợp | Balanced detection + association | ≥63 (MOT17) |
| MOTA | Multi-Object Tracking Accuracy | Detection quality | ≥80 (MOT17) |
| IDF1 | ID F1-Score | Association quality | ≥77 (MOT17) |
| IDs | Identity Switches | Association stability | ≤200 (MOT17) |
| FPS | Frames Per Second | Speed | ≥35 (target) |
| Params | Model parameters | Model size | <base YOLO + 1M |
| FLOPs | Floating point operations | Compute cost | <base YOLO + 5% |

### 10.3 Baselines để so sánh

| Category | Methods |
|----------|---------|
| TBD (separate) | ByteTrack (YOLOX), BoT-SORT (YOLOX), BoT-SORT-ReID |
| TBD (YOLO11) | YOLO11 + ByteTrack, YOLO11 + BoT-SORT |
| JDE | FairMOT, YOLO11-JDE |
| E2E | MOTR, MOTRv2, MO-YOLO |
| Ours | YOLO11-JDT (TAGate), YOLO26-JDT (NMS-free) |

### 10.4 Ablation Study Design

| # | Experiment | So sánh | Chứng minh |
|---|-----------|---------|------------|
| A1 | w/ TAGate vs w/o TAGate | Full model vs pure JDE | TAGate module effectiveness |
| A2 | TAGate vs Kalman Filter | Temporal attention vs linear motion | Learned temporal > handcrafted |
| A3 | TAGate vs LSTM | Cross-attention vs recurrent | Architecture choice |
| A4 | TAGate vs Full Decoder | 2–3 layers vs full RT-DETR decoder | Micro-decoder sufficient |
| A5 | TAGate 1/2/3/4 layers | Layer count sweep | Optimal depth |
| A6 | Cache P5 vs P4+P5 vs P3-P5 | Feature level sweep | Memory-accuracy trade-off |
| A7 | Offset+ReID+IoU cues | Ablate each association cue | Contribution of each cue |
| A8 | YOLO11 vs YOLOv8 backbone | Backbone generation | Architecture improvement |
| A9 | YOLO11+TAGate vs YOLOv12+TAGate | CNN+Attn vs Attn+Attn | Attention complementarity |
| A10 | YOLO11-JDT vs YOLO26-JDT | NMS vs NMS-free | NMS-free benefit for tracking |

---

## 11. Rủi ro & Giải pháp

| # | Rủi ro | Mức độ | Giải pháp |
|---|--------|--------|-----------|
| R1 | Gradient conflict giữa detect/ReID/offset khi joint training | Cao | Uncertainty weighting loss. Stage-wise training (freeze backbone first). Gradient scaling per task. |
| R2 | TAGate overhead vượt quá budget → FPS < target | Trung bình | Giảm TAGate xuống 1 layer. Dùng efficient attention (FlashAttention). Chỉ cache P5 (smallest feature map). |
| R3 | Feature cache tốn memory trên edge devices | Trung bình | P5 only = ~0.8MB. Có thể quantize cached features sang FP16/INT8. |
| R4 | YOLO26 port gặp khó khăn do NMS-free head khác biệt | Trung bình | Develop modular TAGate → dễ plug vào bất kỳ YOLO head. YOLO26 vẫn dùng decoupled head → tương thích. |
| R5 | Training data thiếu temporal annotations | Thấp | MOT17/MOT20 đều có frame-by-frame GT. CrowdHuman dùng cho pre-training detection+ReID, temporal training trên MOT datasets. |
| R6 | Reviewer yêu cầu so sánh với method quá mới (xuất hiện trong review period) | Thấp | Duy trì codebase flexible, dễ thêm baseline comparison. |
| R7 | YOLO26 ecosystem chưa mature → bugs không lường trước | Trung bình | Develop chính trên YOLO11 (mature). YOLO26 là extension, không phải dependency. Có thể bỏ YOLO26 port mà paper vẫn có contribution rõ ràng (TAGate trên YOLO11). |

---

## 12. Tài liệu tham khảo

### Detection Models

- YOLOv8: Ultralytics (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
- YOLOv9: Wang, C.-Y., Yeh, I.-H., & Liao, H.-Y. M. (2024). YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information. ECCV 2024.
- YOLO11: Ultralytics (2024). YOLO11. https://docs.ultralytics.com/models/yolo11/
- YOLOv12: (2025). YOLOv12: Attention-Centric Real-Time Object Detectors. NeurIPS 2025.
- YOLO26: Sapkota, R. et al. (2025). YOLO26: Key Architectural Enhancements and Performance Benchmarking. arXiv:2509.25164.
- YOLOX: Ge, Z. et al. (2021). YOLOX: Exceeding YOLO Series in 2021. arXiv:2107.08430.

### Tracking-by-Detection

- SORT: Bewley, A. et al. (2016). Simple Online and Realtime Tracking. ICIP 2016.
- DeepSORT: Wojke, N. et al. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. ICIP 2017.
- ByteTrack: Zhang, Y. et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV 2022.
- BoT-SORT: Aharon, N. et al. (2022). BoT-SORT: Robust Associations Multi-Pedestrian Tracking. arXiv:2206.14651.
- OC-SORT: Cao, J. et al. (2023). Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking. CVPR 2023.
- StrongSORT: Du, Y. et al. (2023). StrongSORT: Make DeepSORT Great Again. IEEE TMM.

### Joint Detection & Embedding

- JDE: Wang, Z. et al. (2020). Towards Real-Time Multi-Object Tracking. ECCV 2020.
- FairMOT: Zhang, Y. et al. (2021). FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking. IJCV 2021.
- RetinaTrack: Lu, Z. et al. (2020). RetinaTrack: Online Single Stage Joint Detection and Tracking. CVPR 2020.
- YOLO11-JDE: Erregue et al. (2025). YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID. WACV 2025.

### End-to-End Transformer

- MOTR: Zeng, F. et al. (2022). MOTR: End-to-End Multiple-Object Tracking with Transformer. ECCV 2022.
- MOTRv2: Zhang, Y. et al. (2023). MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors. CVPR 2023.
- TrackFormer: Meinhardt, T. et al. (2022). TrackFormer: Multi-Object Tracking with Transformers. CVPR 2022.
- MO-YOLO: Liao, P. et al. (2024). MO-YOLO: End-to-End Multiple-Object Tracking Method with YOLO and Decoder. arXiv:2310.17170.
- DecoderTracker: Liao, P. et al. (2026). DecoderTracker: Decoder-Only End-to-End Method for Multiple-Object Tracking. Pattern Recognition 2026.
- CO-MOT: (2024). Boosting End-to-end Transformer-based Multi-Object Tracking via Coopetition Label Assignment and Shadow Sets. ICLR 2024.

### Hybrid & Others

- LITE: (2024). LITE: A Paradigm Shift in Multi-Object Tracking with Efficient ReID Feature Integration. arXiv:2409.04187.
- DEFT: Chaabane, M. et al. (2021). DEFT: Detection Embeddings for Tracking. arXiv:2102.02267.
- AttTrack: (2022). AttTrack: Online Deep Attention Transfer for Multi-Object Tracking. arXiv:2210.08648.

### Knowledge Distillation for Detection

- KD-DETR: (2022). KD-DETR: Knowledge Distillation for Detection Transformer. arXiv:2211.08071.
- Shared-KD: (2024). Shared Knowledge Distillation Network for Object Detection. Electronics 2024.

### Metrics & Benchmarks

- HOTA: Luiten, J. et al. (2021). HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking. IJCV 2021.
- MOT17: Milan, A. et al. (2016). MOT16: A Benchmark for Multi-Object Tracking. arXiv:1603.00831.
- MOT20: Dendorfer, P. et al. (2020). MOT20: A Benchmark for Multi Object Tracking in Crowded Scenes. arXiv:2003.09003.
- DanceTrack: Sun, P. et al. (2022). DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion. CVPR 2022.

### Multi-task Learning

- Kendall, A. et al. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. CVPR 2018.

---

## Phụ lục A: Glossary

| Thuật ngữ | Giải thích |
|-----------|------------|
| TBD | Tracking-by-Detection — detect trước, track sau |
| JDE | Joint Detection and Embedding — detect + ReID cùng network |
| E2E | End-to-End — toàn bộ pipeline trong 1 forward pass |
| JDT | Joint Detection and Tracking — đề xuất của project này |
| TAGate | Temporal Attention Gate — module temporal attention nhẹ |
| ReID | Re-Identification — nhận dạng lại object qua frames |
| MOTA | Multi-Object Tracking Accuracy |
| HOTA | Higher Order Tracking Accuracy |
| IDF1 | ID F1-Score — đo quality of identity preservation |
| IDs | Identity Switches — số lần đổi ID sai |
| NMS | Non-Maximum Suppression — loại bỏ duplicate detections |
| FPN | Feature Pyramid Network — multi-scale feature extraction |
| CMC | Camera Motion Compensation |
| GELAN | Generalized Efficient Layer Aggregation Network (YOLOv9) |
| PGI | Programmable Gradient Information (YOLOv9) |
| R-ELAN | Residual ELAN (YOLOv12) |
| A2 | Area Attention module (YOLOv12) |
| STAL | Small-Target-Aware Label Assignment (YOLO26) |

---

## Phụ lục B: Checklist trước khi bắt đầu code

- [ ] Setup môi trường: Python 3.10+, PyTorch 2.x, CUDA 12.x, Ultralytics library
- [ ] Download datasets: MOT17, MOT20, DanceTrack, CrowdHuman
- [ ] Download pretrained weights: YOLO11s, YOLO11m, YOLOv8s (ablation), YOLO26s (later)
- [ ] Setup evaluation tools: TrackEval (official MOTChallenge evaluator)
- [ ] Setup logging: WandB hoặc TensorBoard
- [ ] Baseline experiments: YOLO11 + ByteTrack, YOLO11 + BoT-SORT
- [ ] Fork/study code: YOLO11-JDE, MO-YOLO, FairMOT tracker
- [ ] Design TAGate module (PyTorch implementation draft)
- [ ] Plan GPU resources: ≥1× RTX 3090/4090 hoặc A100 cho training

---

*Document Version: 1.0 — Created: 09/05/2026*
*Cập nhật tiếp theo: sau Phase 1 baseline results*
