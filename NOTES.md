# Project Notes / Interview Prep Log

Running log of decisions, tools, problems, solutions, and results for the Driving Risk Analyzer.
Written so I can re-read an entry months later and still explain *why*, not just *what*.

---

## 2026-09-10 — Initial planning

Talked through the project plan (originally sketched with ChatGPT) and stress-tested it. Three corrections that shaped the design:

1. **A single dashcam can't give you your own speed.** Tailgating (time headway), hard braking, and closing speed all need ego speed. Fix: record footage with an iPhone that also logs GPS (and eventually accelerometer) alongside video, giving ego speed directly *and* ground truth to check against. Estimating speed from video alone and comparing it to GPS is a good stretch goal, not the primary method — vision-only ego-speed estimation is a much harder, separate problem.

2. **TTC does not need distance in meters.** The naive approach is `TTC = distance / closing_speed`, which requires camera calibration first (week 4-5 work). Instead: if the lead car's bounding-box width is `w`, then `TTC ≈ w / (dw/dt)` — if the box is growing 10%/s, you hit it in ~10s. This is a classic monocular-TTC trick (time-to-contact from optical expansion), needs zero calibration, and is a real geometry insight, not just library glue. Decision: build TTC from box-width expansion in week 2, smoothed with a Kalman filter (raw frame-to-frame width is noisy). Meter-based distance comes later, independently, once the camera is calibrated.

3. **A "learned risk model" trained on my own rule-based labels is circular** — it would just relearn the rules I wrote, and any interviewer will spot that immediately. Decided against it. If there's a real ML component, it has to train on real labels: accident anticipation on DoTA/CCD/DAD (public datasets of actual crashes/near-misses), predicting a collision ~2s before it happens, with published baselines to compare against. This became the designated stretch-goal ML component instead of a homegrown scoring model.

**Also decided:**
- Differentiator vs. the sea of "YOLO on dashcam video" GitHub repos: report actual accuracy numbers against ground truth (KITTI for distance, KITTI tracking benchmark for HOTA/IDF1, hand-labeled clips for event precision/recall), not just a cool annotated video.
- Streamlit over React+FastAPI for the report UI — this is an ML portfolio piece, not a full-stack demo. FastAPI/React would be scope creep (see "Not doing" in CLAUDE.md).
- Ultralytics is AGPL-licensed — fine for an open-source portfolio repo, but noted in case it comes up.
- Test footage: ride as a passenger (never film while driving), iPhone in "Most Compatible" format + 1080p30, landscape, roughly level and aimed straight ahead. Alignment/height/calibration precision only matters starting at the meters-based distance stage (week 4-5) — detection, tracking, and box-expansion TTC don't care about exact mounting.
- If no car ride is available immediately, use free stock dashcam footage (pexels.com) to unblock `track.py` development, then swap in real footage later.

---

## 2026-09-10 — Environment setup

**Stack chosen:** Python 3.12 + `uv` for env/dependency management, PyTorch (CPU build), Ultralytics YOLO, OpenCV, pandas, matplotlib, scipy, NumPy.

**Problem: PyTorch defaults to the CUDA build.** `uv add torch` would otherwise pull a multi-GB CUDA wheel that's useless on this laptop (integrated Radeon, no NVIDIA GPU) and just wastes disk/bandwidth. Fix: added a dedicated `pytorch-cpu` index in `pyproject.toml` pointing at `https://download.pytorch.org/whl/cpu`, with `explicit = true` (so it's *only* used for packages explicitly routed to it, not for every package resolution) and `[tool.uv.sources]` entries routing `torch`/`torchvision` there. Confirmed working: installed `torch==2.14.0+cpu` (the `+cpu` suffix is the tell), and `torch.cuda.is_available()` returns `False` as expected.

**WSL + OpenCV:** the classic WSL failure mode is `cv2` import blowing up with `ImportError: libGL.so.1: cannot open shared object file` because the machine is headless and missing GL/GLib system libraries. Checked first — `libgl1`, `libglib2.0-0`, and `ffmpeg` were already installed on this machine, so no `apt install` was needed. Worth re-checking if this ever gets set up on a fresh WSL install.

**uv also auto-managed the Python version.** `.python-version` pins `3.12`, but the system Python was `3.10.12`. `uv add` silently downloaded CPython 3.12.13 into its own managed location and built `.venv` against that — no manual pyenv/apt juggling needed. This is one of the reasons `uv` was chosen over pip/venv.

**Result:** all imports verified in one shot (`torch`, `cv2`, `pandas`, `matplotlib`, `scipy`, `ultralytics`) via `uv run python -c "..."`. `ultralytics` auto-created `~/.config/Ultralytics/settings.json` on first import — expected, not an error.

**Not yet done:** `yolo26n.pt` weights (auto-downloads on first `track.py` run) and actual test footage in `data/raw/`.

---

## 2026-09-10 — track.py first run, ID-flicker debugging

First working run of `track.py` on `data/raw/clip1.mov` (84s). Noticed heavy ID flicker in the annotated output: way more unique `track_id`s than plausible real vehicles.

**Diagnosis, from the tracks CSV:**
- 197 unique track IDs over ~2,500 frames, median track length only 8 frames (~0.3s) — real vehicles were instead showing up as several hundred-frame-long tracks (the top handful ran 400-943 frames), meaning most of the 197 IDs were noise, not distinct cars.
- Compared confidence of short-lived tracks (<10 frames) vs long-lived ones (>=100 frames): 0.46 mean conf vs 0.60. Short tracks clustered right around the `--conf 0.3` threshold — the classic pattern of a detection's confidence oscillating just above/below the cutoff frame to frame, so it blinks in and out of existence and the tracker assigns a new ID each time it reappears (rather than bridging the gap).

**Ablation — tracker choice vs confidence threshold:**

| Config | Unique IDs | Median track len | Tracks <10 frames |
|---|---|---|---|
| ByteTrack, conf=0.3 (baseline) | 197 | 8 frames | 101 (51%) |
| BoT-SORT, conf=0.3 | 196 | 10 frames | 97 (49%) |
| ByteTrack, conf=0.5 | 109 | 12 frames | 50 (46%) |

Switching trackers (ByteTrack -> BoT-SORT) barely moved the numbers. Raising the confidence threshold to 0.5 roughly halved the number of unique IDs. Conclusion: the flicker was mostly a detection-confidence problem, not a tracker re-identification weakness. Set `--conf` default to 0.5 in `track.py` as a result. Tracker default left as `bytetrack.yaml` (faster, and BoT-SORT showed no measurable benefit on this clip).

**Also noticed:** `clip1.mov` is actually 3840x2160 (4K), not the 1080p30 the environment notes call for. This is very likely why CPU inference was slow (~6-8 fps processing on an 84s clip took 5-7 minutes) — 4K has ~4x the pixels of 1080p per frame. Re-shooting or downscaling future clips to 1080p should speed this up substantially. Not fixed yet, flagged for next session.

---

## 2026-09-10 — Remaining flicker traced to inference resolution (imgsz)

Raising `--conf` (above) fixed most of the flicker, but distant, fully unobstructed cars straight ahead were still flickering — and specifically flickering *without* the track ID changing (the box would blink off then back on under the same ID). That's a different failure mode than the conf/ID-switch problem: ByteTrack was correctly bridging short gaps and keeping the ID alive, but the car just wasn't being *detected* on a lot of frames in between.

**Root cause:** `model.track()` resizes frames to `imgsz` (default 640) before running detection, regardless of source resolution. The source video is 4K (3840x2160), so a distant car that's ~130px wide in the original frame gets shrunk to roughly ~20px wide at the default 640 inference size — right at the edge of what any detector can reliably see. This is an input-resolution problem, not a detector-quality problem, so it doesn't call for fine-tuning or a bigger model — just feeding the detector more real pixels for small objects.

**Test:** trimmed `clip1.mov` to a 30s clip (`data/raw/clip1_30s.mov`, via `ffmpeg -t 30 -c copy`, same 4K resolution) for faster iteration, then compared `imgsz=640` vs `imgsz=1280`:

| imgsz | Unique IDs | Median track len | Detection rate on real tracks (>=10 frames) |
|---|---|---|---|
| 640 (old default) | 40 | 16 frames | 76.0% (24 qualifying tracks) |
| 1280 | 51 | 24 frames | 85.3% (34 qualifying tracks) |

Unique-ID count going *up* at first looked like more flicker, but it's the opposite: `imgsz=1280` finds more real, distinct cars overall (more tracks reach the 10-frame bar to count as "real" instead of vanishing as sub-10-frame blips), and for tracks that do qualify, detection rate within the track rose from 76% to 85% (fewer blinks per car). Visually confirmed: much better on distant/unobstructed cars.

**Decision:** set `--imgsz` default to 1280 in `track.py` (was 640). Tradeoff: slower CPU inference — worth watching processing fps on future runs and considering downscaling raw footage to 1080p (see prior entry) if this becomes a bottleneck. `imgsz` stays as a CLI flag so it's easy to trade off speed vs. small-object recall per run.

---

<!-- Add new dated entries above this line as the project progresses. -->
