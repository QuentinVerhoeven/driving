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

<!-- Add new dated entries above this line as the project progresses. -->
