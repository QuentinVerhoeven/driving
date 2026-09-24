Driving Risk Analyzer

A computer vision system that analyzes dashcam video to measure driving risk: detect and track vehicles, identify the lead vehicle, estimate time-to-collision (TTC), distance, and headway, detect risky events (tailgating, hard braking, rapid closing), and produce a report. Portfolio project for ML internship applications.

Two deliverables matter equally:

A live demo video: the system running in real time on a laptop in a car, with live boxes, TTC, and an audio alert.
Quantitative evaluation: accuracy numbers against ground truth, reported in a README results table.
About me and how to work with me
I'm a student learning ML/CV. I need to understand and explain every part of this in interviews.
Explain what code does and why before or while writing it. Prefer small, readable scripts over clever abstractions.
Build one layer at a time. Don't start a new stage until the current one works on real footage.
Every stage should produce something I can look at: a video, CSV, or plot.
Keep changes small and suggest a git commit whenever something works.
Don't add infrastructure I didn't ask for (see "Not doing").
Log everything in NOTES.md as we go: tools/libraries chosen and why, design decisions, problems hit and how they were solved, results/numbers, and the reasoning behind tradeoffs. This is interview prep material, not documentation for its own sake — write it so I can read an entry months later and still explain the decision. Add an entry whenever a stage finishes, a non-trivial problem gets solved, or a design choice gets made, without waiting to be asked.
Environment
Windows laptop, working in WSL (Ubuntu) inside VS Code. Project lives in ~/driving-risk-analyzer, not under /mnt/c.
CPU only (integrated Radeon, no NVIDIA). PyTorch is the CPU build via a pytorch-cpu uv index in pyproject.toml. Never install CUDA packages.
Python 3.12, managed with uv (uv add, uv run).
Heavy jobs (depth models, batch processing, training) run on Kaggle GPUs. Local work uses short clips (30-60 s) and nano models.
The live demo will run in native Windows Python (WSL can't easily access webcams), so live-capture code must not depend on WSL-specific paths.
Sensor: iPhone video (1080p30, H.264) plus GPS logging for ego speed.
Stack

Python, Ultralytics YOLO (yolo26n.pt), built-in trackers (ByteTrack, BoT-SORT), OpenCV, NumPy/SciPy, pandas, matplotlib, PyTorch, scikit-learn, Streamlit for the post-drive report. Later: ONNX/OpenVINO export for faster CPU inference.

Pipeline
Detection (pretrained YOLO, vehicle classes only)
Tracking (consistent IDs across frames)
Lead-vehicle selection (the tracked car directly ahead, in my lane)
Measurement:
TTC from bounding-box expansion: TTC ≈ w / (dw/dt), smoothed with a Kalman filter. Needs no calibration.
Ego speed from iPhone GPS.
Distance in meters (camera calibration + ground-plane geometry, later a metric depth model).
Time headway = distance / ego speed.
Events from patterns over time (sustained low headway, rapidly dropping TTC, hard braking)
Scoring + Streamlit report (video, TTC plot, clickable event timeline)

Stretch ML component: accident anticipation on DoTA/CCD (real crash and near-miss clips), with baselines first.

Evaluation plan
Distance accuracy on KITTI (LiDAR ground truth), comparing ground-plane geometry vs a depth model, reported by range.
Tracker comparison (ByteTrack vs BoT-SORT) on a standard tracking benchmark (HOTA/IDF1).
Event detection precision/recall on hand-labeled clips (BDD100K + my own drives).
Roadmap
Week	Milestone
1	Detection + tracking on a recorded clip (track.py)
2	Lead-vehicle selection + TTC on recorded clips, first TTC plot
3	First live version: webcam on windshield, live boxes + TTC + audio alert
4-5	GPS speed, calibration, distance, headway, events
5-7	KITTI/BDD100K evaluation, accident-anticipation model
8+	Final live demo, Streamlit report, polished README
Not doing

No React/FastAPI, no database, no cloud deployment, no Docker (for now), no custom detector from scratch, no mobile app, no Google Maps integration. Don't claim it's a safety system.

Current status

Week 2. track.py runs YOLO + tracking on a video and writes outputs/<clip>_tracked.mp4 and outputs/<clip>_tracks.csv (columns: frame, time_s, track_id, cls, conf, x1, y1, x2, y2, w, h). TTC from box-width expansion, smoothed with a hand-rolled Kalman filter (state = [width, dw/dt]), working in explore.ipynb for a hand-picked lead car (no automatic lead-vehicle selection yet). Validated against real footage: filtered TTC dips line up with closing events noted by eye in the video. See NOTES.md for the Kalman tuning/validation details. Next: automatic lead-vehicle selection.

<!-- Update this section as the project progresses. -->
