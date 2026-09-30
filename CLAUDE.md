Driving Risk Analyzer

A computer vision system that analyzes dashcam video to measure one thing well: detect and track vehicles, identify the lead vehicle, estimate time-to-collision (TTC), and flag a rapid-closing event. Portfolio project for ML internship applications.

One deliverable: an annotated video (boxes, tracking, lead highlight, TTC, rapid-closing flag) run on real footage -- KITTI and/or a downloaded dashcam clip. Scope cut 2026-09-29 from a much larger plan (GPS-based distance/headway/speeding, potholes, a formal quantitative-evaluation deliverable) -- see NOTES.md for why and what's frozen vs. active. This isn't a smaller ambition out of laziness: it's a deliberate call that the one thing a viewer actually watches (the annotated video) needed to be right before anything else mattered.

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
Live-capture code (live.py's camera mode, camera_scan.py, camera_probe.py, webcam_check.py) exists from an earlier live-demo plan and is frozen -- not deleted, but not being extended or tested further. live.py's file mode (run any recorded video through the same detection/tracking/lead/TTC pipeline, with boxes/lead/lane-band/TTC drawn) is very much active -- it's the reusable path for a non-KITTI demo clip.
No new self-recorded driving footage going forward, and no phone GPS/accelerometer logging app (see NOTES.md, 2026-09-28) -- the project runs on downloaded/public video (KITTI, YouTube-style dashcam clips). clip1_30s.mov (already recorded, Week 1-2) stays in use for detection/tracking/lead-selection/TTC.
KITTI (already downloaded, all 21 tracking sequences' label/oxts/calib, images for 0019/0009/0020) is used for: (a) ground-truth-backed checks of TTC and lead-selection (no GPS/oxts needed for this -- just 2D/3D box labels), and (b) rendering an annotated demo video. The oxts/calib-dependent work (distance, ego speed, speeding) was dropped 2026-09-29 -- see NOTES.md and "Not doing".

Stack
Python, Ultralytics YOLO (yolo26n.pt), built-in trackers (ByteTrack, BoT-SORT), OpenCV, NumPy/SciPy, pandas, matplotlib.

Pipeline
Detection (pretrained YOLO, vehicle classes only)
Tracking (consistent IDs across frames)
Lead-vehicle selection (the tracked car directly ahead, in my lane -- steady same-lane following only, see "Scope" below)
TTC from bounding-box expansion: TTC ≈ w / (dw/dt), smoothed with a Kalman filter. Needs no calibration -- works on any camera/video.
Situational awareness: TTC/distance-proxy also reported for every OTHER nearby tracked vehicle, not just the selected lead (same width-expansion method, just keyed per track id instead of one singleton) -- broadens what the report shows without pretending to understand cut-ins, merges, or cross-traffic.
Rapid-closing event: TTC sustained below a threshold for a minimum duration (not a single noisy frame).

Scope: lead-vehicle detection targets steady same-lane following (highway-style closing/tailgating), not general urban scene understanding. It is not designed to identify or predict cross-traffic, merges, cars pulling out of driveways, or other vehicles' turning intent -- those need real trajectory/intent prediction, a much bigger research problem than this project takes on. This was learned concretely, not assumed: investigating real KITTI closing events for the TTC ground-truth check (see NOTES.md, 2026-09-28) turned up several that were a parked car passed closely or a car merging in from a driveway, not a lead vehicle -- LeadSelector correctly ignoring these is the system working as scoped, not a gap to silently paper over.

Stretch ML component (deprioritized, not active): accident anticipation on DoTA/CCD (real crash and near-miss clips), with baselines first. Not explicitly ruled out in the 2026-09-29 scope cut, but it lived under the same "big evaluation surface" umbrella that got trimmed -- revisit only if there's real time left after the core demo is solid.

Evaluation plan: dropped as a deliverable, 2026-09-29 -- see NOTES.md. The KITTI ground-truth checks that remain (TTC accuracy, lead-selection agreement) are used to validate/tune the pipeline (e.g. the lane-band width retune, NOTES.md 2026-09-29), not reported as a formal accuracy table.

Roadmap
Week	Milestone
1	DONE. Detection + tracking on a recorded clip (track.py)
2	DONE. Lead-vehicle selection + TTC on recorded clips, first TTC plot
3	Current work: real lane-band bug found and fixed (retuned against ground truth, see NOTES.md, 2026-09-29), polishing the annotated-video demo(s) -- KITTI (0020) plus a user-supplied non-KITTI clip via live.py's file mode.

Not doing
No React/FastAPI, no database, no cloud deployment, no Docker (for now), no mobile app. Don't claim it's a safety system. No live/real-time capture going forward (not enough time to validate it in a real car) -- the Week 3 live code stays in the repo as a frozen record of finished work, not deleted, but nothing further is built on it. No cut-in/merge/cross-traffic intent prediction (see "Scope" above) -- a real trajectory-prediction problem, out of scope for the time available. No more self-recorded driving footage and no phone GPS/accelerometer logging app going forward (see NOTES.md, 2026-09-28).
As of 2026-09-29 (see NOTES.md): no GPS-dependent measurement (distance, headway, tailgating, hard braking, speeding), no pothole detection, no formal quantitative-evaluation deliverable. The code for the dropped GPS features (kitti_distance.py, kitti_ego_motion.py, kitti_speeding.py, kitti_event_eval.py) stays in the repo, frozen, not deleted -- real, completed, honestly-documented work, just out of the current narrower scope.

Current status
Weeks 1-2 done: detection/tracking/lead-selection/TTC work on real footage. A real lane-band bug was found (the lead-selection band, fixed at ±10% of frame width, let an adjacent-lane car outrank the true same-lane traffic on a wide multi-lane highway) and fixed by tuning against ground truth rather than guessing (band_half retuned 0.10 -> 0.07, kitti_band_tune.py) -- verified this fixed the exact complained-about window (77.4% -> 96.7% agreement with ground truth in the first 15s of KITTI 0020). Same day, made a deliberate scope cut: dropped everything GPS/calibration-dependent (distance, headway, tailgating, hard braking, speeding) and potholes, and dropped the formal quantitative-evaluation deliverable entirely, to put full effort behind the one thing that gets watched -- the annotated video. kitti_headway_events.py and kitti_annotate.py were trimmed to the surviving scope (lead + TTC + rapid-closing only); kitti_distance.py/kitti_ego_motion.py/kitti_speeding.py/kitti_event_eval.py are frozen, not deleted. Full history of everything built (including the frozen GPS/evaluation work, which was real and validated in its own right) is in NOTES.md. Next: process a user-supplied non-KITTI clip through live.py's file mode as a second demo, then final polish.

<!-- Update this section as the project progresses. -->
