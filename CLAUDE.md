Driving Risk Analyzer

A computer vision + sensor-fusion system that analyzes dashcam video (downloaded/public datasets, not self-recorded -- see "Environment") and synced GPS/IMU logs to measure driving risk: detect and track vehicles, identify the lead vehicle, estimate time-to-collision (TTC), distance, and headway, detect risky events (tailgating, hard braking, rapid closing, speeding, potholes), and produce a report. Portfolio project for ML internship applications.

Two deliverables matter equally:

An annotated recorded video: the system run offline on a downloaded drive recording, with boxes, TTC, and event overlays.
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
Live-capture code (live.py's camera mode, camera_scan.py, camera_probe.py, webcam_check.py) exists from an earlier live-demo plan and is frozen -- not deleted, but not being extended or tested further. All active work is on recorded video.
No new self-recorded driving footage going forward, and no phone GPS/accelerometer logging app (see NOTES.md, 2026-09-28) -- not enough time to go re-record drives with a synced sensor log. clip1_30s.mov (already recorded, Week 1-2) stays in use for detection/tracking/lead-selection/TTC, but has no GPS/accelerometer log and never will, so it can't be used for distance/speed/headway/speeding work.
Sensor/ego-motion data source: KITTI's oxts files (already downloaded) -- a real, frame-synced GPS+IMU log per sequence, plus calib (camera calibration) and LiDAR-derived 3D ground truth. This replaces the phone-log role for distance, ego speed, headway, and hard-braking, and is higher quality than a phone log would have been (properly calibrated, exactly synced, has ground truth to check against). Other video (YouTube dashcam clips, BDD100K) has no GPS/IMU, so it's usable for detection/tracking/TTC only, not for anything needing ego motion.
Stack

Python, Ultralytics YOLO (yolo26n.pt), built-in trackers (ByteTrack, BoT-SORT), OpenCV, NumPy/SciPy, pandas, matplotlib, PyTorch, scikit-learn, Streamlit for the post-drive report. osmnx/Overpass API for road speed-limit lookups (speeding check). Folium for the pothole map (not Google Maps -- see "Not doing"). Later: ONNX/OpenVINO export for faster CPU inference.

Pipeline
Detection (pretrained YOLO, vehicle classes only)
Tracking (consistent IDs across frames)
Lead-vehicle selection (the tracked car directly ahead, in my lane -- steady same-lane following only, see "Scope" below)
Measurement:
TTC from bounding-box expansion: TTC ≈ w / (dw/dt), smoothed with a Kalman filter. Needs no calibration.
Ego speed from GPS/IMU (KITTI oxts -- see "Environment").
Distance in meters (camera calibration + ground-plane geometry, later a metric depth model).
Time headway = distance / ego speed.
Situational awareness: TTC/distance also reported for every OTHER nearby tracked vehicle, not just the selected lead (same width-expansion method, just keyed per track id instead of one singleton) -- broadens what the report shows without pretending to understand cut-ins, merges, or cross-traffic; only the lead's headway feeds tailgating/rapid-closing scoring.
Events from patterns over time (sustained low headway, rapidly dropping TTC, hard braking from accelerometer)
Speeding: compare GPS speed against OpenStreetMap speed-limit tags for the matched road segment, flagging sustained excess (not single noisy samples). Report speed-limit tag coverage honestly -- many minor roads are untagged, skip them rather than guess.
Potholes: fine-tuned detector (separate from the vehicle YOLO) on a labeled pothole dataset, GPS-tagged and plotted on a Folium map. Suggested by a contact at NVIDIA -- see NOTES.md for the scoping discussion.
Scoring + Streamlit report (video, TTC plot, clickable event timeline, pothole map)

Scope: lead-vehicle detection targets steady same-lane following (highway-style closing/tailgating), not general urban scene understanding. It is not designed to identify or predict cross-traffic, merges, cars pulling out of driveways, or other vehicles' turning intent -- those need real trajectory/intent prediction, a much bigger research problem than this project takes on. This was learned concretely, not assumed: investigating real KITTI closing events for the TTC ground-truth check (see NOTES.md, 2026-09-28) turned up several that were a parked car passed closely or a car merging in from a driveway, not a lead vehicle -- LeadSelector correctly ignoring these is the system working as scoped, not a gap to silently paper over.

Stretch ML component: accident anticipation on DoTA/CCD (real crash and near-miss clips), with baselines first.

Evaluation plan
Distance accuracy on KITTI (LiDAR ground truth), comparing ground-plane geometry vs a depth model, reported by range.
Tracker comparison (ByteTrack vs BoT-SORT) on a standard tracking benchmark (HOTA/IDF1).
Event detection precision/recall on hand-labeled clips (BDD100K + KITTI).
Pothole detector precision/recall on its own held-out labeled set.
Roadmap
Week	Milestone
1	Detection + tracking on a recorded clip (track.py)
2	Lead-vehicle selection + TTC on recorded clips, first TTC plot
3	Distance, ego speed & headway, core events (tailgating, hard braking, rapid closing) -- built and validated on KITTI oxts/calib/LiDAR, not a phone log (see NOTES.md, 2026-09-28)
4	Speeding detection (GPS + OSM speed limits) -- reuses week 3's GPS source; only runs where a GPS log actually exists (KITTI), not on GPS-less downloaded video
5	Pothole detection: dataset + fine-tuned detector + GPS map (Folium)
6-8	KITTI/BDD100K evaluation, accident-anticipation model
9+	Annotated demo video polish, Streamlit report, polished README
Not doing

No React/FastAPI, no database, no cloud deployment, no Docker (for now), no custom detector from scratch (except the pothole detector, which is a small fine-tune, not from-scratch), no mobile app, no Google Maps integration (Folium instead -- no API key/billing account needed). Don't claim it's a safety system. No live/real-time capture going forward (not enough time to validate it in a real car) -- the Week 3 live code stays in the repo as a frozen record of finished work, not deleted, but nothing further is built on it. No cut-in/merge/cross-traffic intent prediction (see "Scope" above) -- a real trajectory-prediction problem, out of scope for the time available, not a gap to quietly try to patch with more heuristics. No more self-recorded driving footage and no phone GPS/accelerometer logging app going forward (see NOTES.md, 2026-09-28) -- the project now runs on downloaded/public video and datasets (KITTI, BDD100K, YouTube dashcam clips); ego-motion work (distance, speed, headway, speeding) uses KITTI's real oxts/calib data instead of a phone log.

Current status

Week 2 complete. track.py runs YOLO + tracking on a video and writes outputs/<clip>_tracked.mp4 and outputs/<clip>_tracks.csv (columns: frame, time_s, track_id, cls, conf, x1, y1, x2, y2, w, h). --conf default is 0.4 (not 0.5 -- see NOTES.md, 0.5 blocked ByteTrack's own occlusion-recovery thresholds and caused real gaps in the lead car's track). In explore.ipynb: automatic lead-vehicle selection (lane-band +-10% of frame width + area-based pick + self-detection filter + temporal hysteresis, 100% correct on clip1_30s, no manual ID picking) now feeds directly into TTC from box-width expansion, smoothed with a hand-rolled Kalman filter (state = [width, dw/dt]) -- the two pieces are wired together end to end and re-validated against real footage. Week 3 (live capture) was built -- live.py's camera mode, lead_ttc.py's LeadSelector/TTCKalman -- and tested against fake cameras, but is now frozen: not enough time to validate it in a real car, so scope has narrowed to recorded video only (see NOTES.md, 2026-09-28). The lead-selection and Kalman-TTC logic itself is unaffected by this and stays in active use for recorded/replayed video (live.py's file mode, imgsz 640 vs 1280, stride replay). A KITTI-based side-check of TTC against 3D-derived ground truth (kitti_gt_ttc.py, kitti_track.py, kitti_pipeline_compare.py, kitti_label_ttc_check.py) found two real, understood failure modes -- box-width TTC breaking down when the box is truncated by the frame edge near the closest approach, and occlusion causing tracker ID switches -- plus, importantly, that KITTI's closing-distance tracks are mostly NOT simple lead-vehicle scenarios (parked cars passed closely, merges), which led to the explicit "Scope" note above rather than more KITTI-hunting. This check was always a nice-to-have on top of the original evaluation plan, not a blocker. A further scope decision followed the same day: no more self-recorded driving or phone sensor logging going forward -- the project runs entirely on downloaded/public video and datasets from here on (see "Environment" and NOTES.md, 2026-09-28). Week 3 is relabeled accordingly: distance, ego speed, headway, and core events, built and validated against KITTI's real oxts/calib/LiDAR data (not a phone log, which no longer exists in this project).

<!-- Update this section as the project progresses. -->
