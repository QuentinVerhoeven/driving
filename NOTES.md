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

## 2026-09-12 — First TTC attempt, and why raw box-width TTC is too noisy

Set up `explore.ipynb` and computed TTC from `outputs/clip1_30s_tracks.csv` for a hand-picked lead car (track ID 4, chosen by watching the annotated video and noting it held a stable ID through a stretch of closing-in around 16-30s). Implemented the plan from NOTES.md's 2026-09-10 entry: rolling mean over box width `w`, `dw_dt = np.gradient(w_smooth, time_s)`, `ttc = w_smooth / dw_dt` (only defined when `dw_dt > 0`). Tried smoothing windows of 5, 15, and 30 frames.

**Result: TTC was extremely jagged at every window size** — not just noisy-looking, but flipping between `nan` and huge (>1000s) values constantly. Diagnosed with two checks before touching the code:

1. **Track continuity:** ID 4 has 812 of 902 possible frames (90% coverage), mostly single 1-2 frame gaps plus one 26-frame (~0.87s) gap. Real, but too small and localized to explain jaggedness across the *entire* 30s clip.
2. **`dw_dt` behavior:** `dw_dt` flips sign 54 times in 30s (~1.8/sec) — far faster than a real car's closing/pulling-away pattern could physically change. 18% of frames have `|dw_dt| < 5 px/s` (i.e. near-zero, dominated by detector noise, not real motion). Only 58% of frames end up with a usable (non-nan) TTC. Raw TTC stats: mean 39.5s, std 149.7s, max 2372s before clipping.

**Root cause:** `TTC = w / dw_dt` is a ratio, and dividing by a near-zero, noisy denominator amplifies whatever noise is left after smoothing `w` — this is separate from (and dominates over) the frame-gap issue. It's especially bad during steady-following stretches, which is most of a normal drive: the *true* `dw_dt` is close to zero there, so measurement noise easily flips its sign frame to frame. A bigger rolling window on `w` doesn't fix this because the derivative of even a smooth signal is still sensitive near zero — you have to smooth the *rate estimate itself*, with memory across frames, not just the position.

**Decision:** this confirms the plan from the 2026-09-10 entry was right to specify a Kalman filter rather than a raw derivative. Next step: implement a constant-velocity Kalman filter over box width (state = `[width, dw/dt]`, width as the only measurement) by hand, no `filterpy` dependency, so every line stays explainable in an interview. Expect this to damp the sign-flipping because the filter's rate estimate has inertia — a single noisy frame can't flip it the way a raw frame-to-frame derivative can.

**Also confirmed (not yet fixed):** `.rolling(window, center=True)` uses future frames, which a live system can't do — noted already in the 2026-09-10 plan as a reason a Kalman filter (which only uses past + current measurements) is needed for the live version anyway, not just as a noise fix for the offline version.

---

## 2026-09-24 — Hand-rolled Kalman filter for TTC, built step by step

Implemented a constant-velocity Kalman filter over box width in `explore.ipynb`, added incrementally (state -> predict -> update -> full loop -> retuning) so every line could be explained rather than pasted in as a block. No `filterpy` dependency — done by hand with plain NumPy, per the 2026-09-12 decision.

**Model:**
- State `x = [width, dw/dt]`, only width is measured (`H = [1, 0]`).
- `F = [[1, dt], [0, 1]]` — constant-velocity assumption: predicted width = current width + rate * dt, rate assumed to persist.
- `Q` (process noise) scaled by `dt` on both state components — the longer the gap since the last measurement (e.g. across a dropped-frame gap), the more the true rate could have drifted, so uncertainty should grow more over bigger gaps. This falls out naturally from scaling by `dt` rather than needing special-case gap handling.
- `R` (measurement noise) is a fixed guess at per-frame box-width jitter (px²).
- Each step: predict (`x = Fx`, `P = FPF^T + Q`), then update using the innovation `y = measurement - prediction`, Kalman gain `K` computed from `P` and `R`, `x = x + Ky`, `P` shrinks accordingly.

**First result (q_width=1, q_rate=50, R=25):** `dw_dt` sign flips dropped from 54 (raw derivative baseline, see 2026-09-12 entry) to 15 over the same 30s clip; fraction of frames with usable (non-nan) TTC stayed ~58% (expected — that ratio reflects how much of the real drive was spent closing vs. not, which filtering shouldn't change, only the noise-driven flips should).

**TTC plot was calmer but still visibly spiky.** Two distinct explanations, not one:
1. **Tuning was too loose.** `q_rate=50` relative to `R=25` still let the rate estimate react quickly to individual noisy measurements. Retuned to `q_rate=5, R=100` (trust momentum more, trust each width reading less) and sign flips dropped further, 15 -> 10. An even tighter test (`q_width=0.5, q_rate=1, R=200`) got flips down to 4.
2. **Some spikiness is inherent to the TTC formula, not filter error.** During genuine steady-following stretches, the *true* `dw/dt` really is near zero (not just noisy) — and `TTC = width/rate` is mathematically supposed to blow up toward infinity there. A perfectly-estimated near-zero rate still produces a TTC that swings between a real low value and the clip ceiling as it crosses zero. This is the metric working as intended, not a bug — worth remembering as a real limitation of box-expansion TTC to explain in interviews (it's meaningful during active closing events, not as a continuously stable "distance-like" signal).

**Tuning decision: keep `q_width=1, q_rate=50, R=25` (the original/"current" values), not the tighter retune.** Plotting "current" vs. "tighter" (`q_rate=5, R=100`) side by side showed why: during real closing events, the tighter filter's TTC dips only reached ~8-9.4s, while "current"'s dips reached the true ~4-8.5s range. Lowering `q_rate` makes the filter distrust that the rate can change quickly, so it lags and *understates* real closing events — for a collision-relevant metric, quietly blunting a real dip is worse than some residual jaggedness. Lesson: don't just chase "smoother-looking plot" when tuning a Kalman filter for a safety-adjacent signal — check whether it's still capturing the true depth/timing of real events, not just reducing noise. Some jaggedness in the final signal is expected and partly inherent to the TTC formula (see above), not automatically a sign of bad tuning.

**Validation against real footage:** grouped consecutive Kalman-TTC frames below an 8s threshold into discrete "events" (start/end time, frame count) instead of eyeballing the plot. Result on `ID 4`, current tuning:

| start_s | end_s | n_frames |
|---|---|---|
| 4.14 | 4.94 | 18 |
| 14.95 | 15.35 | 13 |
| 18.72 | 19.35 | 16 |
| 23.29 | 25.16 | 57 |

The longest, clearest event (23.3-25.2s, 57 frames — far more sustained than the others) falls inside the 16-30s closing stretch noted by eye while watching `outputs/clip1_30s_tracked.mp4` in VLC before this session (see step-1 note). Two shorter dips (14.95s, 18.72s) also fall in that window. One dip (4.14s) is outside the noted window — likely a real event that just wasn't written down, not investigated further. Overall: the filtered TTC is finding events at real, plausible timestamps rather than noise-driven spikes, which is the actual validation this stage needed (not just "does the plot look smoother").

**Stage complete:** hand-rolled Kalman-filtered TTC from box-width expansion works on real footage for a single hand-picked lead car, with parameters and reasoning documented above. Next stage (per CLAUDE.md roadmap): automatic lead-vehicle selection, so TTC doesn't require manually picking a track ID.

---

## 2026-09-24 (cont.) — Automatic lead-vehicle selection, first pass

Started the next roadmap item: automatic lead-vehicle selection (previously TTC used a hand-picked track ID). Heuristic: "in my lane" = box horizontal center within a band around frame-center (no real lane-line detection); among in-band boxes per frame, pick the largest by area (nearest).

**Band choice check:** histogram of box center-x across all detections/frames showed a clear peak near frame-center, supporting a center-band proxy for "in lane" on this clip. Set `BAND_HALF_WIDTH = 0.15 * FRAME_WIDTH` (~30% of frame width total) as a first guess.

**Result on `clip1_30s`:** selects a candidate in 896/901 frames. Track ID 4 (the hand-picked lead car) dominates at 698 frames (78%). Second-most is ID 94 at 107 frames, taking over cleanly and holding continuously from 26.06s through the end of the clip.

**Initial (wrong) guess vs. actual cause:** first assumed ID 94 was the same physical lead car re-acquired under a new ID after a tracking gap. Checked the actual video instead of just the numbers: **ID 94 is a self-detection** — the dashcam's own car (hood/mirror/dash) being misclassified as a vehicle by YOLO, not a real car ahead. It sat centered and large enough to win the "biggest box in the lane band" heuristic for the whole last ~4s of the clip. Lesson: a plausible-looking pattern in the numbers (clean single handoff, holds till the end) is not the same as confirming ground truth against the actual footage — should have looked at the video before writing down a causal story.

**Also seen:** several other IDs (30, 32, 34, 53, 55, 56) get picked for a handful of scattered frames each, concentrated in the busier 8-20s stretch — likely other real vehicles briefly crossing into the center band that momentarily have a larger box than the true lead car. Separate problem from the self-detection one; still needs addressing.

**Self-detection filter, implemented:** confirmed via the actual video (not just the numbers) that ID 94 is a self-detection of the dashcam car's own hood/dash. Measured its actual box geometry to set real thresholds rather than guessing: ID 94 boxes average 4.97:1 width:height (vs. 1.26:1 for the real lead car, ID 4) and sit with bottom edge at 99.8% of frame height (vs. 55.5% for ID 4) — essentially glued to the bottom of the frame. Added a filter excluding boxes with `aspect_ratio > 2.5` or `y2 > 0.95 * frame_height`, applied on top of the lane-band filter. Deliberately filtered on shape/position, not the track ID itself, since IDs aren't stable across runs/clips.

**Result after the fix:** ID 4's share rose from 698 -> 787 of 892 candidate frames (88%); ID 94 dropped out of the candidates entirely. Remaining scattered picks (IDs 32, 89, 30, 55, etc., a few dozen frames each) are the separate, still-unaddressed flicker problem from other real vehicles briefly crossing into the lane band.

**Temporal stickiness (hysteresis), implemented:** once a car is "the lead," keep it unless (a) missing from candidates for `LOST_STREAK=5` consecutive frames, or (b) a different car has been the single biggest candidate for `SWITCH_STREAK=10` consecutive frames (a sustained overtake, not a 1-2 frame blip). Frame-by-frame loop, not vectorizable (decision at frame *i* depends on running state). Result: unique picked IDs dropped to just 4 (from 8 with the raw per-frame rule), and remaining non-lead picks got *longer*, not shorter (e.g. ID 32: 41 -> 63 frames) -- expected, since hysteresis makes whichever ID currently holds the "lead" title sticky, including a wrongly-adopted one.

**New failure mode surfaced by hysteresis: ID 89.** Watched the video -- ID 89 is a real car pulling up alongside while the dashcam car is stopped, not the lead car. It got adopted as lead for 25 sustained frames because, while alongside, its box is both large (close) AND ends up within the 15%-of-center band -- a 2D box position alone can't distinguish "ahead of me" from "beside me but currently centered in this frame," especially when stopped (no forward motion to disambiguate). Confirmed in the numbers: ID 89's box starts at frame edge (`x1=9.3`) and visibly slides across ~2s to center as it pulls alongside -- the giveaway is in its *trajectory*, which a single-frame heuristic doesn't see.

**Tried and rejected:** switching the primary selection rule from "biggest box in band" to "most-central box" (dropping the band filter) made things worse -- e.g. ID 30 (a small, distant, real car) jumped to 155 picks. A small box's center position is much noisier frame-to-frame than a large box's, so pure centering without a size prior lets tiny far-away detections win by chance. Area-based selection was already doing useful noise-rejection that pure centering threw away.

**Actual fix: tighten the band, keep area-based selection.** Tested `band_half_width` at 15%/10%/8%/6% of frame width (all with the same area-based-within-band selection + hysteresis). ID 89 never gets closer than ~260px from center at its closest (vs. ID 4's ~46-160px), so a 10%-of-frame-width band (384px half-width) excludes it entirely while still covering the real lead car's horizontal jitter. **Decision: `BAND_HALF_WIDTH = 0.10 * FRAME_WIDTH`.** Result: ID 4 share actually improved slightly (86.8% -> ~88%+) and ID 89 dropped out completely. Remaining minor ID 30/32 picks are a handful of frames each, likely the tracker briefly losing ID 4 during one of its known gaps (2026-09-12 entry) and temporarily falling back to a real, more distant in-lane car -- reasonable behavior, not a false positive.

**Known remaining limitation, accepted for now:** a stopped-car-with-another-pulling-alongside scenario can still fool this heuristic if the alongside car gets close enough to fall inside even a 10% band for 10+ sustained frames. True lane-line/road-geometry detection (out of scope for this stage) or motion-history features (was this box sliding in from a frame edge recently?) would fix this properly. Noting it now rather than chasing it further -- matches "build one layer at a time."

---

## 2026-09-24 (cont.) — Root cause of the ID 30/32 fallback switches: `--conf` was blocking ByteTrack's own occlusion recovery

The remaining ID 30/32 lead-selection fallbacks (from the entry above) traced back to the real lead car (ID 4) briefly not being detected at all for ~0.2-0.9s stretches a few times per clip (29 gaps total in the original `clip1_30s` run, one of them 26 frames). Investigated whether fine-tuning the detector would help, but found a much cheaper root cause first: **`track.py` passes `--conf` straight into `model.track(conf=...)`, and Ultralytics applies that threshold to filter raw detections BEFORE they reach ByteTrack.** ByteTrack is explicitly designed with two thresholds for this exact situation — `track_high_thresh: 0.25` for confident matches and `track_low_thresh: 0.1` for a second-stage pass that recovers occluded/awkward-angle detections without letting them spawn new tracks (guarded by `new_track_thresh: 0.25`). Raising `--conf` to 0.5 (2026-09-10 decision, to fix the original ID-flicker problem) discarded everything below 0.5 upstream, so ByteTrack's own 0.1-0.25 recovery band never received any detections to work with.

**Tested `--conf` at 0.1, 0.25, 0.4, and 0.5 (current) on `clip1_30s`, both on raw tracker output and on the actual downstream lead-selection pipeline (band+area+hysteresis from the entries above):**

| conf | unique track IDs (raw) | tracks <10 frames | ID4 frames/gaps | lead-selection: ID4 share |
|---|---|---|---|---|
| 0.5 (old default) | 51 | 17 | 812/902, 29 gaps (max 26) | 692/781 (88.6%), 2 fallback IDs |
| 0.4 | 75 | 30 | 875/902, 15 gaps (max 5) | 793/793 (**100%**) |
| 0.25 | 121 | 65 | 895/902 | 813/813 (**100%**) |
| 0.1 | 118 | 56 | 897/902, 3 gaps (max 4) | 815/819 (99.5%), 1 tiny blip |

Key insight: raw "unique track ID count" is a misleading metric for this specific purpose. It roughly triples at low `conf` (more background noise spawning brief spurious tracks), but that noise barely affects lead-selection because the lane-band + self-detection + area filters already reject almost all of it -- what actually matters is whether the *lead car itself* stays continuously tracked, and lower conf directly fixes that.

**Decision: set `--conf` default to 0.4`, not 0.1 or 0.25.** It achieves the same 100% lead-selection stability as the more permissive settings, while keeping meaningfully less general-purpose noise (75 vs. ~120 unique IDs) -- relevant if this pipeline later needs to track/reason about vehicles other than just the lead car (e.g. adjacent-lane events). Regenerated the official `outputs/clip1_30s_tracked.mp4`/`_tracks.csv` with the new default.

**Process note:** before touching the model or tracker config, actually read how `--conf` flows through `model.track()` rather than assuming "raise conf = less flicker" was still the right lever -- the original 2026-09-10 fix was correct for the problem it solved (noisy low-confidence detections causing ID churn) but had an unintended side effect (blocking legitimate occlusion recovery) that only showed up once a downstream consumer (lead selection) made track continuity matter in a new way.

**Re-validated end-to-end against the new `conf=0.4` CSV:** re-ran `explore.ipynb`'s lead-selection pipeline (band=10%, area-based, hysteresis, self-detection filter) on the regenerated data. Result: **793/793 frames (100%) picked ID 4** as lead, zero fallback switches to other IDs -- matches the table above exactly. Automatic lead-vehicle selection is now solid on this clip with no manual ID picking and no ID-switch artifacts.

**Stage complete:** automatic lead-vehicle selection (lane-band + self-detection filter + hysteresis) + the `--conf` fix together give a clean, continuous lead-car track with no manual intervention. Combined with the earlier Kalman-filtered TTC work, this finishes the roadmap's "Lead-vehicle selection + TTC on recorded clips" milestone. Next: wire automatic selection directly into the Kalman-TTC computation (currently still two separate pieces in the notebook), then decide on Week 3's first live version.

---

## 2026-09-24 (cont.) — Wiring automatic lead-selection into the Kalman-TTC pipeline

Last loose end from the two previous entries: automatic lead-vehicle selection and the Kalman-filtered TTC were built and validated separately, each still reading from a hardcoded `track_id == 4`. Wired them together in `explore.ipynb`, one step at a time:

**Step 1 — build a width series from the auto-selected lead, not a hardcoded ID.** Merged `lead_hyst` (the hysteresis output: one `track_id` per frame) against `all_boxes` on `(frame, track_id)` to pull back `w`/`h`. Sanity check: on every frame the two series have in common, the auto-selected width matches the old hardcoded-ID width exactly (max diff 0.0) — confirms the merge is just relabeling the same data, not introducing new logic. The auto series has fewer rows (793 vs. 875) because hysteresis sometimes has zero in-band candidates for a frame; the hardcoded filter never cared about "in-band," just "is this ID present."

**Step 2 — checked whether the auto series needs special handling before feeding the Kalman filter.** Two ways this could go wrong that don't exist with a hardcoded ID: (a) a genuine ID switch would make width jump for a fake reason, not a real distance change, and (b) missing frames create bigger gaps between real measurements than the usual 1/30s. On this clip, (a) doesn't happen — auto-selection was 100% ID 4, already confirmed in the previous entry. For (b): the largest gap is 83 frames (2.77s), around t=4.1-6.9s. Checked the raw boxes in that window — ID 4 was detected the whole time, just drifted left of the 10% lane band (cx down to ~1430px vs. the band edge at 1536px, likely the road curving) before coming back in-band. Not a detection failure. Concluded this needs **no new code**: the Kalman `Q` matrix is already scaled by `dt` specifically so a longer gap between measurements makes the filter trust its own prediction less — that was designed in from the start (see the 2026-09-24 Kalman-build entry, step 2) and this is exactly the situation it was for, just not one we'd actually hit yet.

**Step 3 — reran the Kalman filter on the auto-selected series (same tuning: q_width=1, q_rate=50, R=25) and re-validated against real footage.** Grouped consecutive frames with Kalman TTC < 8s into events, same method as the original hand-picked-ID validation:

| start_s | end_s | n_frames |
|---|---|---|
| 4.14 | 4.14 | 1 |
| 14.95 | 15.38 | 14 |
| 17.35 | 17.85 | 11 |
| 18.72 | 18.85 | 5 |
| 18.99 | 19.49 | 16 |
| 23.29 | 25.16 | 57 |

The longest, clearest event (23.3-25.2s, 57 frames) is essentially unchanged from the hand-picked-ID version — same start, same end, same frame count. The 14.95s and 18.7-19.5s events also still show up, just split slightly differently (18.7-19.5s became two events instead of one, and a new short one appears at 17.35s) because a handful of frames in those stretches got excluded by the lane-band filter, not because the underlying physical event changed. The 4.14s event shrank from 18 frames to 1 for the same reason — most of that frame range fell outside the auto-selection's in-band criterion. None of this is a regression: it's the expected cost of layering "is this box actually in my lane" on top of "is this box present," and the one event that matters most for validation (the long, sustained 23-25s stretch) is untouched.

**Stage complete:** automatic lead-vehicle selection now feeds the Kalman-filtered TTC directly — no more manually picking a track ID anywhere in the pipeline. This closes out Week 2 (`Lead-vehicle selection + TTC on recorded clips` per the CLAUDE.md roadmap) end to end. Next: Week 3, first live version (webcam on windshield, live boxes + TTC + audio alert).

**Side note:** needed to actually execute `explore.ipynb` headlessly to verify these cells (rather than just reading the code and assuming it runs) — `jupyter nbconvert`/`jupyter run` weren't set up for this. Added `nbclient`/`nbformat` as dev dependencies to run the notebook from a script and write outputs back in place.

---

## 2026-09-24 (cont.) — Scope correction: speeding + potholes belong on the roadmap after all

Caught a gap between what was actually agreed and what `CLAUDE.md` says. The original planning chat (before the 2026-09-10 "Initial planning" session logged above) explicitly agreed to do both the core CV pipeline *and* a sensor-fusion track: speeding (GPS speed vs. OpenStreetMap speed-limit tags) and potholes (a fine-tuned detector, GPS-tagged and mapped with Folium, explicitly *not* Google Maps, to avoid needing a billing account for a portfolio project). Somewhere between that conversation and writing `CLAUDE.md`, both features silently dropped out of the roadmap — no entry anywhere recorded an actual decision to cut them, they just didn't make it into the doc.

Re-decided with two more inputs than the original chat had:
1. A contact at NVIDIA specifically suggested the pothole detection feature.
2. Weighing cost: speeding is nearly free (GPS logging is already week 4 work for ego-speed/headway; the speeding check itself is a lookup + subtraction), while potholes is a second, real detection subsystem — its own dataset, its own fine-tuned model, its own precision/recall evaluation — not an afternoon add-on the way it might have sounded in the original chat.

**Decision: add both back into the roadmap for real**, not as a vague stretch goal:
- Week 5: speeding detection (GPS + OSM speed limits), right after week 4's GPS/accelerometer logging work, since it reuses that data directly.
- Week 6: pothole detection as its own week — dataset, fine-tuned detector, GPS map — treated with the same evaluation rigor as the rest of the project (precision/recall on a held-out set), not just a cool-looking map of pins.

Both pushed the KITTI/BDD100K evaluation + accident-anticipation stretch work back by two weeks (now 7-9) and the final polish week to 10+. Updated `CLAUDE.md`'s description, pipeline, stack (added osmnx/Overpass, Folium), evaluation plan, and roadmap accordingly. The "no custom detector from scratch" line in "Not doing" got a caveat: the pothole detector is a small fine-tune of an existing model, not training from zero, so it doesn't actually contradict that rule.

**Lesson:** a real planning conversation happened, real decisions got made in it ("maybe we could do both"), and none of it got written down anywhere durable — it only existed in a chat transcript I could easily have lost track of. `CLAUDE.md` is supposed to be the source of truth for what the plan actually is; if a decision doesn't make it in there (or here), it might as well not have happened.

---

## 2026-09-24 (cont.) — Week 3 setup: native Windows environment, and the phone is the live camera

**Native Windows setup (WSL can't reach a webcam).** Nothing was installed on the Windows side, so: `uv` via the official PowerShell installer (lets uv manage Python 3.12 itself, same as in WSL), git via `winget install --id Git.Git`, then a fresh `git clone` of the GitHub repo into a normal Windows folder + `uv sync`. Chose a separate clone synced through git over running from `\\wsl$\...` paths: the WSL copy stays the main dev environment, the Windows copy exists only to run live capture. The `pytorch-cpu` index in `pyproject.toml` is cross-platform, so no config changes were needed. Problem hit: after `winget install`, `git` was still "not recognized" in new PowerShell windows (Windows shells can inherit a stale PATH from their parent process); fixed by verifying the install path and restarting so the PATH refreshed.

**Design decision: the phone is the live camera too, not a separate webcam.** The roadmap originally said "webcam on windshield." Better: the phone captures (streamed to the laptop via a phone-as-webcam app, so it appears as an ordinary camera index in OpenCV), the laptop does all compute. Reasons: better camera than any laptop/USB webcam, easy to mount, and it's the same device that already logs GPS/accelerometer in Weeks 4-5, so live and recorded footage share one sensor. Open risks to test rather than assume: (1) the stream will likely be 720p/1080p, not the 4K clip the detection settings (`imgsz`, `--conf`, lane band) were tuned on, and earlier debugging showed low resolution makes distant cars flicker, so tuning may need rechecking on streamed footage; (2) latency/frame-rate over USB vs Wi-Fi; (3) whether the phone can stream video and log GPS at the same time.

**First live-step script:** `webcam_check.py` opens a camera with OpenCV (DirectShow backend on Windows) and shows raw FPS, with no YOLO. Purpose: separate "can we get frames and how fast" from "how slow is the model," so if the full live loop is slow we know which piece to blame. Works with any device Windows exposes as a camera (`--camera N`), so it's the same test for a laptop webcam or the phone stream.

---

## 2026-09-24 (cont.) — Week 3: lead selection + Kalman TTC rewritten for streaming (`lead_ttc.py`)

**Scope change:** the audio alert is dropped from the live demo. Week 3 now ends at live boxes + a TTC overlay. (CLAUDE.md still mentions the alert in three places and needs updating.)

**Why a rewrite.** In `explore.ipynb` everything ran in batch over the whole CSV (group by frame, look at any frame at any time). A live camera gives one frame at a time and no future, so anything that depends on earlier frames has to be stored between calls. Moved the logic into `lead_ttc.py` as two stateful classes:
- `LeadSelector.update(boxes)`: the lane band, area pick and self-detection filter are stateless. The hysteresis (current lead, challenger + streak, missing streak) is the state the notebook kept in loop variables and is now on `self`.
- `TTCKalman.update(width, t, track_id)`: keeps `x = [width, rate]`, `P` and the last timestamp. `Q` scales with `dt` (as in the notebook), so a longer gap between measurements makes it trust its own prediction less.

**Verification.** Streamed the recorded `clip1_30s` tracks through the new classes one frame at a time and compared against the notebook's batch output (by running the notebook cells themselves): same 793 frames picked, same track IDs (100% ID 4), width/rate/TTC identical to ~1e-14. Problem hit on the way: a 0.1 px width difference. Cause was only CSV rounding (`w` is stored to 0.1 px, and recomputing `x2 - x1` from separately rounded coordinates differs). Not a logic bug, and it doesn't arise live because boxes come straight from YOLO.

**Issue found and fixed: the Kalman filter wasn't reset when the lead changed.** The state describes one physical car. If the lead switches from a 200 px car to a closer 300 px car, the filter reads the 100 px jump as a huge closing speed and TTC collapses, a false near-collision. It never showed up on `clip1_30s` (lead is always ID 4), but lane changes in live traffic would trigger it. Same problem when the road empties and a new car appears later.
- Fix: `TTCKalman` remembers which `track_id` its state belongs to and re-initializes (`x = [new width, 0]`, large `P`) when that changes. TTC is `nan` for the first frame after a reset, which I prefer to a wrong number.
- Considered and rejected: resetting after a long time gap. On this clip the lead leaves the lane band for 2.77 s and returns as the same car (ID 4), and the filter handles it correctly via `Q * dt`. A gap-based reset would wipe good state and change the validated results. An ID-change rule handles both cases without that cost.
- Synthetic test: 200 px car for 1 s, then a 300 px car with a new ID. Without the reset, TTC dips to ~2.3 s (false alarm). With the reset, no TTC is reported after the switch. Re-ran the `clip1_30s` comparison afterwards: still identical, since there are no ID changes.
- Remaining downside: if the tracker briefly loses the lead and gives the *same* physical car a new ID, we reset unnecessarily and lose a few frames of smoothing. It fails safe (no estimate rather than a wrong one).

**Known limitation:** `LeadSelector` keeps `current_lead` set while the lane band is empty and only switches when a candidate next appears (after `LOST_STREAK` missed frames). The selection is correct; the stale-state problem was on the Kalman side and is covered by the ID-change reset.

**Next:** live loop that reads frames from a video file (swap to the phone camera index later), then retuning at 720p/1080p.

---

## 2026-09-24 (cont.) — Week 3, live loop step 1: per-frame YOLO + tracking + FPS (`live.py`)

**What it is.** The bare live loop, with no lead selection or TTC yet: open a source, read one frame, run YOLO + tracker on it, draw boxes, measure FPS. Building it one layer at a time so each piece can be checked on its own.

**Design decisions and why:**
- **One code path for file and camera.** `cv2.VideoCapture(source)` accepts a path or an integer camera index, so `--source data/raw/clip1_30s.mov` and `--source 0` run the same loop. Everything can be built and tested in WSL on recorded clips now, and the phone is a one-argument swap once the USB cable is available.
- **`model.track(frame, persist=True)` per frame.** `track.py` handed Ultralytics the whole video with `stream=True` and it ran the loop. Live, we own the loop and call `track()` once per frame. `persist=True` tells it to keep ByteTrack's state between calls. Without it, the tracker restarts every call and IDs change every frame. (`track()` returns a list, one `Results` per image, hence `[0]`.)
- **Cross-platform camera backend.** `cv2.CAP_DSHOW` only exists as a sensible choice on Windows, so it is only used when `sys.platform == "win32"`. Same file works in WSL (files) and native Windows (camera).
- **Timing YOLO separately from the whole loop.** Same idea as `webcam_check.py`: if the total is slow, the two numbers show whether the model or the rest (decode, drawing, writing) is the bottleneck.
- **`--show`, `--out`, `--max-frames`.** WSL may not have a display, so a window is optional and the annotated video can be saved instead. `--max-frames` allows short benchmarks.

**First measurement** (60 frames of the 4K `clip1_30s`, `imgsz=1280`, this machine's CPU, WSL): YOLO + tracker ~111 ms/frame after warm-up (first ~20 frames were ~230 ms, model warm-up), whole loop ~5.6 fps (~180 ms/frame). The ~70 ms gap between them is the rest of the loop: decoding 4K, `r.plot()`, and writing 4K video.

**How to read these numbers:**
- They are not the live demo's numbers. The phone stream will be 720p/1080p, so decode/draw/write get much cheaper. YOLO's cost depends on `imgsz`, not source resolution, so it should stay about the same.
- A recorded clip is processed as fast as the CPU allows, so "fps" here is throughput, not real-time. Live, the camera delivers ~30 fps and we would process only ~5-9 of them.
- **Consequence for TTC:** the time between *processed* frames will be ~0.1-0.2 s, not 1/30 s. The Kalman filter must use real timestamps (video time for files, wall-clock time for a camera), not assume a fixed frame rate. To handle when wiring TTC in.

**Next:** step 2, run `LeadSelector` and `TTCKalman` inside this loop and draw the lead box and TTC on the frame.

---

## 2026-09-24 (cont.) — Week 3, live loop step 2: lead selection running per frame

**What changed in `live.py`.** Each frame's YOLO output now goes through `LeadSelector`, and the chosen lead is drawn as a thick green `LEAD id N` box, with the two lane-band lines in white.

**Pieces and why:**
- **`to_boxes(r)`:** Ultralytics returns tensors (`r.boxes.id`, `r.boxes.xyxy`); `LeadSelector` wants a list of `Box` objects. `r.boxes.id` is `None` when nothing is tracked, which becomes an empty list, so the selector counts a missing frame instead of crashing.
- **One `LeadSelector` created before the loop.** Its state (current lead, challenger streak, missing streak) has to persist across frames. Creating it inside the loop would wipe its memory each frame and silently turn hysteresis off, which is the whole reason for the streaming refactor. It takes the frame size from the source, so nothing is hardcoded to 4K.
- **Drawing the lane band.** Lets you answer "why wasn't this car picked?" by eye. A debugging aid, not part of the algorithm.

**Result** (first 300 frames of `clip1_30s`, frame-by-frame tracking): a single car, ID 4, was lead for 217/300 frames. Checked on actual frames:
- Frame 60: LEAD box on the grey car in our lane. A larger white car to the right is outside the band and correctly ignored (this is why the lane band is applied before the area pick).
- Frame 200: no lead drawn. ID 4 has drifted just left of the band while the road curves. This is the same ~2.8 s stretch found in the notebook (see the wiring-step-2 note above), so the live version matches the batch behavior. Not a bug; it is a limitation of using a fixed band as a lane proxy on curves.

**Observation:** the same car got the same ID (4) when tracked frame by frame with `persist=True` as in the whole-video run of `track.py`, which is a good sign that per-frame tracking behaves like streaming mode.

**Known cosmetic issue:** Ultralytics' default label text is large on a 4K frame and labels overlap. Irrelevant at phone resolution, left alone.

**Next:** step 3, run `TTCKalman` on the lead's width in the loop using real timestamps, and overlay the TTC number.

---

## 2026-09-24 (cont.) — Week 3, live loop step 3: Kalman TTC in the loop, with real timestamps

**What changed in `live.py`.** The lead's box width now feeds `TTCKalman` every frame, the TTC is shown as a fixed on-screen readout, and `--log` saves `frame, t, track_id, width, ttc` per frame to a CSV.

**Design decisions and why:**
- **Timestamps must be real.** The Kalman filter uses `dt` between measurements (in both the motion model and the process noise `Q * dt`), so `dt` has to be the true elapsed time. Live we only process ~5 of the camera's 30 fps, so assuming 1/30 s would make every velocity estimate ~6x too large. For a *file*, `t = frame_idx / src_fps` (position in the video, independent of how fast this laptop runs, and identical to the notebook's `time_s`). For a *camera*, `t = time.time() - start` (wall clock: frames arrive when they arrive).
- **Kalman created once, before the loop**, same reason as the selector: its `[width, rate]` state must persist. On frames with no lead we skip the Kalman step; `Q * dt` absorbs the gap.
- **Display:** `TTC 4.2 s`; `TTC --` when TTC is `nan` (box not growing, or the filter just reset); `TTC >10 s` above 10 (matching the 10 s clip on the notebook plots).

**Validation against the notebook** (first 300 frames of `clip1_30s`; ran the notebook's batch code and compared): same 217 lead frames, widths identical, and the same frames have/lack a TTC. TTC values differ by up to 7.8 s, but only at frames 59-64 where TTC is ~300 s: there the rate is almost zero and `TTC = width / rate` amplifies any tiny difference. Where TTC actually matters (batch TTC < 10 s, 18 frames), the max difference is 0.007 s (0.08%). Cause: the batch run uses the CSV's `time_s` rounded to milliseconds, live uses exact `frame/30`. An earlier test feeding the filter those rounded times matched to 1e-14, so timestamps are the only difference.
- **Lesson:** TTC is a ratio and blows up as the denominator approaches zero, so compare it only in the range where it is meaningful (small TTC). Comparing the raw max difference was misleading.

**Problem hit: unreadable overlay.** First version drew the TTC text above the lead box. Over a small, distant box it collided with Ultralytics' own label text and could not be read. Moved it to a fixed large readout in the top-left under the FPS counter, which is better for the demo anyway (fixed place to look).

**Also visible in the output frames:** Ultralytics detects our own hood as a car (`id:7 car 0.41`); the self-detection filter correctly ignores it.

**Open risk for the retuning step:** `R = 25 px^2` and `q_rate = 50 px/s` are in pixel units tuned on 4K widths. TTC itself is a ratio and does not depend on resolution, but the noise parameters do: at 720p widths are ~1/5 as large, so those values will need scaling or retuning.

**Also to watch with a real camera:** OpenCV camera buffers can hand back stale frames when we process slower than the camera delivers, adding latency. May need to shrink the buffer or drop frames.

**Next:** step 4, retest at 720p/1080p (resize the clip) to see whether `imgsz`, `--conf`, the lane band and the Kalman noise values hold up, since the phone stream won't be 4K.

---

## 2026-09-25 — Week 3: getting the phone stream into OpenCV (debugging log)

**Symptom.** With the phone streamed through Iriun, `webcam_check.py` showed a black picture at ~1 fps, 640x480, while the Windows Camera app and Iriun's own client showed the phone's video perfectly. So the phone and driver were fine and the problem was between OpenCV and the virtual camera.

**Investigation (each step narrowed it down):**
1. **`camera_scan.py`:** tries indices 0-5 x DirectShow/Media Foundation, reads for 2 s, reports resolution, FPS and average brightness (0 = black). Result: three cameras. Two gave real video (0 and 2), one gave black at ~1 fps (1). I first *assumed* the healthy 720p one was the phone. Wrong: opening the saved frames showed indices 0 and 2 were both the laptop webcam. **Lesson: look at the picture, don't infer identity from resolution/FPS.**
2. **`camera_probe.py --index 1`:** tried every backend x resolution x pixel format. Every combination was black, and the camera accepted any resolution even 1080p. A real camera refuses sizes it can't do, so this was a virtual camera accepting requests but getting no frames.
3. **Device list** (`Get-PnpDevice -Class Camera`) showed both *Camo* and *Iriun Webcam* registered. I guessed index 1 was Camo's idle virtual camera. Wrong again: `pygrabber` (run with `uv run --with`, so it isn't a project dependency) printed the DirectShow device names in OpenCV's index order: `0 Integrated Webcam, 1 Iriun Webcam, 2 Camo`. So index 1 *was* Iriun.
4. **Fix:** closed other apps that could hold the camera (Windows Camera app), restarted Iriun (phone app first, then Windows client), rechecked. `camera_probe.py --index 1` then gave real pictures in all 18 combinations, up to **1920x1080 at ~30 fps** on both backends.

**What I do NOT know:** which of the changes fixed it, because several were changed at once. Leading suspects: another app holding the virtual camera, or a stale Iriun connection. If it recurs: close the Camera app and any other camera user, restart Iriun in that order, then re-run `camera_probe.py`.

**Takeaways for the code:**
- Camera index numbers are just DirectShow enumeration order and are not stable identities. Identify the camera by looking at a frame, or list device names with `pygrabber` when it matters.
- The scripts (`camera_scan.py`, `camera_probe.py`) print brightness, so "black" is a number, not a judgment call.
- DirectShow (what `live.py`/`webcam_check.py` already use) works fine with Iriun once the stream is healthy, so no backend option was needed.
- OpenCV's default for this camera is 640x480; asking for a size (1280x720 or 1920x1080) works. Which size to use for the live loop is a speed decision, to be made with the CPU FPS numbers.

---

## 2026-09-25 (cont.) — Week 3: first live run on the phone stream, and what the numbers mean

**Setup:** phone via Iriun (`--source 1`, DirectShow), OpenCV's default 640x480, `imgsz 640`, on the Windows laptop. The phone was filming a video of cars on a screen, so this is a rough test (screen glare, perspective, cuts between scenes), not a real drive.

**Results:**
- Camera alone (`webcam_check.py --camera 1`): 30 fps, brightness ~100-150 depending on the scene.
- Full loop (`live.py`): YOLO + tracker ~85-90 ms/frame -> ~11 fps steady. So we process about 1 in 3 of the camera's frames. Visible lag when waving a hand: small.
- The end-of-run summary said 7.0 fps / 137 ms average, which was **misleading**: the very first YOLO call took ~30 s (one-time lazy initialization), and the average included it. Steady-state numbers are the rolling ones (~11 fps, ~87 ms).
- Lead selection picked 8 different track IDs over 660 frames. Not meaningful on a screen video with scene changes; needs a real drive to judge.

**Fixes made from this run:**
1. `CAP_PROP_FPS` returns -1 for this camera. `cap.get(...) or 30.0` doesn't catch it (-1 is truthy), which would have broken `--out` video writing. Now: use the reported value only if `> 0`.
2. Warm-up: one inference on a blank frame before the loop, so the ~30 s startup isn't paid on the first live frame, and the wall clock used for TTC starts after it.
3. Summary now reports the **median** YOLO time, not the mean: one slow outlier no longer distorts it. (Lesson: use a median for latency numbers with occasional huge outliers.)

**Open risk (not yet confirmed): stale frames from the camera buffer.** The camera produces 30 fps but we consume ~11 fps. If the driver queues frames and `cap.read()` returns the oldest, the picture lags reality, and worse, TTC timestamps could be wrong: consecutive frames are really ~33 ms apart in the world, but the wall clock between our reads is ~90 ms, so dt would be overstated ~2.7x, making the estimated closing rate ~2.7x too small and TTC ~2.7x too large. If DirectShow instead drops old frames, dt is right. I don't know which it does. The standard fix is a background thread that reads the camera continuously, stamps each frame with the capture time, and keeps only the newest; the main loop then always processes the latest frame with its true timestamp. That also removes the lag. Next step.

**Also to decide:** which capture size to request (1280x720 or 1080p vs the 640x480 default), balancing detection of distant cars against decode cost. YOLO cost depends on `imgsz`, not the capture size, so a larger capture mostly costs decode time, but this is untested.

---

## 2026-09-25 (cont.) — Week 3: latest-frame grabber thread (fixes camera lag and TTC timestamps)

**Problem.** The live loop was `cap.read()` -> YOLO (~90 ms) -> `cap.read()` ... The camera makes 30 frames/s while we process ~11/s. If the driver queues the extra frames, `read()` returns an *old* frame, so (1) the picture lags reality (seen when waving a hand), and (2) the TTC timestamps would be wrong: consecutive queued frames are ~33 ms apart in the world, but our wall clock sees ~90 ms between reads, so `dt` is overstated ~2.7x, the estimated closing rate ~2.7x too small, and TTC ~2.7x too big. Whether DirectShow really queues that way was not confirmed, but the fix removes the question either way.

**Fix: `LatestFrameGrabber` in `live.py`.** A background thread calls `cap.read()` continuously (so nothing queues) and stores only the newest frame plus `time.time()` at the moment it arrived. The main loop calls `get(last_seq)`, which waits until a frame newer than the last one it processed exists, then returns `(frame, capture_time, seq)`. Frames that arrive while YOLO is busy are dropped on purpose. TTC's timestamp is now the capture time, not the time we finished processing.

**Threading pieces (for the interview):**
- A `Condition` (lock + wait/notify) protects the shared frame so the two threads never read and write it at the same instant, and lets the main loop *sleep* until the grabber signals a new frame, instead of busy-waiting.
- The GIL isn't a bottleneck here: `cap.read()` spends its time waiting on the camera (I/O) and releases the GIL while it waits.
- `seq` (a counter of frames read) is how `get()` knows a frame is new, and the difference between consecutive `seq` values is how many frames we skipped.
- The thread is a daemon and `stop()` joins it, so the program can exit.
- When the camera stops, the thread sets `running = False` and notifies, so the main loop wakes and exits instead of hanging.

**Only used for cameras.** For video files we want every frame in order, so files keep the plain `cap.read()` path with `t = frame_idx / fps`.

**How it was verified (no camera available in WSL):** a fake 30 fps camera (frame pixel value = frame number) and a fake 90 ms "YOLO". Results: 150 frames read = 57 processed + 93 skipped (counts add up exactly; ratio matches 33 ms vs 90 ms); every processed frame's content matched its sequence number (no stale or mixed-up frames); frame age when processing started: median 17 ms, max 48 ms, i.e. bounded by one camera frame interval + jitter; clean exit when the camera stops. The file path was re-run and unchanged. **Not yet verified on the real phone stream** (pending).

**New numbers printed at the end of a camera run:** frame age when processing started (median / max) and the number of camera frames dropped vs processed. These are the evidence for "lag is small" and "we process ~1 in 3 frames."

**Limitation:** the capture time is when OpenCV *returned* the frame, not when the sensor exposed it. Any latency inside the phone, Iriun and the USB link before that point is not measured. A constant delay like that doesn't affect `dt` (differences between timestamps), so TTC is unaffected, but it would matter for aligning the video with GPS/accelerometer data in Week 4.

---

## 2026-09-25 (cont.) — Week 3: decoupling display from inference (the real cause of the lag)

**What the measurements said.** After the grabber thread, the real-phone run reported frame age at processing start of median 19 ms / max 44 ms (fresh), 748 of 1196 camera frames dropped, ~10 fps processed, and it *felt laggier*. `webcam_check.py` (same camera, no YOLO) felt much less laggy. So the delay was added by our own pipeline, not by the phone, Iriun or USB. **The grabber thread was still the right design (it fixed the timestamps) but it did not cure the perceived lag**: the age I measured is the age when processing *starts*, and I had not accounted for the window itself only updating after each ~90 ms YOLO call.

**Root cause:** the window was redrawn only when YOLO finished, so the display ran at ~10 fps and every picture was already >= 90 ms old when shown. `webcam_check.py` shows a fresh frame every ~33 ms.

**Fix: two loops.**
- **Display loop (main thread, ~30 fps):** newest camera frame -> *copy* -> draw the latest available boxes/TTC on it -> show.
- **`InferenceWorker` (background thread, ~10 fps):** whenever free, takes the newest frame, runs YOLO + lead selection + Kalman, publishes the latest `FrameResult`.
- Result: the picture is as live as the camera allows, and only the overlay trails by about one YOLO call (~100 ms). Standard practice in real-time vision. TTC values are unchanged: same timestamps (capture time), same filter.

**Code changes (`live.py` restructured into small pieces):**
- `Pipeline.process(frame, t) -> FrameResult`: YOLO -> `LeadSelector` -> `TTCKalman`. No threads, no drawing. Used by both file and camera modes.
- `draw_overlay(img, result, info)`: draws lane band, every box + track id, the lead in green and the TTC readout. **Replaces Ultralytics' `r.plot()`**, because the overlay is now drawn on a *newer* frame than the one YOLO saw and `r.plot()` can only draw on its own frame. Sizes scale with frame width, so text is readable at 640 px and at 4K (also fixes the oversized-label problem noted earlier). Boxes now show `id N`, not class + confidence.
- The display loop draws on `frame.copy()`: the worker may still be reading that frame inside YOLO, and drawing in place would paint boxes into YOLO's input (a race between threads).
- `LatestFrameGrabber.get()` now serves several consumers (display + worker), each with its own `last_seq`, via `notify_all`.
- File mode keeps its sequential every-frame behavior and uses the same `Pipeline` + `draw_overlay`, so most of the new code is testable in WSL.
- `--log` columns changed: `processed_frame` (count of processed frames) instead of the camera frame index, since frames are now skipped.

**Verification without the phone (WSL):** file mode on 300 frames works (lead picked on 206 frames). Camera mode with a fake 30 fps camera playing the real clip at 640x480, real YOLO: display 27 fps while the model ran at only 2 fps (YOLO ~360 ms there because the fake camera decodes 4K on the same CPU and starves the model, so this is a harsher test than the laptop). Frame age median 17 ms / max 39 ms; all threads exited cleanly; the overlay looked right on a frame (lane band, ids, green lead, HUD, own hood ignored). The lead ID jumped in that test because the model saw frames ~0.5 s apart, which is expected and not representative.

**Not yet verified on the real phone.** Expect: display ~30 fps, model ~10 fps, and much less lag.

**Trade-off to remember:** boxes trail moving objects by ~100 ms. Fine for a car ahead, but visible on fast lateral motion. Faster inference (smaller `imgsz`, ONNX/OpenVINO later) shrinks it.

---

<!-- Add new dated entries above this line as the project progresses. -->
