# Driving Risk Analyzer

A computer vision + sensor-fusion system that analyzes dashcam video and phone GPS/accelerometer logs to measure driving risk: detects and tracks vehicles, identifies the lead vehicle, estimates time-to-collision (TTC), distance, and headway, and detects risky events (tailgating, hard braking, rapid closing, speeding, potholes).

Portfolio project built to be explainable end-to-end, not just demoed: every design decision, tradeoff, and failure mode below is backed by a real experiment, logged as it happened in [`NOTES.md`](NOTES.md).

**Not a safety system.** A research/portfolio prototype, not something to rely on while driving.

## Status

Weeks 1–4 complete: detection, tracking, lead-vehicle selection, Kalman-filtered TTC, ground-plane distance, ego speed/headway, core events (tailgating/rapid-closing/hard-braking), and speeding detection (GPS vs. OSM limits) — all built and evaluated on real KITTI data. Live real-time capture (an earlier plan) was built and tested against fake cameras, then dropped from scope to focus on recorded/downloaded video given time constraints — see `CLAUDE.md`'s Scope section and `NOTES.md`, 2026-09-28.

## Demo

`outputs/kitti_0020_annotated.mp4` — the full pipeline (detection, tracking, lead selection, TTC, distance, headway, speeding, and event flags) overlaid on a real Autobahn traffic sequence. Picked after screening all 21 KITTI tracking sequences and visually checking candidates, not just the first one downloaded — see `NOTES.md`, 2026-09-28. This one has real, sustained heavy-traffic tailgating (headway hovering ~1.9s for over 6 seconds) and genuine closing events, not just a quiet clip with nothing happening.

## Results

**Distance accuracy on KITTI, by range** (ground-plane geometry fit on sequence 0019, evaluated held-out on sequence 0009 -- see `NOTES.md`, 2026-09-28 for the fitting method and full discussion):

| range (m) | n objects | median \|error\| (m) |
|---|---|---|
| 0-10  | 179 | 0.71  |
| 10-20 | 476 | 1.23  |
| 20-40 | 797 | 4.61  |
| 40+   | 503 | 12.22 |

Accurate at practical following distances, degrades sharply with range -- an expected, textbook property of monocular ground-plane distance (error grows roughly with the square of distance), not a bug. A metric depth model is the natural comparison at long range, planned but not yet built.

**Speeding detection (GPS vs. OSM speed limits)**, verified on two KITTI sequences:

| sequence | tag coverage | peak speed | matched limit | flagged? |
|---|---|---|---|---|
| 0019 | 100% (1059/1059 frames) | 20.6 km/h | 30 km/h | no (correct) |
| 0009 | 59% (477/803 frames) | 53.5 km/h | 50 km/h | no (correctly under the 5 km/h noise margin) |
| 0020 | 100% (837/837 frames) | 53.8 km/h | steps 30→50→100 km/h | no (congestion-limited, correct) |

Sequence 0020's matched limit steps exactly where the real road changes (residential → arterial → Autobahn), cross-checked against a literal "100" speed-limit sign visible in the footage at that point — a nice independent confirmation the road-matching is working, not just the threshold logic.

**Core events (tailgating / rapid closing / hard braking)**, on the same three sequences, using our full detector+tracker output (not ground-truth boxes):

| sequence | tailgating | rapid closing | hard braking | note |
|---|---|---|---|---|
| 0019 | 0 | 0 | 0 | quiet clip (turned out to be a pedestrian plaza, not a road — see Demo note) |
| 0009 | 1 | 6 | 0 | busy merge-heavy area; some events likely reflect the Scope limitation below, not confirmed either way |
| 0020 | 5 (incl. a genuine 6.6s sustained stretch at ~1.9s headway) | 2 | 0 | real heavy-traffic congestion |

No sequence in the whole 21-sequence KITTI tracking set ever exceeds the 0.3g hard-braking threshold (peak across all of them is ≈-2.8 m/s²) — these are short urban/highway clips, not near-miss footage, so 0 hard-braking recall here reflects the dataset, not a pipeline gap.

Only runs on KITTI, the only source in this project with real GPS — a downloaded dashcam clip has no GPS to check against a speed limit at all.

*Still in progress: tracker comparison (ByteTrack vs. BoT-SORT, HOTA/IDF1) and event detection precision/recall on hand-labeled clips.*

## Scope

Lead-vehicle detection targets **steady same-lane following** — highway-style closing, tailgating, rapid approach. It is not designed to identify or predict cross-traffic, merges, cars pulling out of driveways, or other vehicles' turning intent. That's a real trajectory/intent-prediction problem, and a much bigger one than this project takes on.

This boundary was found concretely, not assumed up front: investigating real KITTI driving sequences for a TTC ground-truth check (below) turned up several plausible-looking "closing vehicle" tracks that turned out to be a parked car being driven past closely, or a car merging in from a driveway — not a lead vehicle. The lead-selector correctly ignoring these is the system working as scoped, not a gap.

## Known limitations (found and root-caused, not guessed at)

Each of these was found by testing against real data — either real footage or KITTI's ground-truth 3D labels — and root-caused before being written down here. Full details, numbers, and the plots behind each one are in `NOTES.md` (2026-09-28 entries).

**1. Box-width TTC breaks down when the box is clipped by the frame edge — exactly at the closest, most dangerous range.**
TTC is estimated from how fast a vehicle's bounding box grows (`TTC ≈ width / (d width/dt)`). When a car gets close enough to start exiting the frame, its *labeled* box (checked against KITTI ground truth) can shrink even while the car keeps closing, because it's being clipped rather than shrinking/growing with the car's true extent. Naively trusting that measurement makes the filter report TTC *increasing* right when it should be at its lowest. Mitigated by detecting edge-truncation and freezing the filter's measurement update on those frames (coast on the last trusted rate, same principle used for a temporarily-missing lead) — this stops the filter from reporting the wrong direction, but it does not recover the lost information: once a box is clipped, its width genuinely no longer tells you how much closer the car has gotten. This is an inherent blind spot of box-width TTC at the closest ranges, not a tuning problem.

**2. Occlusion breaks tracker ID continuity, which resets the TTC filter.**
On a real KITTI sequence, a car passing behind another object for roughly 1–4 seconds caused the tracker (ByteTrack) to assign it a new ID once it reappeared — twice, in one 26-second clip. Each ID change resets the Kalman filter's running estimate, since a new ID might be a different vehicle. This is a genuine cost of occlusion in real footage, not specific to KITTI.

**3. On real ground-truth-backed evaluation, both of the above compounded to produce zero usable TTC readings during the one actual near-collision in the test clip.** The car was undetected during occlusion, and — pending further work — even once redetected, was outside the frame's expected lane region during the truncated final approach. This is the honest headline result of the KITTI side-check so far: the current pipeline's biggest weaknesses are upstream of the TTC math itself (detection/tracking continuity, lead selection), not the Kalman filter.

**4. A lead-vehicle switch resets the TTC filter's state, so detection now waits ~1s before trusting a rapid-closing reading after any switch.** When `LeadSelector` switches "current lead" to a different tracked vehicle — a legitimate call when several real vehicles are plausible candidates, not just an occlusion ID-switch — `TTCKalman` intentionally restarts from scratch. A guard (`MIN_TRUST_AFTER_SWITCH_S`) now suppresses rapid-closing detection for 1s after any switch. Tested honestly rather than assumed to matter: on a real KITTI sequence with 6 rapid-closing events, disabling the guard changed almost nothing (one event started 0.2s earlier) — every event turned out to be backed by a track visible for 1.7-8.5 seconds with TTC decreasing smoothly, not single-frame noise. So this guard is correct in principle but wasn't actually the fix needed here; what's genuinely unresolved is whether those 6 events reflect real same-lane closing risk or the merge/cross-traffic scope limitation above — the measurement is real, what it means isn't confirmed. See `NOTES.md`, 2026-09-28 (the correction entry).

## Situational awareness beyond the single lead vehicle

In addition to the single selected lead vehicle (used for headway/tailgating scoring), the system reports TTC for **every** currently-tracked vehicle in frame, using the same box-width method (`NearbyVehicleTTC` in `lead_ttc.py`). This doesn't attempt to judge which nearby vehicle matters most or predict cut-ins — it's the same "how fast is this box growing" question already answered for the lead, applied to the rest of the scene, so the report reflects more of what's actually happening around the car.

## Not doing

See `CLAUDE.md` for the full list and reasoning. Notably: no live/real-time demo (dropped for time — the code exists, frozen, in `live.py`'s camera mode); no cut-in/merge/cross-traffic intent prediction (see Scope, above); no custom detector trained from scratch except a small pothole-detector fine-tune; no cloud deployment, database, or mobile app.
