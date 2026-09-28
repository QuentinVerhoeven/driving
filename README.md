# Driving Risk Analyzer

A computer vision + sensor-fusion system that analyzes dashcam video and phone GPS/accelerometer logs to measure driving risk: detects and tracks vehicles, identifies the lead vehicle, estimates time-to-collision (TTC), distance, and headway, and detects risky events (tailgating, hard braking, rapid closing, speeding, potholes).

Portfolio project built to be explainable end-to-end, not just demoed: every design decision, tradeoff, and failure mode below is backed by a real experiment, logged as it happened in [`NOTES.md`](NOTES.md).

**Not a safety system.** A research/portfolio prototype, not something to rely on while driving.

## Status

Weeks 1–2 complete (detection, tracking, lead-vehicle selection, Kalman-filtered TTC on recorded clips). Live real-time capture (an earlier plan) was built and tested against fake cameras, then dropped from scope to focus on recorded video given time constraints — see `CLAUDE.md`'s Scope section and `NOTES.md`, 2026-09-28. Current work: GPS/accelerometer logging, distance, headway, and core event detection.

## Results

*Evaluation in progress. This table will report distance accuracy on KITTI (by range), tracker comparison (ByteTrack vs. BoT-SORT, HOTA/IDF1), and event detection precision/recall on hand-labeled clips, once each is complete.*

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

## Situational awareness beyond the single lead vehicle

In addition to the single selected lead vehicle (used for headway/tailgating scoring), the system reports TTC for **every** currently-tracked vehicle in frame, using the same box-width method (`NearbyVehicleTTC` in `lead_ttc.py`). This doesn't attempt to judge which nearby vehicle matters most or predict cut-ins — it's the same "how fast is this box growing" question already answered for the lead, applied to the rest of the scene, so the report reflects more of what's actually happening around the car.

## Not doing

See `CLAUDE.md` for the full list and reasoning. Notably: no live/real-time demo (dropped for time — the code exists, frozen, in `live.py`'s camera mode); no cut-in/merge/cross-traffic intent prediction (see Scope, above); no custom detector trained from scratch except a small pothole-detector fine-tune; no cloud deployment, database, or mobile app.
