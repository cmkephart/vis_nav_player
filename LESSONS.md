# LESSONS — how our visual-navigation agent works, and how to build it again from scratch

Written for Caleb (new to robot perception), to understand the final product well enough to rebuild it and
explain it. It replaces the earlier lesson notes and reflects the **final** design. Phase A changed completely
during the project; §5–§7 show what changed and why.

How to read it: §1–§2 give the problem and the toolbox; §3 is phase B (the finish); §4 is the original phase A and
why it failed; §5 lists what we tried and rejected; §6 is the final phase A, step by step; §7 compares old and new
side by side; §8 holds the engineering lessons; §9 is a rebuild order.

---

## 1. The problem in plain terms

- A robot in a maze, and you see what its camera sees. You get two things before you start: recordings of 3 earlier
  drives (the **exploration data**: about 3,000 pictures with the key that was pressed), and **4 photos taken at the
  goal**, facing front, left, back and right.
- You drive with the arrow keys. Each press is one **tick**: about 0.029 m forward, or a 3.44° turn.
- **Scoring:** being close (≤ 0.1 m) and facing the right way (≤ 10°) gives 6 points. The last 4 points are **SPL**,
  the shortest possible route's ticks divided by yours. So SPL rewards not wandering.
- **Facts that shaped every decision** (all measured):
  - The maze is a grid of 0.2 m cells: narrow corridors, lots of dead ends, gaps in walls you can see through but
    not drive through.
  - Wall textures repeat, so one picture can match many places.
  - Motion is noisy. On the real server, driving forward also turns you left about 1.2–2.4° per tick.
  - The exploration labels are lossy: a "LEFT" record might be 3 or 5 ticks of turning.

## 2. The toolbox (concepts you need)

* **Features (SIFT / RootSIFT).** A feature is a small distinctive patch in an image (a corner of a poster),
  described by 128 numbers that stay similar when the patch is seen from a slightly different angle. RootSIFT is
  SIFT with a square-root normalisation that matches better.
* **Matching + ratio test.** For each feature in image A, find its two nearest features in image B. Keep the match
  only if the best is clearly better than the second best (ratio 0.8). This removes ambiguous matches, which is
  important when textures repeat.
* **RANSAC.** Matches still contain mistakes. RANSAC repeatedly picks a few matches, fits a model (for example a
  homography), and keeps the model most other matches agree with. Those are the **inliers**. The inlier count is
  our main "how similar are these two views" number.
* **Homography for pure rotation.** If the camera only *rotates* (no movement), every pixel moves according to
  H = K R K⁻¹ (K is the camera matrix, R the rotation). Fit H, recover R, and you get the turn angle between two
  pictures, accurate to about 0.5°. **This is our most-used tool:** heading at the goal, the turn between frames, and
  "same position" tests.
* **Parallax (the "same position" test).** If two pictures were taken from different positions, nearby things shift
  differently from far things, and no pure rotation explains the matches. The leftover error after the best
  rotation fit is the parallax, in pixels. **Small parallax means the same spot**; large parallax means a different
  spot, even with many matches.
* **Essential matrix.** When the camera both moves and turns, the essential matrix (from ≥ 5 matches) gives the
  rotation and the *direction* of movement, but not the distance. A single camera can't measure distance
  (monocular scale ambiguity).
* **Place recognition (VLAD).** A whole image summarised as one vector, to find "which recorded picture looks most
  like this". Fast but easily fooled by repeated textures. The course baseline uses it.
* **Teach-and-repeat.** A robot is driven along a route once (teach), then repeats it by matching its live view to
  the recorded pictures in order (repeat). It needs no map, only the sequence.
* **SPL (Success weighted by Path Length).** The oracle's path length divided by yours, capped at 1.

## 3. Phase B — the finish (unchanged since mid-project, frozen)

**Job:** once you're near the goal, line up exactly and say when to check in.

1. **Measure each frame:** RootSIFT-match the live view against all 4 goal photos.
   - A rotation homography against the front view gives the **heading error** (how many degrees to turn).
   - The essential matrix gives the **direction to the goal** (left or right of straight ahead).
   - The inlier count says how close you are; it's a proxy, since there's no metric distance.
2. **Engage** when the front view has ≥ 15 inliers, or any view passes its "spot" bar.
3. **Probe turn:** turn 3 ticks and check that the heading estimate moved by the known amount (~10°). A look-alike
   poster gives inconsistent estimates. After 3 failures, treat the place as a look-alike and ignore it for 30 frames.
4. **ALIGN** (turn until the heading error ≤ 3°) or **MOVE** (short straight legs that end when the goal is level
   with you).
5. **READY** when the front view has ≥ 35 inliers and the heading is aligned. The **driver presses SPACE** (confirm
   mode). The first SPACE before READY is blocked; a second within 3 s forces a check-in.

**Lessons from phase B:**
- Heading is easy (~0.5° error); distance is hard (monocular).
- The READY bar of 35 inliers means about 4 cm on texture-poor mazes but 10–15 cm on rich ones, which is why the
  human confirms and may nudge closer.
- A parallax limit we tried caused 232 ticks of wandering on one maze, so it was turned off.
- Phase B's moves are `(Action, n)` tuples; wrapping them again crashed a run.

## 4. Phase A — the original approach, and why it failed

**4a. Course baseline:**
- VLAD over the exploration frames (subsampled to every 5th record); a graph of consecutive frames plus VLAD
  shortcuts; hints "next move along the shortest path".
- Result: phase B carried every first drive to 6/10, but SPL was 0.11–0.32. The hints were coarse, and the route
  wasn't short.

**4b. Louie's route tracker** (the version this project started from):
- **Graph:** every recorded frame (subsample 1, a node per 5 ticks), consecutive steps, plus "verified overlap"
  links, where two drives saw overlapping scenery (a diagonal of matching neighbour frames).
- **Goal:** match all frames against all 4 goal photos, cluster ±3 frames, and **sum the scores across the views**.
- **Tracking:** live SIFT matching against a window of frames around the last position, widening up to 140
  frames per tick when lost.
- **Hints:** turn-by-turn text from the live view versus frames further along the route; about 100 tuned constants
  in 930 lines.

**What went wrong, with the evidence:**
- **Wrong goal.** On maze 21 the vote chose a place far away (node 2069) whose score came from *partially* matching
  three different views; the real goal was the runner-up. A different look-alike, one big reused map poster, scored
  97 matches against 50 for the real goal.
- **Flickering, jumping hints.** The hint text changed on 450 of 657 updates (68%), and the position jumped more
  than 30 nodes 14 times. With global matching, any similar-looking place can capture the tracker.
- **The goal was never found** on maze 21 (an unseen maze).
- **Overlap links can cross walls:** seeing a place through a gap made a link that can't be driven.
- **Even with the right goal**, routes along recorded drives were long (maze 21: 2,579 ticks vs an optimal 532),
  because the explorer wanders into dead ends.

**What was right and kept:** subsample 1 (each frame needs overlapping neighbours to track); the idea that a single
frame match lies, while a *run* of consistent matches can be trusted; phase B; the human driver.

## 5. What we tried and rejected (and why: the evidence matters for the report)

| idea | test | result |
|---|---|---|
| Map from recorded actions (dead reckoning), snapped to the 90° grid | `sim_probe.py`: replay against the simulator's true maze | motion noise (0.3, 0.1) plus lossy labels (LEFT records bimodal 8–11° / 17–20°, FORWARD hides 4–7° of turn, IDLE rotates 10°): paths sprawl far beyond the 6×6 m maze. **Rejected.** |
| Map from image heading + label distances | integrate frame-to-frame rotation | heading fine, distance labels too lossy; only 21–61% of positions landed in free cells. **Rejected.** |
| Tell goal from look-alike automatically by layout spread | count image strips holding matches | real goal 4 strips, look-alike 3–4. No separation. **Human picks instead.** |
| ORB-SLAM | reasoning | C++ build on Windows, no metric scale, loses track when turning on the spot, too little time. **Rejected.** |
| DINO features | reasoning | better retrieval, but the failures were geometric (no map, look-alikes), not retrieval. **Rejected.** |
| Only same-pose links between recorded drives | maze 21 | safe, but few (20–65), so the route stays ~5× the optimum. **Kept, but not enough alone** (learning fixed this, §6.8). |

## 6. The final phase A — build it step by step

**6.1 Load the exploration data.**
- Every record is one frame, 5 ticks apart, from 3 drives that all start at the spawn.
- Compute RootSIFT for every frame (about 2 minutes) and cache it.
- Compute the turn between each pair of consecutive frames with the rotation homography; this gives each drive's
  heading track.

**6.2 Same-pose links.**
- Shortlist candidate pairs with tiny thumbnails (32×24, correlation over ±6 px horizontal shifts).
- Verify each with SIFT + RANSAC + rotation fit: **parallax ≤ 6 px, ≥ 30 inliers, matches spread over ≥ 5 of 8
  vertical strips.**
- Such a pair is the same spot, so switching between those drives is a turn on the spot.
- Lesson: **never link places just because one can see the other.** That's how routes cross walls through gaps.

**6.3 Route graph.**
- Edges: consecutive frames of a drive (cost 5 ticks); undoing a pure turn (turn back on the spot); same-pose links
  (cost = turn angle / 3.44); and "all drives start at the spawn" (cost 0).
- Dijkstra from the spawn to the goal frame gives the route.

**6.4 Goal candidates (the human picks).**
- Score every frame against each goal view **separately**: homography inliers, with matches in ≥ 2 strips.
- Sum over a ±3-frame window, per view, and never across views (summing across views was the old bug).
- Show the top 5 clusters, separated in time, each **beside the goal photo it matched**. The driver chooses the one
  whose *neighbouring* panels also match.
- Candidate 1 is only the strongest match: on maze 21 the goal was #2.

**6.5 Live tracking along the route only.**
- Each new frame is compared only with route frames near the current position (−6 … +30), plus a sparse look over
  the rest of the route ahead (to catch the driver skipping the explorer's loops).
- Steps: thumbnail pre-filter → SIFT verify the best 8 → accept at ≥ 15 inliers.
- A **far jump** needs ≥ 30 inliers, **two confirmations**, and **parallax ≤ 25 px**. One replay showed a 69-px
  "match" sending the driver back 180 frames; the parallax check stops that.
- Because only the route is searched, look-alikes elsewhere in the maze can't capture the tracker. This is the core
  difference from the original.

**6.6 Hints people can follow.**
- Steer toward the route point **0.25 m ahead**, not the explorer's momentary heading (the explorer wiggled
  constantly).
- An earlier look-ahead of 0.5 m cut corners in 0.2 m corridors and gave twice as many false TURN hints; tuned on a
  replayed real drive (291 → 158).
- Say TURN only beyond **30°**.
- One instruction at a time, and a new kind must repeat before it replaces the shown one.
- Show pictures: **YOU ARE HERE · NEXT TURN starts here · after the turn · GOAL**.
- The panel shows only ~54 characters per line, so wrap the text.

**6.7 Helpers added after real drives:**
- **Turn memory.** If the driver turns away from the route, no route frame matches any more. Keep measuring the turn
  between consecutive live frames, and as long as the driver hasn't driven more than 10 ticks, say "OFF THE ROUTE:
  TURN LEFT ~n presses". On the maze-77 replay, "not sure" fell 579 → 318 steps.
- **Dead-end warning.** When the route ahead turns around (the explorer entered a dead end), say so; the driver can
  skip it and the tracker catches up.
- **Look-alike guard.** If phase B sees a goal-like view while the route says the goal is more than 60 frames away,
  phase B doesn't take over, and the hint warns. (Maze 77: an 81-inlier, perfectly aligned "goal" at route 317 of
  713.) Holding **G** overrides, in case the route itself is wrong.

**6.8 Learning from earlier drives — the biggest gain.**
- Every earlier drive of the *same maze* (`runs/nav_*.pkl`) is added as an extra drive, linked to exploration frames
  wherever the driver's view matched one from the same spot (parallax ≤ 10 px).
- Why it works: the driver's camera bridged explorer passes that never shared a pose, so the exploration data alone
  couldn't link them.
- Results: maze 77, 3,534 → 1,426 ticks (drive 2: 7/10); maze 88, drive 2: 10/10; online challenge, 2,512 → 782
  ticks (attempt 2: 10/10).
- Lesson: **attempt 1's drive is data for attempts 2–3.** Drive it cleanly.

**6.9 Drift trim (server only).**
- On the server, every forward tick also turned the robot ~1.5° left; a local control measured 0.07°.
- Measure it from the rotation between consecutive frames on the driver's forward ticks. Once consistent (≥ 30
  samples, ≥ 0.6°/tick, 70% on one side), turn some plain FORWARD ticks into FORWARD+RIGHT. That's still one tick, so
  it's free in SPL. At 1.5° per tick, that's about 2 in 5.

## 7. Original vs final, side by side

| aspect | original (Louie's tracker) | final (`nav_agent`) | why it changed |
|---|---|---|---|
| goal | automatic vote, scores summed across views | candidates scored per view; **human picks** | vote picked a look-alike (maze 21); a human sees the neighbouring panels |
| links between drives | "saw overlapping scenery" | **same position only** (pure rotation, ≤ 6 px parallax) | overlap links can cross walls through gaps |
| where the tracker searches | whole maze (window, widening to 140/frame) | **only along the chosen route** | global matching jumped between look-alikes (14 jumps > 30 nodes) |
| far jumps | 3 confirmations | 2 confirmations **plus parallax ≤ 25 px** | replay: a 69 px false match sent the driver back |
| hint | turn-by-turn from the explorer's heading; 68% of updates changed it | travel direction 0.25 m ahead, 30° threshold, hold-before-change, pictures | explorer wiggles; corridors 0.2 m |
| when lost | "lost: retrace" | turn memory, dead-end warning, "near the goal" message | most lost time came from turning away |
| phase B start | when goal-like | also requires the route to agree (look-alike guard, G overrides) | maze 77 false READY |
| later attempts | same hints | **learn from earlier drives** | maze 77: 3,534 → 1,426 ticks |
| actuation | assumed ideal | **drift trim** | server: +1.5°/tick left on forward |
| size | tracker 931 + route agent 327 lines, ~100 constants | map 455 + follower 270 + agent 472 + drift trim 67 lines, ~55 named constants | |
| unseen-maze results | maze 21: never found the goal | 42/77/88/online: every goal found; 6–8 on attempt 1, 7–10 on attempt 2 | |

## 8. Engineering lessons (how we worked)

1. **Measure before you build.** Every big decision came from a small measurement: the vote's mistake, the
   dead-reckoning failure, the lossy labels, the drift. Two days of building a metric map would have failed; one
   probe script proved it.
2. **Always run a control.** The drift looked real; a local drive measuring 0.07° proved it was the server, not our
   measurement.
3. **Replay real drives offline.** Every drive saves frames, keys and hints. Feeding them back through the follower
   reproduced each failure exactly, so every fix was tested before it was driven. (The development sandbox had no
   simulator at all.)
4. **Put the human where humans are good.** People are excellent at "does this place look like the goal photo" and at
   local driving. Machines are good at precise angles and remembering long routes. Split the work that way.
5. **Fair tests need unseen mazes.** A maze you've driven three times measures your memory, not the agent.
6. **Safe beats short on the first attempt; learning makes the second one short.**
7. **Check the plumbing.**
   - The SDK runs a pre-flight `setup()` with random-static goal photos; skip it.
   - The keyboard object supports `"key" in keys` but isn't iterable, and quick taps only show in `keys.tapped`.
   - Files downloaded in a browser aren't in `source/` until you copy them.
   - PowerShell starts in `C:\WINDOWS\system32`, so `cd` into the repo first.
8. **Secrets go in `.env` (git-ignored), never in `.env.example` (tracked).** Our key leaked that way and had to be
   removed and regenerated.

## 9. Rebuild order (if starting from scratch)

1. Kit setup; drive locally with the keyboard agent to get a feel for the maze.
2. Phase B: goal-photo matching → heading from the rotation homography → probe/align/move → READY + SPACE. Test
   with the true pose in the local simulator (`calibrate.py`).
3. Goal candidates, scored per view + the picture + number keys.
4. Exploration features, consecutive-frame turns, same-pose links, Dijkstra route.
5. Route follower: local search, confirmed far jumps with the parallax check, hint from travel direction, picture
   strip.
6. Record every drive; build the offline replay; tune on replays.
7. Turn memory, dead-end warning, look-alike guard.
8. Learning from earlier drives.
9. Drift trim (test online; local mazes have no drift).
