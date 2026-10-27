# HANDOFF — state after Tue 10/27 midday (overnight session + maze-77 follow-up)

Read `README.md` first (setup, agents, driving, repo map). This file is the *working state*: what was decided, why,
the evidence, how to work on it offline, and what is left. It supersedes `PROJECT_STATE.md` and
`PHASE_A_HANDOFF.md` (both historical).

**Deadline:** graded challenge Wed 10/28 8 pm – Thu 8 am ET (`da90a443e215`, 3 attempts); report Fri 10/30 8 pm.
**Driver:** Caleb (manual arrows). Louie is developing his own agent independently and will use the README.

---

## 1. Where things stand (one paragraph)

Phase B (final alignment + READY) is frozen and reliable: 6/10 (distance 3 + heading 3) on every first drive. Phase A
was rebuilt from scratch as **`nav_agent.py`** (teach-and-repeat along recorded exploration drives) after Louie's
route hints proved unreliable. On the one fair test so far (**unseen maze 42**) it reached the goal, scoring 6/10 with
SPL 0.34; the driver said the hints were "much much better than before, still a little finicky". Replaying that drive
offline led to three fixes (corner-cutting turn hints, a backward jump near the goal, the lost message near the goal)
and a fix for number-key taps.

**Tue midday, unseen maze 77:** 6/10, SPL 0.44 (best so far); the key-tap fix worked live. Three new features came
from replaying that drive: a **look-alike guard** (phase B can't take over while the route says the goal is more than
60 frames away; hold G to override), **turn memory** when lost, and **dead-end warnings**. The biggest one:
**learning from earlier drives**: your drives of the same maze in `runs/` join the map, shortening the route on
maze 77 from 3,534 to 1,426 ticks. **Learning, the guard and turn memory are tested on replays only, not yet live.**
The fallback (`final_route_agent.py` + picker) still works.

## 2. Decisions and the evidence behind them

| decision | evidence |
|---|---|
| Keep phase B as is | 0.002–0.07 m, ~1° on every finish; caveat: the READY bar of 35 inliers can come 10–15 cm early on rich mazes (maze 202: 0.11 m cost a point) → the driver checks inliers and nudges |
| Human picks the goal from candidates | maze 21: the vote picked node 2069 (look-alike), the real goal was runner-up 2302–2306. A single reused map poster gave a look-alike 76–97 inliers vs 50 for the real goal; layout spread (strips) can't separate them (3–4 vs 4). A person sees it at once from the neighbouring panels |
| Score candidates per goal view, not summed | summing let a far spot that half-saw several views win (maze 21) |
| No metric map / dead reckoning | `sim_probe.py`: motion noise (0.3, 0.1); exploration labels are lossy (LEFT records measure bimodal 8–11° / 17–20°, FORWARD records hide a median 4–7° of turn, IDLE rotates ~10°); dead-reckoned paths sprawl far beyond the 6×6 m maze. Replaying recorded actions doesn't reproduce the drives (pixel diff ~100) |
| No ORB-SLAM / DINO | ORB-SLAM3: C++ on Windows, no scale, loses track in turns-on-the-spot; DINO improves retrieval, but failures came from missing geometry, not retrieval. Not achievable or useful by Wednesday |
| Routes only through recorded driving + same-pose links | the driver warned of wall gaps you can see through but not pass, and many dead ends. "Saw it" links (Louie's overlaps) can cross such gaps; same-pose links can't |
| Track along the route only | Louie's tracker matched the whole maze: on maze 21 the hint text changed on 450 of 657 updates, with 14 jumps of more than 30 nodes |
| Steer at 0.25 m ahead, turn threshold 30° | replay of the real maze-42 drive: at 0.5 m / 20° there were 291 steps of "TURN" while the driver (correctly) drove FORWARD; 0.25 m / 30° gives 158 |
| Look-alike guard: phase B can't engage while the route has a fix and > 60 frames remain (G overrides) | maze 77: phase B saw 81 aligned inliers at route ~317 of 713 (real goal at 708: 140 inliers). The follower had a fix there for 18/18 frames; on replay both false-positive spans are blocked |
| Turn memory when lost (≤ 10 ticks driven since the last fix) | maze 77: the driver turned against a hint at route 16 and the follower said "not sure" for 162 steps. On replay "not sure" fell 579 → 318, with 283 steps of turn guidance |
| Learn from the driver's own drives | maze 77: the driver's live view bridged two explorer passes that never share a pose (13–37 px between them), so the exploration data alone can't link them. Adding the drive: route 3,534 → 1,426 ticks. Maze 42: drive 2 replayed on the route learned from drive 1 tracked to the end with no backward jumps; route 1,992 → 1,543 |
| Far jumps: 2 confirmations + parallax ≤ 25 px | replay: after 33 lost frames near the goal it accepted route frame 210 (from 388) at 69 px parallax → "go back" hints. Fixed: 0 backward jumps on replay |

## 3. Facts learned this session (not in older notes)

* **SDK pre-flight:** every session first calls `setup()` with random-static goal photos (this explains the old
  "setup ran twice with different goals" mystery). Skip it with `goal_picker.is_noise`.
* **`viewer.keys()`** returns a `Keys` object: membership only (`"2" in keys`), not iterable; it has `held`,
  `tapped` and `any`. Quick taps can fall between polls of `held`, so `goal_picker.digit_of_any` also checks `tapped`.
* **Local simulator** (`vis_nav_sim.Simulator`) exposes `grid` (list of strings, `#`/`.`), `cell_size` 0.2,
  `spawn`/`spawn_cell`, `target`/`target_cell`, `oracle_steps`, `spl()`, `occupancy`, `motion_noise` (0.3, 0.1),
  `drawn_bias()` → [1,1,0]. `record()` is native, so per-frame true poses can't be captured. The cell index is
  (row = y/0.2, col = x/0.2); yaw 0 = +x.
* **Exploration `data_info.json`:** `step` counts records (1 per record, 5 ticks each), not ticks. The action label
  of record k+1 best describes the motion from frame k to k+1.
* **Server data includes `target.jpg`** (the `record()` docstring says it writes "the format the server hands
  out"), so `nav_agent` pre-scores the goal before connecting.
* **Oracle routes:** maze 21 = 532 ticks (spawn cell (13,1), goal (16,23)); maze 42 = 571 (spawn (13,1), goal (29,19));
  maze 77 = 1,009 (spawn (13,1), goal (29,23)). The spawn cell was (13,1) on all three.
* **Recorded-route lengths:** maze 21 = 2,579 ticks to the true goal (candidate #2, traj_2/1161); maze 42 = 1,992
  (candidate #1, traj_2/1238). Same-pose links: 65 (maze 21), 190 (maze 42). Build time (sandbox): ~3–4 min for
  features, turns and links, plus ~1 min for goal scoring.

## 4. Open items, in priority order

0. **Live test of learning:** drive maze 77 again on the computer that holds `runs/nav_20261027_113820.pkl`.
   - The console should say `nav: learned from 1 earlier drive(s)`, and the route should be ~1,426 ticks.
   - Goal: SPL ≥ 0.5 (1 point) or ≥ 0.75 (2 points). This rehearses challenge attempts 2–3 exactly.
   - Also watch for the guard message at look-alikes and for turn-memory hints.
1. **Live test of the fixed `nav_agent` on a fresh unseen seed** (not 3, 7, 11, 21, 25, 42, 58, 101, 202, 303, 314),
   e.g. `--local 77`.
   - Confirm that a number-key tap registers (console `nav: goal = candidate N` / `(… already the goal)`).
   - Collect the console output, `runs/nav_*.pkl` and `export_truth.py 77` zips.
   - Replay offline (§5) and compare: steps of TURN while the driver drove FORWARD, lost spans, backward jumps.
2. **Remaining "finicky" hints.** On replay, 665 of 1,683 steps still showed a TURN kind. Ideas, in order:
   - (a) Raise `TURN_SAY` to 35–40°.
   - (b) Measure `need` against a route frame ~0.25 m *ahead*, via its own rotation fit, instead of `fix.yaw + B−H`.
   - (c) Suppress TURN while the driver is already turning that way (we know the last key).
   Tune each on the maze-42 replay before shipping.
3. **Attempts 2–3 / SPL.** **Built since: learning from earlier drives (`tr_map.add_runs`, on by default) — see
   item 0.** The notes below are the earlier analysis. SPL ≥ 0.5 needs about 2× the oracle or better; recorded routes
   are 3.5–5×, and the driver's own first drives 3×. Options:
   - (a) A "replay my run" mode: a route from the driver's own attempt-1 `runs/nav_*.pkl` (each step is 1 tick
     with the key known exactly), with loops cut by same-pose links within the run, followed by the same `Follower`.
     Needs `TRMap` to support ticks-per-record = 1. This only helps if loops get cut, and same-pose revisits are rare.
   - (b) A route sheet (key counts plus thumbnails at turns) from the attempt-1 run, to help the driver remember.
   - (c) Accept driver memory (9/10 on mazes 202 and 7 when the route was known).
   Discuss with Caleb before building.
4. **Verify on the course site:** does downloading the data / building caches before connecting spend an attempt
   (the scripts have `--yes`, which suggests a prompt before connecting)? Does the best or the last attempt count?
5. **Bigger maze risk:** the link shortlist builds an n×n float32 matrix (n = exploration frames). 3,000 frames →
   36 MB; 10,000 → 400 MB, plus a temporary of the same size. If the challenge data is much larger, chunk the
   matrix in `tr_map.build`.
6. **Report slides:** problem + references; method ≤ 2 slides (phase A teach-and-repeat + human goal choice;
   phase B rotation/essential-matrix alignment). Use the results table in the README.

## 5. How to work offline (no simulator in the sandbox)

The sandbox can't install `vis_nav_sim` or `vis_nav_sdk` (the course package index is blocked). Everything above
was developed against **uploaded data plus a stub SDK**:

* Data: the driver runs `uv run source/export_truth.py <seed>` and uploads `logs/upload_<seed>_part*.zip`
  (data + `truth.json` with grid, spawn, goal and oracle). Unzip into e.g. `/home/claude/m42`.
* Drives: `runs/nav_*.pkl`, a pickle stream. The first record is a header `{header, data, goal}`; then one record
  per step `{step, key, count, route_p, hint, jpg}`. Older `runs/run_*.pkl` files come from Louie's tracker
  (`{n, tick, key, count, jpg, node, mode, matches, hint, lost}`).
* Stub SDK (put it on `PYTHONPATH` before `source/`):

```python
# stub/vis_nav_sdk/__init__.py
from enum import IntFlag
class Action(IntFlag):
    IDLE=0; FORWARD=1; BACKWARD=2; LEFT=4; RIGHT=8; CHECKIN=16; QUIT=32
WAIT = "WAIT"
class Agent: viewer = None
class Client: pass
class SimError(Exception): pass
def run(*a, **k): pass
def __getattr__(name): return object
```

* Build and inspect a map: `python source/tr_map.py <data dir>` (prints goal candidates and route lengths).
* Replay a real drive through the follower (the core tuning loop): load the pkl frames, build
  `Follower(m, m.route(goal)[0], K)`, feed each frame to `localise` + `stable_hint`, and count kinds, lost frames,
  jumps, and TURN-vs-FORWARD against the recorded keys.
  - Run it detached: `setsid nohup python3 replay.py > r.log 2>&1 < /dev/null &`, because one tool call is capped at
    300 s and a 1,700-frame replay takes ~3 min.
  - Settings can be overridden by `setattr(tr_follow, NAME, value)` after `importlib.reload`.
* Agent smoke test: build `NavAgent(data, cache)`, call `setup()` with noise targets and then with
  `tr_map.split_strip(target.jpg)`, give it a fake viewer whose `keys()` returns an object with `__contains__`, and
  feed route frames as observations (`SimpleNamespace(image, step, steps_left)`).

## 6. Workflow notes with Caleb

* He runs Windows PowerShell from the repo root. Downloads land in `~\Downloads`; copy them into `source\` with:
  `Get-ChildItem $HOME\Downloads\<name>*.py | Sort-Object LastWriteTime | Select-Object -Last 1 | Copy-Item -Destination source\<name>.py`
  (two runs failed because a file hadn't been copied yet).
* He is new to robot perception; explain the "why" in plain terms and keep instructions to exact commands.
* Every delivered file was compiled, linted (pyflakes) and smoke-tested against the stub before delivery; keep
  doing that. Then commit to GitHub so the other computers (and Louie) get it: all new `source/*.py`, `README.md`,
  `HANDOFF.md`.

## 7. Timeline to the challenge

* **Tue morning:** commit everything; live test on a fresh seed (item 1); replay and tune (item 2).
* **Tue afternoon/evening:** decide the attempts 2–3 approach (item 3); a second fresh-seed rehearsal; decide the
  agent for attempt 1 (`nav_agent` unless the rehearsals regress).
* **Tue night time-box:** if `nav_agent` isn't at least as good as `final_route_agent` + picker on fresh seeds, use
  the fallback.
* **Wed before 8 pm:** `git pull` on the challenge computer, run once on a local seed so the environment is warm,
  and check `.env`.
