# vis_nav_player — team Blue Collar Mechanics

NYU Robot Perception mid-term (Caleb Kephart, Louie Baloh), built on the course starter kit
([tutorial](https://visual-navigation-challenge.ai4ce.dev/)). A robot sits in a maze; our code sees its camera and
gets four photos taken at the goal. A person drives with the arrow keys while the agent gives hints. Near the goal
the agent takes over, lines the robot up with the goal photo and says READY; the driver presses SPACE.

**Graded challenge:** Wed 10/28 8 pm – Thu 10/29 8 am ET, challenge id `da90a443e215`, 3 attempts
(an attempt is spent as soon as an agent connects). Report due Fri 10/30 8 pm.

**Score /10:** distance ≤ 0.1/0.15/0.2 m → 3/2/1, heading ≤ 10/15/20° → 3/2/1 (both 0 beyond 0.2 m / 20°),
SPL (shortest route ÷ our ticks) ≥ 0.95/0.85/0.75/0.5 → 4/3/2/1. Time is not scored.

---

## 1. Set up a computer (once)

```powershell
irm https://astral.sh/uv/install.ps1 | iex      # Windows; macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/cmkephart/vis_nav_player.git
cd vis_nav_player
uv sync                                         # Python, packages and the local simulator
copy .env.example .env                          # then fill in VIS_NAV_API_KEY, and VIS_NAV_CHALLENGE=da90a443e215
```

Run everything **from the repo root**, e.g. `uv run source/nav_agent.py --local 42`. The first start on any maze
builds caches in `cache/` (3–5 min). `data/`, `cache/` and `runs/` are git-ignored, so every computer builds its own
caches the first time.

## 2. Which agent to run

| agent | phase A (getting there) | when to use |
|---|---|---|
| **`source/nav_agent.py`** | **teach-and-repeat route hints** (new, recommended) | attempt 1 on an unseen maze |
| `source/final_route_agent.py` | Louie's route tracker + goal-candidate picker | fallback (reached maze 21: 6/10) |
| `source/final_agent.py` | kit baseline (VLAD) hints | second fallback (first drives: 6/10) |

All three share the same **phase B** (`final_approach.py`): final alignment, READY, guarded SPACE.

```powershell
uv run source/nav_agent.py --local 42      # rehearse on local maze 42 (no attempt spent, nothing recorded)
uv run source/nav_agent.py                 # THE CHALLENGE (uses VIS_NAV_CHALLENGE from .env; spends an attempt)
```

Options for all agents:

- `--probe`: phase B measures only, never moves.
- `--auto-checkin`: phase B checks in by itself (not recommended).
- `--single-space`: no double-SPACE guard.

Options for `nav_agent.py` only:

- `--no-open`: don't pop up the candidate picture.
- `--handoff N`: how close to the end of the route (in route frames) phase B starts measuring every frame.

## 3. Driving `nav_agent.py`

1. **Wait** for the cache build (first start on a maze only). Before the real session the SDK runs a quick check
   with random-static "goal photos". The agent ignores it and prints `setup ... skipped`; that is normal.
2. **Choose the goal.**
   - The bottom strip shows up to 5 *candidate* places from the exploration data, each beside the goal photo it
     resembles. `logs/goal_candidates.png` opens with the same pairs.
   - **Candidate 1 is only the strongest match, not necessarily the goal.** The maze reuses posters, and a lone
     matching poster scores high (on maze 21 the real goal was #2). Pick the candidate whose *neighbouring* panels
     match too.
   - Tap **1–5**, after clicking the simulator window so it has the keyboard. The console confirms with
     `nav: goal = candidate N` or `(candidate N is already the goal)`.
   - You can switch any time later, and the choice is remembered for the next attempt.
3. **Drive** with the arrows.
   - The top line (`>>>`) is the one instruction to follow. It can say:
     - `TURN LEFT/RIGHT ~n presses`
     - `FORWARD ~n presses, then turn LEFT`
     - `TURN AROUND`
     - `TURN … NOW`
     - `NOT SURE WHERE YOU ARE`: turn slowly on the spot until the YOU ARE HERE picture matches; the hints come
       back by themselves.
     - `NEAR THE GOAL: look for the goal view`
     - `GOAL AHEAD: let go`
   - The strip shows **YOU ARE HERE** (the route frame you match), **NEXT TURN … starts here**, **…after the turn**
     and **GOAL**. Use the pictures as much as the text.
   - The status line shows route progress, ticks left and match strength.
4. **Near the goal, let go of the keys.**
   - Phase B aligns and prints `READY -- press SPACE`.
   - If the view is not clearly the goal view, nudge with the arrows and let go again.
   - SPACE before READY is blocked once; press it again within 3 s to force a check-in.

Every drive is saved to `runs/nav_<time>.pkl` (all frames, keys and hints). **Keep these files**: they are how
the hints get tuned, because a drive can be replayed offline frame by frame.

## 4. How the new phase A works

**`tr_map.py`** runs offline and caches its results:

- It computes RootSIFT features for every exploration frame, and the turn between consecutive frames (from a
  pure-rotation homography).
- It finds **same-pose links**: two frames taken from the same spot. A pure rotation must explain the matches,
  with parallax ≤ 6 px and matches spread across the view.
- The route graph is the recorded drives, plus same-pose links, plus "all drives start at the spawn". **No link
  is ever made from merely seeing a place**, so gaps in walls that you can see through but not drive through never
  become routes.
- Goal candidates: every frame is scored against each goal view **separately**. (Summing over views picked a
  wrong place on maze 21.)

**`tr_follow.py`** runs live:

- It tracks your position **along the chosen route only**, never against the rest of the maze, so look-alike
  places elsewhere cannot make it jump. Each frame gets a thumbnail pre-filter plus a SIFT check against nearby
  route frames, and a sparse look further ahead so you can skip the explorer's loops.
- A far jump needs 2 confirmations and ≤ 25 px of parallax.
- Hints steer toward the route point 0.25 m ahead, so the explorer's wiggles average out. A new hint must repeat
  before it replaces the one shown.

**Trade-off:** the route follows where the explorer actually drove, so it is safe but longer than the shortest
path (maze 21: 2,579 ticks vs 532 optimal; maze 42: 1,992 vs 571). If you cut corners, the follower keeps up.

## 5. Results so far (local mazes)

| maze | agent | result |
|---|---|---|
| 21 (unseen then) | final_route_agent + picker | 6/10, 0.01 m, 1°, SPL 0.30 |
| 21 (3rd drive, known) | nav_agent | 6/10, 0.04 m, 1°, SPL 0.30 |
| **42 (unseen)** | **nav_agent** | **6/10, 0.07 m, 1°, SPL 0.34**; hints "much better than before" |
| 202 (route known) | final_agent | 9/10, SPL 1.00 (0.11 m) |
| 7 (route known) | final_agent | 9/10, SPL 0.90 |
| 58, 101, 314, 303, 202 (first drives) | final_agent | 6/10 each, SPL 0.11–0.32 |

Lesson: distance and heading (6 points) are reliable. SPL points have only come from driving a route the driver
already knew.

## 6. Repository map (`source/`)

| file | what | status |
|---|---|---|
| `nav_agent.py` | new agent: picker + teach-and-repeat hints + phase B | current |
| `tr_map.py` | offline route map (features, turns, same-pose links, goal candidates, routes) | current |
| `tr_follow.py` | live route follower + hint text | current |
| `goal_picker.py` | candidate picture, number keys, static-noise check | shared by nav_agent and final_route_agent |
| `final_approach.py` | **phase B** | frozen |
| `final_agent.py` | baseline hints + phase B | frozen fallback |
| `final_route_agent.py` | Louie's route agent + phase B + picker | fallback |
| `route_agent.py`, `tracking_agent.py` | Louie's phase A (route graph, tracker) | Louie |
| `baseline_agent.py` | kit baseline, **modified by Louie** (`MIN_SHORTCUT_GAP=250`, `PREVIEW_NODES=10`; `final_agent.py` restores 50/5) | kit/Louie |
| `keyboard_agent.py`, `cli.py`, `vlad.py`, `my_agent.py` | course kit | kit |
| `offline_probe.py` | match all exploration frames vs goal photos; `--views logs --show N…` saves comparisons | tool |
| `export_truth.py` | zip a local maze's data + simulator truth (grid, goal, oracle) for offline work | tool |
| `sim_probe.py` | print what the local simulator knows (grid, oracle steps, noise) | tool |
| `calibrate.py` | phase B vs the true pose (local) | tool |
| `goal_richness.py`, `auto_agent.py`, `test_geometry.py`, `rl/` | experiments / kit extras | not used |

`deadreckon_check.py` is obsolete (superseded by `sim_probe.py`); delete it if you have it. `PROJECT_STATE.md`
and `PHASE_A_HANDOFF.md` are historical; `HANDOFF.md` is current.

## 7. For Louie (or anyone plugging in their own phase A)

**Reuse the map and follower as a library:**

```python
m = tr_map.build(data_dir, cache_dir)
tr_map.score_goal(m, targets, cache_dir)
cands = m.goal_candidates()
route, ticks = m.route(goal_record)
f = tr_follow.Follower(m, route, K)
# then, for each new camera frame:
f.localise(image)
hint = f.stable_hint()
```

`f.remaining_records()` tells phase B when to start measuring every frame.

**Pitfalls we hit:**

- **Phase B integration:** copy `_phase_b` and `_space_guard` from `nav_agent.py` (or `final_route_agent.py`).
  Phase B's moves are already `(Action, n)` tuples or a bare `Action.CHECKIN`; never wrap them again.
- **Keys:** `viewer.keys()` returns a `Keys` object. Use `"up" in keys`. It is **not iterable**; it has `held`,
  `tapped` and `any` parts, and a quick tap only shows up in `tapped`.
- **Pre-flight setup:** each session starts with a `setup()` whose goal photos are random static. Skip it with
  `goal_picker.is_noise`, or your caches and votes get polluted.
- **First camera frame:** it is static noise too; skip it.

## 8. Facts about the world (measured)

- **Camera:** 320×240 BGR, K = [[92,0,160],[0,92,120],[0,0,1]] (~120° wide).
- **Motion:** turn 3.44°/tick (90° ≈ 26 ticks); drive ~0.029 m/tick. Motion noise (0.3, 0.1).
- **Mazes:** local mazes are a 31×31 grid of 0.2 m cells, with corridors 1 cell wide and some 3-wide rooms.
- **Exploration data:** 3 drives from the spawn, one frame per 5 ticks. The action labels are *lossy*: a "LEFT"
  record may hold 3 or 5 ticks of turning, and "IDLE" records still rotate ~10°. Maps from dead reckoning
  therefore fail; use the images.
- **Goal photos** only match from within ~0.3 m.
- **Session limits:** silence timeout 120 s, 3600 s total.

## 9. Challenge-day checklist

1. `git pull`, and check that `.env` has `VIS_NAV_CHALLENGE=da90a443e215`.
2. `uv run source/nav_agent.py`: it downloads the data and builds the cache **before** connecting (3–5 min).
3. Choose the goal from the picture, drive, let go near the goal, and press SPACE at READY.
4. After each attempt, copy `runs/nav_*.pkl` and `logs/` somewhere safe (for the report and for attempts 2–3).
5. Attempts 2–3: drive the route you now know as directly as you can. SPL points come from these.

## 10. Hand-in zip (flat)

`nav_agent.py, tr_map.py, tr_follow.py, goal_picker.py, final_approach.py, keyboard_agent.py, cli.py`, plus
`final_route_agent.py, route_agent.py, tracking_agent.py, final_agent.py, baseline_agent.py, vlad.py` if a fallback
agent was used. Not `source/files.zip` (old snapshot).
