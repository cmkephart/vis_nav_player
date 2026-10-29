# REPORT HANDOFF — writing the mid-term report (Blue Collar Mechanics)

For a new chat whose job is the **report**, not the code. The code is frozen (GitHub
`cmkephart/vis_nav_player`, branch `master`). Companion files: `LESSONS.md` (how everything works and why, from
scratch), `README.md` (how to run it), `HANDOFF.md` (engineering state).

**Course:** NYU Robot Perception mid-term. **Team:** Caleb Kephart, Louie Baloh (Louie also built his own agent
independently; this report covers the team agent described here).
**Graded challenge:** Wed 10/28 8 pm – Thu 10/29 8 am ET, id `da90a443e215`, 3 attempts.
**Report due: Fri 10/30 8 pm.**
**Format we know of:** slides; a problem slide plus references, and **at most 2 slides for the method**. Confirm the
exact requirements on the course site before writing.

> **To fill in after Wednesday:** the graded challenge results (points, distance, heading, SPL per attempt, the
> route length printed at start of each attempt, whether learning and drift trim engaged). Keep the console output
> and `runs/nav_*.pkl` of each attempt; they are the evidence.

---

## 1. The problem (for the problem slide)

* A simulated robot in an unknown maze sees only its camera: 320×240, ~120° field of view, K = [[92,0,160],[0,92,120],[0,0,1]].
* It gets **exploration data**: 3 recorded drives from the spawn, about 3,000 frames with action labels. It also gets
  **4 photos taken at the goal** (front, left, back, right).
* Actions: forward/back about 0.029 m per tick; turns 3.44° per tick; check-in.
* **Score /10:**
  - distance ≤ 0.1 / 0.15 / 0.2 m → 3/2/1;
  - heading ≤ 10 / 15 / 20° → 3/2/1 (both 0 beyond 0.2 m or 20°);
  - **SPL** = oracle shortest-route ticks ÷ our ticks, ≥ 0.95 / 0.85 / 0.75 / 0.5 → 4/3/2/1. Time is not scored.
* **What makes it hard (measured, good for the slide):**
  - **Repeated posters:** the same textures appear in many places, so single-image place recognition lies. On maze
    21 a look-alike scored 97 feature matches against 50 for the real goal.
  - **Noisy, biased actuation:** motion noise (0.3, 0.1); on the course server, driving forward also turns the robot
    1.2–2.4° left per tick.
  - **Lossy exploration labels:** a "LEFT" record holds 3 or 5 ticks of turning, and "IDLE" records still rotate
    ~10°. So maps built from the recorded actions (dead reckoning) fail.
  - **Narrow grid maze:** 0.2 m corridors, many dead ends, and wall gaps you can see through but not drive through.
  - **Monocular:** no metric depth.
  - **Strategy chosen:** a human drives with the arrow keys; the agent provides perception (where you are, which way
    to go, final alignment). This was decided early: the robot is driven by hand, and our contribution is perception.

## 2. Method in one paragraph (draft for the method slides)

**Phase A, getting there (teach-and-repeat):**
- From the exploration data, the agent builds a route graph made **only of places the explorer physically drove
  through**. Drives are joined where two frames were taken from the **same position**: a pure-rotation homography
  explains the RootSIFT matches with ≤ 6 px of residual parallax.
- **The driver picks the goal** from candidate places ranked by per-view homography matches against the goal photos.
- Live, the agent tracks the driver's position **along that one route only**, using a thumbnail pre-filter plus
  SIFT/RANSAC verification, so look-alikes elsewhere can't fool it. It gives one steady instruction, steering toward
  the route point 0.25 m ahead.
- On later attempts the agent **learns from the driver's earlier drives**, which join the graph as extra drives and
  shorten the route. It also **cancels the server's forward drift**, measured from frame-to-frame rotation, by
  mixing turn bits into forward ticks at no tick cost.

**Phase B, the finish:**
- Near the goal, RootSIFT matches against the 4 goal views give the heading error (a rotation homography, H = K R
  K⁻¹) and the direction to the goal (essential matrix + recoverPose).
- A small controller probes, aligns and steps toward the goal, rejects look-alikes with a probe turn, and says READY;
  the driver confirms with SPACE.
- A cross-check with phase A stops phase B taking over at a look-alike while the route says the goal is far.

## 3. The story arc (what changed, and why; for the slides or the speaker notes)

1. **Kit baseline** (VLAD place recognition + graph + turn hints): every first drive reached the goal with phase B
   (6/10), but SPL was 0.11–0.32.
2. **Original phase A (Louie's route tracker):** a global SIFT tracker against all exploration frames, a goal vote
   that summed matches across views, and turn-by-turn text hints from about 100 tuned constants.
   - **Failed on unseen maze 21:** the vote picked a look-alike; the hint text changed on 450 of 657 updates; the
     position jumped more than 30 nodes 14 times; the driver never found the goal.
3. **Diagnosis:**
   - Offline matching showed the vote's mistake. A human sees the real goal immediately from the *neighbouring*
     panels, which led to the **goal picker**.
   - With the picker, the old hints reached the goal (6/10, SPL 0.30), but they were still confusing.
4. **Ruled out, with evidence:**
   - **Dead-reckoning map:** the motion noise and lossy labels make it sprawl beyond the 6×6 m maze.
   - **ORB-SLAM:** no metric scale, loses track in turns on the spot, a C++ build on Windows, and no time.
   - **DINO:** retrieval wasn't the bottleneck; geometry was.
   - **Layout spread to tell the goal from look-alikes:** 4 vs 3–4 strips, no separation.
5. **New phase A, teach-and-repeat:**
   - Unseen maze 42: 6/10, SPL 0.34, "much better hints".
   - Replaying real drives offline led to three fixes: turn hints no longer cut corners, far jumps need a parallax
     check, and the lost message near the goal is clearer.
6. **Unseen maze 77:** 6/10, SPL 0.44. The replay found three more problems, each fixed: a **look-alike READY**
   (now gated by the route), **lost after turning away** (now covered by turn memory), and **dead ends** (now
   warned about).
   - The key discovery: the driver's own drive bridged explorer passes that never share a pose. **Learning from
     earlier drives** cut maze 77's route from 3,534 to 1,426 ticks; drive 2 scored 7/10, SPL 0.52.
7. **Full rehearsal on maze 88:** drive 1 ~8/10; drive 2 (learned) **10/10, SPL 1.00**.
8. **Online development challenge:**
   - Trial 1: 7/10, SPL 0.68, but the robot veered. Measured: forward ticks turn it 1.2–2.4° left on the server
     (0.07° locally). That led to **drift trim**.
   - Trial 2 (learned, trim on): **10/10, SPL 0.96**.

## 4. Results table (all local runs report "attempt 0"; the online ones are real server sessions)

| maze | agent | points | distance | heading | SPL | notes |
|---|---|---|---|---|---|---|
| 58, 101, 314, 303, 202 (first drives) | kit baseline + phase B | 6 each | — | — | 0.11–0.32 | phase B carried every finish |
| 202 (route known) | baseline + phase B | 9 | 0.11 m | — | 1.00 | memory gives SPL |
| 7 (route known) | baseline + phase B | 9 | — | — | 0.90 | |
| 21 (unseen) | Louie's tracker | 0 | — | — | — | wrong goal; never found it |
| 21 | Louie's tracker + goal picker | 6 | 0.01 m | 1° | 0.30 | oracle 532 ticks |
| 42 (unseen) | nav_agent | 6 | 0.07 m | 1° | 0.34 | oracle 571 |
| 77 (unseen) | nav_agent | 6 | 0.02 m | 1° | 0.44 | oracle 1,009; one look-alike READY |
| 77, drive 2 | nav_agent + learning | **7** | 0.02 m | 1° | 0.52 | route 3,534 → 1,410 ticks |
| 88 (unseen), drive 1 | nav_agent | ~8 | — | — | ≈0.72 | 783 steps, oracle 562 |
| 88, drive 2 | nav_agent + learning | **10** | 0.04 m | 0° | 1.00 | 437 steps |
| online dev challenge, attempt 1 | nav_agent | 7 | 0.09 m | 1° | 0.68 | server drift felt |
| online dev challenge, attempt 2 | nav_agent + learning + drift trim | **10** | 0.05 m | 3° | 0.96 | route 2,512 → 782 ticks |
| **graded challenge (Wed)** | | | | | | **fill in** |

**Key numbers for figures:**
- Route length vs oracle: maze 21 recorded route 2,579 vs 532; maze 42 1,992 vs 571; maze 77 3,534 → 1,426 with
  learning, vs 1,009.
- Hint stability: Louie's tracker changed the hint on 68% of updates. The new follower's tuning (false TURN hints
  while the driver correctly drove forward) went 291 → 158 steps on the same replayed drive.
- Lost time on the maze-77 replay: 579 → 318 steps "not sure", plus 283 steps of turn guidance.
- Drift: server forward ticks +1.65° median (IQR −1.1..+3.0); local +0.07°. Turn ticks 3.40–3.44° on both.
- Phase B accuracy: ~0.5° median heading error near the goal; finishes at 0.002–0.09 m.

## 5. Figures available or easy to make

* `logs/goal_candidates.png` (and maze-21 `compare_node1420.png` vs `compare_node2305.png`): look-alike vs real
  goal. **Best single figure for "repeated textures".**
* `logs/sim_probe_21.png`: dead-reckoned paths sprawling (why no metric map).
* A screenshot of the viewer during a drive: hint line, YOU ARE HERE / NEXT TURN / GOAL strip.
* A bar chart of route length (exploration only vs learned vs oracle) for mazes 77, 88 and the dev challenge.
* A drift histogram (forward-tick yaw, server vs local): the numbers are in §4; the raw frames are in `runs/`.
* Before/after hint stability (68% changes vs the follower), and the per-run results table.

## 6. References (verify the details before citing)

- D. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," IJCV 2004 (SIFT).
- R. Arandjelović, A. Zisserman, "Three things everyone should know to improve object retrieval," CVPR 2012 (RootSIFT).
- M. Fischler, R. Bolles, "Random Sample Consensus," CACM 1981 (RANSAC).
- R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., 2004 (homography H = K R K⁻¹,
  essential matrix).
- D. Nistér, "An efficient solution to the five-point relative pose problem," TPAMI 2004 (essential matrix).
- H. Jégou et al., "Aggregating local descriptors into a compact image representation," CVPR 2010 (VLAD; kit baseline).
- P. Furgale, T. Barfoot, "Visual teach and repeat for long-range rover autonomy," J. Field Robotics 2010 (teach-and-repeat).
- P. Anderson et al., "On Evaluation of Embodied Navigation Agents," arXiv 2018 (SPL).
- R. Mur-Artal, J. M. M. Montiel, J. D. Tardós, "ORB-SLAM," T-RO 2015 (considered, not used).
- M. Oquab et al., "DINOv2," 2023 (considered, not used).
- The course starter kit / tutorial: https://visual-navigation-challenge.ai4ce.dev/

## 7. Honesty notes for the write-up

* A human drives. The agent's contribution is perception and guidance: place recognition, goal candidates, route
  tracking, drift estimation, final alignment. Say so plainly.
* SPL depends heavily on attempt 1's drive, because learning reuses it. Repeat drives with route knowledge score far
  higher than first drives.
* The goal choice is human. Automatic ranking can't separate the goal from a look-alike in this maze (measured).
* Local runs aren't the graded server: the server adds forward drift; local mazes have none.
* Parameters were tuned by replaying recorded drives offline (no simulator in the development sandbox). The
  evidence is the replay statistics above, not a large benchmark: 3 unseen local mazes plus 1 online challenge.

## 8. Where the material is

* Repo `README.md` §4–§5 (method and results), `HANDOFF.md` §2–§3 (decisions with evidence), and `LESSONS.md`
  (full explanation).
* `logs/results.txt` (every run's result line), `logs/*.png`, `logs/phase_b.csv`.
* `runs/nav_*.pkl` (frames, keys and hints of each drive; on Caleb's challenge computer, not in git).
* Course attempt pages: `https://visual-navigation-challenge.ai4ce.dev/challenges/<id>/attempts/<session>`.
