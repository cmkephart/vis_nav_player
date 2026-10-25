# Stable route agent (keyboard hints)

Camera-only agent for the NYU Visual Navigation Challenge. You drive with the keyboard; the agent tells you where you are
and which key to press next, along a route through the recorded exploration drives.

This is the version that reached the goal on every maze it was tested on (SPL about 1 of 4 points).

## Run (from the repo root)

    uv sync
    uv run python source/stable/route_agent.py --local 21        # any maze seed
    uv run python source/stable/route_agent.py --challenge <id>  # the real challenge (needs .env with the API key)

First start on a maze builds and caches the verified links (about 3-5 minutes); later starts take seconds.
Cached files go to `cache/` (git-ignored). The goal is found by voting (it prints `goal: node N by voting`).

## Files this needs

| file | what it is |
|---|---|
| `source/stable/route_agent.py` | route graph, verified shortcuts, goal vote |
| `source/stable/tracking_agent.py` | live position tracking (RootSIFT + RANSAC) and the hint text |
| `source/stable/baseline_agent.py` | copy of the course baseline with our small changes (kept here so it does not replace yours) |
| `source/vlad.py`, `source/cli.py`, `source/keyboard_agent.py` | course template files (already in the repo) |

Python packages: numpy, opencv-python-headless, networkx, scikit-learn, tqdm (all in the course `pyproject.toml`).

## Dials

    $env:LINK_LOOSENESS=1   # PowerShell. 1 = strict (default, recommended), 2/3 = more shortcuts, shorter but riskier routes

## Known limits

- Routes follow the recorded drives plus verified shortcuts, so they are far longer than the shortest path (low SPL).
- A false shortcut or a look-alike place can send the hints the wrong way; if the position estimate says "lost", look around.
- Later experiments (live goal candidates, N key, route table) live in `source/route_agent.py` / `source/tracking_agent.py` and are not part of this stable copy.
