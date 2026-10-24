import sys, types, math
import numpy as np

# stand-in for the course SDK so the module imports outside the kit
stub = types.ModuleType("vis_nav_sdk")
class Action:
    FORWARD, BACKWARD, LEFT, RIGHT, CHECKIN, IDLE = "FORWARD", "BACKWARD", "LEFT", "RIGHT", "CHECKIN", "IDLE"
stub.Action = Action
sys.modules["vis_nav_sdk"] = stub
import final_approach as fa

rng = np.random.default_rng(0)
W, H, F = 320, 240, 160.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])

def Ry(phi):  # positive phi = turn RIGHT (y axis points down)
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def project(X, C, yaw_left_deg):
    R_body = Ry(-math.radians(yaw_left_deg))
    Xc = (X - C) @ R_body            # = R_body^T (X - C) per row
    uv = (Xc @ K.T)
    return uv[:, :2] / uv[:, 2:3], Xc[:, 2]

def scene(n=400):
    return np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-0.6, 0.6, n), rng.uniform(1.5, 4, n)])

def run(C2, yaw_left, noise=0.3):
    X = scene()
    p1, z1 = project(X, np.zeros(3), 0.0)
    p2, z2 = project(X, np.array(C2), yaw_left)
    ok = (z1 > .1) & (z2 > .1) & (abs(p1[:, 0] - W/2) < W/2) & (abs(p1[:, 1] - H/2) < H/2) \
         & (abs(p2[:, 0] - W/2) < W/2) & (abs(p2[:, 1] - H/2) < H/2)
    p1, p2 = p1[ok] + rng.normal(0, noise, (ok.sum(), 2)), p2[ok] + rng.normal(0, noise, (ok.sum(), 2))
    return fa.pose_from_points(p1, p2, K, 1.0, 4.0)

print("case                                  yaw_left  bearing_right  forward  parallax  inliers")
cases = [
    ("goal 0.8 ahead, 0.3 right, turned L15", (0.3, 0, 0.8), 15),
    ("goal 0.8 ahead, 0.3 left,  turned R10", (-0.3, 0, 0.8), -10),
    ("goal 0.5 BEHIND,           turned L 5", (0.0, 0, -0.5), 5),
    ("on the spot, turned L20 (pure rotation)", (0.0, 0, 0.0), 20),
    ("0.05 m off,  turned 0", (0.03, 0, 0.04), 0),
]
for name, C, yaw in cases:
    r = run(C, yaw)
    print(f"{name:40s}", "None" if r is None else "  ".join(f"{v:8.2f}" for v in r))

# controller decisions on hand-made estimates
fa_obj = fa.FinalApproach(K, [np.zeros((240, 320, 3), np.uint8)] * 4)
def E(**kw):
    d = dict(view=0, matches=50, inliers=40, yaw_left=0, yaw_front=0, bearing_right=0, forward=1, parallax=10)
    d.update(kw); return fa.Estimate(**d)
print()
for label, e in [
    ("goal 30deg right, far",      E(bearing_right=30)),
    ("goal 30deg left, far",       E(bearing_right=-30)),
    ("goal dead ahead, far",       E(bearing_right=3)),
    ("goal right behind, far",     E(bearing_right=175, forward=-1)),
    ("goal behind-left, far",      E(bearing_right=-150, forward=-.8)),
    ("on spot, facing 25 left of front (turn LEFT)",  E(parallax=1, yaw_front=25)),
    ("on spot, facing 25 right of front (turn RIGHT)", E(parallax=1, yaw_front=-25)),
    ("on spot, matched LEFT view, facing it exactly",  E(parallax=1, view=1, yaw_left=0, yaw_front=fa.wrap(0-90))),
    ("on spot, aligned",           E(parallax=1)),
    ("on spot, aligned, weak",     E(parallax=1, inliers=18)),
]:
    print(f"{label:50s}", fa_obj.decide(e))
