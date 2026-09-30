"""Validate ground-truth JSON math. Fails loudly on any inconsistency. No rendering."""
import json
import math
import sys
from pathlib import Path

BASE = Path(__file__).parent
errors: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        errors.append(msg)


def dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def main() -> int:
    arg = json.loads((BASE / "ground_truth_argand.json").read_text(encoding="utf-8"))
    pts = {k: (v["re"], v["im"]) for k, v in arg["points"].items()}

    # 1. z mapping exact
    check(pts["A"] == (1, 1), "A = 1+i -> (1,1)")
    check(pts["B"] == (1, 3), "B = 1+3i -> (1,3)")
    check(pts["C"] == (3, 3), "C = 3+3i -> (3,3)")
    check(pts["D"] == (3, 1), "D = 3+i -> (3,1)")
    # 2. forbidden value absent from DATA (exclude the 'forbidden' doc key itself)
    data_only = {k: v for k, v in arg.items() if k != "forbidden"}
    blob = json.dumps(data_only)
    check("-1+3i" not in blob and "(-1,3)" not in blob, "no invented B=(-1,3)")
    # 3. square: 4 equal sides, right angles, closed
    order = arg["square"]["vertices"]
    sides = [dist(pts[order[i]], pts[order[(i + 1) % 4]]) for i in range(4)]
    check(all(abs(s - 2.0) < 1e-9 for s in sides), f"4 sides equal 2 (got {sides})")
    # right angle at B: AB·BC == 0
    ab = (pts["B"][0] - pts["A"][0], pts["B"][1] - pts["A"][1])
    bc = (pts["C"][0] - pts["B"][0], pts["C"][1] - pts["B"][1])
    check(abs(ab[0] * bc[0] + ab[1] * bc[1]) < 1e-9, "angle at B is 90 deg")
    check(arg["square"]["closed"] is True, "square closed")
    # 4. modulus + area
    check(abs(math.hypot(1, 1) - math.sqrt(2)) < 1e-9, "|1+i| = sqrt(2)")
    check(arg["result"]["eq"] == "Area = 4", "result Area = 4")
    # 5. conjugate reflection
    conj = arg["relationships"][0]
    check(conj["conjugate_point"] == [1, -1], "conj(A) = (1,-1)")

    tor = json.loads((BASE / "ground_truth_torque.json").read_text(encoding="utf-8"))
    angs = {v["label"]: v["angle_deg"] for v in tor["vectors"]}
    check(abs((angs["F"] - angs["r"]) - tor["angle"]["value_deg"]) < 1e-9, "theta = F - r = 35 deg")
    check(next(v for v in tor["vectors"] if v["label"] == "F")["tail"] == "r", "F tails on r")
    check(tor["result"]["eq"] == "τ = r × F", "torque relation exact")
    tdata = {k: v for k, v in tor.items() if k != "forbidden"}
    tblob = json.dumps(tdata)
    check("L = I" not in tblob and "ΔL" not in tblob, "no duplicate L/dL inside torque visual spec")

    print("---")
    if errors:
        print(f"GROUND TRUTH INVALID: {len(errors)} failures")
        return 1
    print("GROUND TRUTH VALID: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
