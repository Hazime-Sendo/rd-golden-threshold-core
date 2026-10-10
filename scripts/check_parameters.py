"""Checks that the parameters used by src/reduced_rd.py and scripts/verify_all.py agree with data/parameters.json."""
import ast, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import reduced_rd as M

ref = json.load(open(ROOT / "data" / "parameters.json"))["parameters"]
ok = True
for name, P in (("src/reduced_rd.py", M.P), ("scripts/verify_all.py", None)):
    if P is None:                     # read the dict literal of verify_all.py without running it
        tree = ast.parse((ROOT / "scripts" / "verify_all.py").read_text())
        PHI = (1 + 5 ** 0.5) / 2
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "P":
                P = eval(compile(ast.Expression(node.value), "P", "eval"), {"dict": dict, "PHI": PHI})
    bad = {k: (P[k], ref[k]) for k in ref if abs(P[k] - ref[k]) > 1e-12}
    print(f"[{'PASS' if not bad else 'FAIL'}] parameters of {name} match data/parameters.json" + (f"  {bad}" if bad else ""))
    ok &= not bad
sys.exit(0 if ok else 1)
