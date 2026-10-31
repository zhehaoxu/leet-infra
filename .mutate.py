"""Generic mutation harness: .mutate.py <slug> <muts.json>
muts.json = [[name, old, new], ...]
"""
import sys, json, os, types

slug, muts_path = sys.argv[1], sys.argv[2]
base = f"problems/pytorch/{slug}/variants/pytorch"
sys.path.insert(0, base)
import test_cases as tc  # noqa: E402

src = open(f"{base}/solution.py").read()
checks = [v for k, v in sorted(tc.__dict__.items()) if k.startswith("check_")]
muts = json.load(open(muts_path))
for name, old, new in muts:
    if old not in src:
        print(f"{name}: PATTERN NOT FOUND")
        continue
    mod = types.ModuleType("m")
    exec(compile(src.replace(old, new), "<mut>", "exec"), mod.__dict__)
    caught = False
    try:
        for c in checks:
            c(mod.solve)
    except Exception:
        caught = True
    print(f"{name}: {'CAUGHT' if caught else '*** SURVIVED ***'}")
