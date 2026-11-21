"""Accept the current 2025 validation numbers as the regression baseline. Run only after a deliberate model change
that you have checked is an improvement (or an accepted trade-off), and say why in the commit message."""
import json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src, dst = ROOT / "results" / "validation_2025_latest.json", ROOT / "tests" / "validation_baseline.json"
old = json.load(open(dst)) if dst.exists() else {}
new = json.load(open(src))
for pos in new:
    c = new[pos][new[pos]["chosen"]]
    o = old.get(pos, {}).get(old.get(pos, {}).get("chosen", ""), {})
    print(f"{pos}: {old.get(pos, {}).get('chosen', '-')} MAE {o.get('MAE', '-')} rank {o.get('rank', '-')}  ->  {new[pos]['chosen']} MAE {c['MAE']} rank {c['rank']}")
shutil.copy(src, dst)
print("baseline updated:", dst)
