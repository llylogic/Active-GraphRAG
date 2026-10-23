"""Generate component-ablation and retrieval-horizon configuration files."""
import argparse
from pathlib import Path
import yaml

p = argparse.ArgumentParser()
p.add_argument("--base", default="configs/default.yaml")
p.add_argument("--output", default="configs/experiments")
args = p.parse_args()
base = yaml.safe_load(Path(args.base).read_text())
root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
variants = {"full":{}, "symmetric_extraction":{"extractor":base["navigator"]},
            "forced_completion":{"graceful":False}, "no_stateful_fusion":{"stateful":False},
            "no_topology_reflection":{"reflection":False}}
variants.update({f"horizon_{k}":{"max_rounds":k} for k in range(1,7)})
for name, changes in variants.items():
    (root/f"{name}.yaml").write_text(yaml.safe_dump({**base, **changes},sort_keys=False),encoding="utf-8")
