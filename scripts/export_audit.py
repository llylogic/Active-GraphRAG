import argparse
import json
from pathlib import Path
from active_graphrag.data import read_json
from active_graphrag.experiments import audit_sample

p=argparse.ArgumentParser()
p.add_argument("--records",required=True)
p.add_argument("--output",required=True)
p.add_argument("--seed",type=int,default=42)
args=p.parse_args()
audit=audit_sample(read_json(args.records),args.seed)
Path(args.output).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
