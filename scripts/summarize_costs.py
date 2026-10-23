import argparse
import json
from statistics import mean
from active_graphrag.data import read_json

p=argparse.ArgumentParser()
p.add_argument("records")
args=p.parse_args()
rows=read_json(args.records)
if not rows:
    raise ValueError("No records")
result={"questions_and_seeds":len(rows),"preparation_seconds":mean(r["preparation_seconds"] for r in rows),
        "response_seconds":mean(r["seconds"] for r in rows),
        "mean_sampled_peak_gpu_gib":mean(r["sampled_peak_gpu_gib"] for r in rows)}
for name,stages in {"navigator":{"gate","navigation","fallback"},"extractor":{"extraction"}}.items():
    result[name+"_tokens_per_question"]=mean(sum(c["input_tokens"]+c["output_tokens"] for c in r["calls"] if c["stage"] in stages) for r in rows)
for stage in ["gate","navigation","extraction","fallback"]:
    calls=[c["seconds"] for r in rows for c in r["calls"] if c["stage"]==stage]
    result[stage+"_seconds_per_call"]=mean(calls) if calls else None
for stage in ["retrieval","fusion"]:
    values=[e[stage+"_seconds"] for r in rows for e in r["events"] if e["type"]=="explore"]
    result[stage+"_seconds_per_exploration"]=mean(values) if values else None
print(json.dumps(result,indent=2))
