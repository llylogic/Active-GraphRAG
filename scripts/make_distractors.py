import argparse
from dataclasses import asdict
from pathlib import Path
from active_graphrag.data import read_json, convert, load_examples, write_jsonl
from active_graphrag.experiments import nested_distractors

p=argparse.ArgumentParser()
p.add_argument("--evaluation",required=True)
p.add_argument("--raw-dev",required=True)
p.add_argument("--raw-train",required=True)
p.add_argument("--output",required=True)
args=p.parse_args()
gold={str(r["_id"]):{title for title,_ in r["supporting_facts"]} for r in read_json(args.raw_dev)}
pool=[p for r in read_json(args.raw_train) for p in convert(r,"hotpotqa").passages]
levels={k:[] for k in [8,15,20,25,30]}
for example in load_examples(args.evaluation):
    for k,e in nested_distractors(example,pool,gold[example.id]).items():
        levels[k].append(asdict(e))
for k,rows in levels.items():
    write_jsonl(Path(args.output)/f"distractors_{k}.jsonl",rows)
