import argparse
import json
from pathlib import Path
import torch
import yaml
from active_graphrag.models import HuggingFaceModel
from active_graphrag.schema import Config
from active_graphrag.data import read_json, write_jsonl
from active_graphrag.experiments import representation_inputs, evaluate_representations

p=argparse.ArgumentParser()
p.add_argument("--records",required=True)
p.add_argument("--config",default="configs/default.yaml")
p.add_argument("--output",required=True)
args=p.parse_args()
c=Config(**yaml.safe_load(Path(args.config).read_text()))
model=HuggingFaceModel(c.navigator,c.navigator_device,c.input_limit,c.output_limit,c.temperature,c.top_p)
rows=representation_inputs(read_json(args.records),model.tokenizer)
write_jsonl(args.output,evaluate_representations(rows,model,torch.manual_seed))
