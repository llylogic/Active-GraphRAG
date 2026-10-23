from dataclasses import asdict
import json
import pytest
from active_graphrag.data import convert, prepare, load_examples
from active_graphrag.experiments import pruning_rates, representation_inputs


def test_data_does_not_expose_annotations():
    row={"_id":"q", "question":"q?", "answer":"secret", "context":[["T",["sentence."]]],"supporting_facts":[["T",0]]}
    e=convert(row,"hotpotqa")
    assert "secret" not in json.dumps([asdict(p) for p in e.passages])
    assert "supporting" not in json.dumps(asdict(e))


def test_splits_disjoint_and_repeatable(tmp_path):
    rows=[{"_id":str(i),"question":"q?","answer":"a","context":[]} for i in range(5)]
    source=tmp_path/"raw.json"; source.write_text(json.dumps(rows))
    for name in ["a","b"]:
        prepare(source,"hotpotqa",tmp_path/name,calibration=2,evaluation=3)
    cal=load_examples(tmp_path/"a/calibration.jsonl")
    ev=load_examples(tmp_path/"a/evaluation.jsonl")
    assert not {e.id for e in cal}&{e.id for e in ev}
    assert (tmp_path/"a/evaluation.jsonl").read_bytes()==(tmp_path/"b/evaluation.jsonl").read_bytes()


def test_pruning_events_can_count_both():
    result=pruning_rates([dict(removes_unsupported=True,removes_support=True)])
    assert all(r["percent"]==100 for r in result.values())
    with pytest.raises(ValueError):
        pruning_rates([dict(removes_unsupported=None,removes_support=False)])
