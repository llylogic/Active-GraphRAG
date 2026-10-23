import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from .schema import Example, Passage


def read_json(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")


def convert(row, dataset):
    if dataset not in {"hotpotqa", "2wiki", "musique"}:
        raise ValueError(dataset)
    passages = []
    if dataset == "musique":
        contexts = [(str(p["title"]), p["paragraph_text"]) for p in row["paragraphs"]]
        answers = [row["answer"]] + row.get("answer_aliases", [])
    else:
        contexts = [(title, " ".join(sentences)) for title, sentences in row["context"]]
        answers = [row["answer"]]
    for title, text in contexts:
        pid = hashlib.sha256((title+"\n"+text).encode()).hexdigest()[:24]
        passages.append(Passage(pid, title, text))
    return Example(str(row.get("_id", row.get("id"))), row["question"], passages, answers, dataset)


def load_examples(path):
    return [Example(**{**r, "passages":[Passage(**p) for p in r["passages"]]}) for r in read_json(path)]


def prepare(source, dataset, directory, seed=42, calibration=200, evaluation=1000):
    examples = [convert(r, dataset) for r in read_json(source)]
    if len({e.id for e in examples}) != len(examples):
        raise ValueError("Duplicate question identifiers")
    if len(examples) < calibration + evaluation:
        raise ValueError("Insufficient examples for disjoint requested subsets")
    random.Random(seed).shuffle(examples)
    root = Path(directory)
    write_jsonl(root/"calibration.jsonl", (asdict(e) for e in examples[:calibration]))
    write_jsonl(root/"evaluation.jsonl", (asdict(e) for e in examples[calibration:calibration+evaluation]))
    manifest = dict(source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(), split_seed=seed,
                    calibration_ids=[e.id for e in examples[:calibration]],
                    evaluation_ids=[e.id for e in examples[calibration:calibration+evaluation]])
    (root/"split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
