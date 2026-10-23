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


def load_aliases(path):
    if path is None:
        return None
    aliases = {}
    for row in read_json(path):
        qid = row["Q_id"]
        if qid in aliases:
            raise ValueError(f"Duplicate alias identifier: {qid}")
        values = row["aliases"] + row["demonyms"]
        if not all(isinstance(value, str) for value in values):
            raise ValueError("Answer aliases and demonyms must be strings")
        aliases[qid] = sorted(set(values))
    return aliases


def answer_references(row, dataset, aliases=None):
    if dataset == "musique":
        if row.get("answerable") is False:
            raise ValueError("Use MuSiQue-Ans: unanswerable MuSiQue-Full rows are not supported")
        return [row["answer"]] + row.get("answer_aliases", [])
    if dataset == "2wiki":
        if "answer_id" in row and aliases is None:
            raise ValueError("2Wiki data with answer_id requires --aliases id_aliases.json")
        if aliases is not None and "answer_id" not in row:
            raise ValueError("2Wiki alias evaluation requires the official data_ids release with answer_id")
        return [row["answer"]] + (aliases.get(row["answer_id"], []) if aliases is not None else [])
    if dataset == "hotpotqa":
        return [row["answer"]]
    raise ValueError(f"Unsupported dataset: {dataset}")


def reference_policy(dataset, aliases=None):
    if dataset == "musique":
        return "canonical_and_answer_aliases"
    if dataset == "2wiki" and aliases is not None:
        return "canonical_aliases_and_demonyms"
    return "canonical_only"


def convert(row, dataset, aliases=None):
    if dataset not in {"hotpotqa", "2wiki", "musique"}:
        raise ValueError(dataset)
    passages = []
    if dataset == "musique":
        contexts = [(str(p["title"]), p["paragraph_text"]) for p in row["paragraphs"]]
    else:
        contexts = [(title, " ".join(sentences)) for title, sentences in row["context"]]
    answers = answer_references(row, dataset, aliases)
    for title, text in contexts:
        pid = hashlib.sha256((title+"\n"+text).encode()).hexdigest()[:24]
        passages.append(Passage(pid, title, text))
    return Example(str(row.get("_id", row.get("id"))), row["question"], passages, answers, dataset,
                   reference_policy(dataset, aliases))


def load_examples(path):
    return [Example(**{**r, "passages":[Passage(**p) for p in r["passages"]]}) for r in read_json(path)]


def prepare(source, dataset, directory, seed=42, calibration=200, evaluation=1000, aliases_path=None):
    if aliases_path is not None and dataset != "2wiki":
        raise ValueError("--aliases is only used for 2WikiMultiHopQA")
    aliases = load_aliases(aliases_path)
    examples = [convert(r, dataset, aliases) for r in read_json(source)]
    if len({e.id for e in examples}) != len(examples):
        raise ValueError("Duplicate question identifiers")
    if len(examples) < calibration + evaluation:
        raise ValueError("Insufficient examples for disjoint requested subsets")
    random.Random(seed).shuffle(examples)
    root = Path(directory)
    write_jsonl(root/"calibration.jsonl", (asdict(e) for e in examples[:calibration]))
    write_jsonl(root/"evaluation.jsonl", (asdict(e) for e in examples[calibration:calibration+evaluation]))
    manifest = dict(source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(), split_seed=seed, dataset=dataset,
                    calibration_ids=[e.id for e in examples[:calibration]],
                    evaluation_ids=[e.id for e in examples[calibration:calibration+evaluation]])
    if aliases_path is not None:
        manifest["aliases_sha256"] = hashlib.sha256(Path(aliases_path).read_bytes()).hexdigest()
    (root/"split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
