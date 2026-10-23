"""Score saved answer fields on the exact question subset recorded at preparation."""
import hashlib
import json
from pathlib import Path
from statistics import mean

from .data import answer_references, load_aliases, read_json, reference_policy, write_jsonl
from .evaluation import score, scorer_id


def evaluate_saved(predictions, source, dataset, manifest_path, output, split="evaluation",
                   seeds=(42, 43, 44), aliases_path=None):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if split not in {"calibration", "evaluation"}:
        raise ValueError("Unknown split")
    scorer = scorer_id(dataset)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("dataset", dataset) != dataset:
        raise ValueError("Dataset differs from the split manifest")
    source_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if source_hash != manifest["source_sha256"]:
        raise ValueError("Gold source SHA-256 differs from the split manifest")
    ids = manifest[f"{split}_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("The split must contain unique, nonempty question identifiers")
    if aliases_path is not None and dataset != "2wiki":
        raise ValueError("--aliases is only used for 2WikiMultiHopQA")
    aliases = load_aliases(aliases_path)
    alias_hash = hashlib.sha256(Path(aliases_path).read_bytes()).hexdigest() if aliases_path else None
    if "aliases_sha256" in manifest and alias_hash != manifest["aliases_sha256"]:
        raise ValueError("Aliases SHA-256 differs from the split manifest")
    gold = {}
    for row in read_json(source):
        qid = str(row.get("_id", row.get("id")))
        if qid in gold:
            raise ValueError(f"Duplicate gold question: {qid}")
        gold[qid] = row
    if set(ids) - gold.keys():
        raise ValueError("The source is missing questions from the split manifest")
    references = {qid: answer_references(gold[qid], dataset, aliases) for qid in ids}
    records = {}
    id_set, seed_set = set(ids), set(seeds)
    for row in read_json(predictions):
        if row["id"] not in id_set or row["seed"] not in seed_set:
            raise ValueError("Predictions contain a question or seed outside the requested subset")
        if row.get("dataset", dataset) != dataset:
            raise ValueError("Prediction dataset differs from the requested dataset")
        key = (row["seed"], row["id"])
        if key in records:
            raise ValueError(f"Duplicate prediction: {key}")
        records[key] = row
    scored = []
    for seed in seeds:
        for qid in ids:
            key = (seed, qid)
            row = dict(records.get(key, {}))
            missing = key not in records or "answer" not in row or row.get("missing_prediction", False)
            answer = row.get("answer", "")
            if "em" in row and "f1" in row:
                row["previous_em"], row["previous_f1"] = row["em"], row["f1"]
            row.update(id=qid, seed=seed, dataset=dataset, answer=answer,
                       missing_prediction=missing, scorer=scorer,
                       answer_reference_policy=reference_policy(dataset, aliases),
                       **score(answer, references[qid], dataset))
            scored.append(row)
    summary = {metric: mean(row[metric] for row in scored) for metric in ("em", "f1")}
    summary.update(questions=len(ids), seeds=seeds, records=len(scored), scorer=scorer,
                   answer_reference_policy=reference_policy(dataset, aliases),
                   source_sha256=source_hash, aliases_sha256=alias_hash,
                   missing_predictions=sum(row["missing_prediction"] for row in scored),
                   predictions_sha256=hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
                   split=split, per_seed={})
    for seed in seeds:
        rows = [row for row in scored if row["seed"] == seed]
        summary["per_seed"][str(seed)] = {metric: mean(row[metric] for row in rows) for metric in ("em", "f1")}
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_jsonl(root / "predictions.jsonl", scored)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
