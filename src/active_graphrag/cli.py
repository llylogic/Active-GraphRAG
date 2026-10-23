import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import yaml
from .schema import Config
from .data import prepare, load_examples, read_json, write_jsonl
from .evaluation import score, calibrate, scorer_id


def main():
    p = argparse.ArgumentParser(description="Active-GraphRAG evaluation")
    subs = p.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--source", required=True)
    prep.add_argument("--dataset", choices=["hotpotqa","2wiki","musique"], required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--calibration", type=int, default=200)
    prep.add_argument("--evaluation", type=int, default=1000)
    prep.add_argument("--aliases", help="Official 2Wiki id_aliases.json for the data_ids release")
    run = subs.add_parser("run")
    run.add_argument("--data", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--threshold-file")
    run.add_argument("--mode", choices=["full","always","closed"], default="full")
    run.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    cal = subs.add_parser("calibrate")
    cal.add_argument("--always", required=True)
    cal.add_argument("--closed", required=True)
    cal.add_argument("--output", required=True)
    scoring = subs.add_parser("score", help="Score saved answers with official benchmark functions; no model inference")
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--source", required=True, help="Original official development file")
    scoring.add_argument("--dataset", choices=["hotpotqa", "2wiki", "musique"], required=True)
    scoring.add_argument("--manifest", required=True, help="split_manifest.json saved during preparation")
    scoring.add_argument("--split", choices=["calibration", "evaluation"], default="evaluation")
    scoring.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    scoring.add_argument("--aliases")
    scoring.add_argument("--output", required=True, help="New output directory")
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.source,args.dataset,args.output,args.seed,args.calibration,args.evaluation,args.aliases)
        return
    if args.command == "score":
        from .scoring import evaluate_saved
        result = evaluate_saved(args.predictions, args.source, args.dataset, args.manifest,
                                args.output, args.split, args.seeds, args.aliases)
        print(json.dumps(result, indent=2))
        return
    if args.command == "calibrate":
        always, closed = read_json(args.always), read_json(args.closed)
        if not always or not closed:
            raise ValueError("Calibration needs nonempty prediction files")
        policies = set()
        for group in (always, closed):
            keys = [(r["id"], r["seed"]) for r in group]
            if len(keys) != len(set(keys)):
                raise ValueError("Duplicate calibration question/seed records")
            for r in group:
                if r.get("missing_prediction") or r.get("scorer") != scorer_id(r.get("dataset")):
                    raise ValueError("Run score on complete calibration predictions with the official scoring functions first")
                if not r.get("answer_reference_policy") or r["answer_reference_policy"] == "unspecified":
                    raise ValueError("Calibration requires a known answer-reference policy")
                policies.add((r["scorer"], r["answer_reference_policy"]))
        if len(policies) != 1:
            raise ValueError("Calibration datasets, scoring versions, or alias policies differ")
        ids = {r["id"] for r in always}
        if ids != {r["id"] for r in closed}:
            raise ValueError("Calibration question sets differ")
        expected_seeds = {r["seed"] for r in always}
        rows = []
        from .schema import Generation
        for qid in sorted(ids):
            a = [r for r in always if r["id"] == qid]
            c = [r for r in closed if r["id"] == qid]
            if {r["seed"] for r in a} != expected_seeds or {r["seed"] for r in c} != expected_seeds:
                raise ValueError("Calibration seed sets differ")
            if len({r["answer"] for r in c}) != 1:
                raise ValueError("Greedy gate candidates differ across seeds")
            candidate = Generation(**c[0]["gate"]) if c[0]["gate"] is not None else None
            rows.append(dict(id=qid, confidence=candidate.confidence if candidate else None,
                             gate_f1=sum(r["f1"] for r in c)/len(c), always_f1=sum(r["f1"] for r in a)/len(a)))
        result = calibrate(rows)
        result.update(scorer=next(iter(policies))[0], answer_reference_policy=next(iter(policies))[1])
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    from .models import BGEEmbedder, HuggingFaceModel
    from .pipeline import ActiveGraphRAG
    from .retrieval import chunk_passages, DenseRetriever
    from .profiling import GPUMemorySampler
    import torch
    config = Config(**yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    threshold = None
    if args.threshold_file:
        threshold = json.loads(Path(args.threshold_file).read_text())["threshold"]
    elif args.mode == "full":
        p.error("Full routing requires --threshold-file from calibration")
    examples = load_examples(args.data)
    if not examples or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Evaluation needs a nonempty dataset and unique seeds")
    if any(e.answer_reference_policy == "unspecified" for e in examples):
        raise ValueError("Re-run prepare to record the official answer-reference policy")
    if args.threshold_file:
        threshold_record = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))
        if any(threshold_record.get("scorer") != scorer_id(e.dataset) or
               threshold_record.get("answer_reference_policy") != e.answer_reference_policy for e in examples):
            raise ValueError("Recalibrate the threshold using the same official scorer and answer-reference policy")
    if len({e.id for e in examples}) != len(examples):
        raise ValueError("Duplicate question IDs")
    navigator = HuggingFaceModel(config.navigator, config.navigator_device, config.input_limit, config.output_limit, config.temperature, config.top_p)
    extractor = navigator if config.extractor == config.navigator else HuggingFaceModel(config.extractor, config.extractor_device, config.input_limit, config.output_limit, config.temperature, config.top_p)
    embedder = BGEEmbedder(config.embedding, config.embedding_device)
    pipeline = ActiveGraphRAG(navigator, extractor, embedder, config)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    metadata = dict(config=asdict(config), seeds=args.seeds, mode=args.mode, threshold=threshold,
                    scorers={e.dataset: scorer_id(e.dataset) for e in examples},
                    dataset_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                    torch=torch.__version__)
    (root/"config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    records = []
    with (root/"predictions.jsonl").open("w", encoding="utf-8") as out:
        for seed in args.seeds:
            for example in examples:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                begin = time.perf_counter()
                retriever = None
                if args.mode != "closed":
                    retriever = DenseRetriever(chunk_passages(example.passages, embedder.tokenizer, config.window, config.stride),embedder)
                preparation = time.perf_counter()-begin
                # Reset the decoding stream independently of index preparation.
                torch.manual_seed(seed)
                with GPUMemorySampler() as memory_sample:
                    result = pipeline.run(example.question, retriever, threshold, args.mode)
                record = dict(id=example.id, seed=seed, dataset=example.dataset, question=example.question,
                              scorer=scorer_id(example.dataset), answer_reference_policy=example.answer_reference_policy,
                              sampled_peak_gpu_gib=memory_sample.peak_bytes/(1024**3),
                              preparation_seconds=preparation, **result, **score(result["answer"],example.answers,example.dataset))
                out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n"); out.flush()
                records.append(record)
                print(f"{seed} {example.id} {record['reason']} F1={record['f1']:.2f}", flush=True)
    summary = {metric:float(np.mean([r[metric] for r in records])) for metric in ["em","f1","seconds","rounds"]}
    summary["per_seed"] = {str(s):{m:float(np.mean([r[m] for r in records if r["seed"]==s])) for m in ["em","f1"]} for s in args.seeds}
    (root/"summary.json").write_text(json.dumps(summary,indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
