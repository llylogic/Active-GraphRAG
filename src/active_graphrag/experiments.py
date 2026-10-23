"""Paired representation comparisons, nested distractors, and audit export."""
from dataclasses import asdict, replace
import random
from .graph import serialize
from .pipeline import answer_field
from .prompts import FINAL, messages


def representation_inputs(records, tokenizer):
    """Exclude a question from token matching if any seed lacks sufficient text."""
    rows, eligible = [], {}
    for record in records:
        graph = record["graph"]
        if not graph["nodes"]:
            eligible[record["id"]] = False
            continue
        formats = {kind:serialize(graph, kind) for kind in ["yaml","json","triples"]}
        seen, passages = set(), []
        for event in record["events"]:
            for item in event.get("extraction", []):
                p = item["passage"]
                if p["id"] not in seen:
                    seen.add(p["id"])
                    passages.append(p["text"])
        text_ids = tokenizer.encode("\n\n".join(passages), add_special_tokens=False)
        yaml_count = len(tokenizer.encode(formats["yaml"], add_special_tokens=False))
        matched = tokenizer.decode(text_ids[:yaml_count], skip_special_tokens=True)
        ok = len(text_ids) >= yaml_count and len(tokenizer.encode(matched, add_special_tokens=False)) == yaml_count
        eligible[record["id"]] = eligible.get(record["id"], True) and ok
        rows.append(dict(id=record["id"], seed=record["seed"], question=record["question"], formats=formats,
                         matched_text=matched, evidence_tokens=yaml_count))
    for row in rows:
        row["paired_text_eligible"] = eligible[row["id"]]
    return rows


def evaluate_representations(rows, model, set_seed):
    import json
    results = []
    for row in rows:
        conditions = dict(row["formats"])
        if row["paired_text_eligible"]:
            conditions["matched_text"] = row["matched_text"]
        for name, evidence in conditions.items():
            set_seed(row["seed"])
            output = model.generate(messages(FINAL, json.dumps({"question":row["question"],"evidence":evidence},ensure_ascii=False)))
            results.append(dict(id=row["id"],seed=row["seed"],condition=name,answer=answer_field(output.text),
                                paired_text_eligible=row["paired_text_eligible"],input_tokens=output.input_tokens))
    return results


def nested_distractors(example, training_passages, supporting_titles, levels=(8,15,20,25,30)):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(example.passages) != 10 or min(levels) < 8:
        raise ValueError("HotpotQA pools must start with ten original paragraphs")
    unique = {}
    for p in training_passages:
        unique.setdefault((p.title,p.text),p)
    original = {(p.title,p.text) for p in example.passages}
    pool = [p for key,p in sorted(unique.items()) if key not in original and p.title not in supporting_titles]
    if len(pool) < max(levels)-8:
        raise ValueError("Insufficient eligible distractors")
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([p.text for p in pool])
    scores = (matrix @ vectorizer.transform([example.question]).T).toarray().ravel()
    order = sorted(range(len(pool)),key=lambda i:(-scores[i],pool[i].id))
    return {level:replace(example,passages=example.passages+[pool[i] for i in order[:level-8]]) for level in levels}


def audit_sample(records, seed=42, question_count=100, entity_count=200):
    rng = random.Random(seed)
    by_id = {}
    for row in records:
        if row["events"]:
            by_id.setdefault(row["id"],row)
    if len(by_id) < question_count:
        raise ValueError("Insufficient graph-routed questions for the audit")
    selected = rng.sample(sorted(by_id), question_count)
    extraction, alignment, pruning = [], [], []
    for qid in selected:
        row = by_id[qid]
        items = [x for event in row["events"] for x in event.get("extraction",[])]
        if items:
            extraction.append(dict(id=qid, **rng.choice(items), reference_triples=None))
    for qid,row in sorted(by_id.items()):
        for index,event in enumerate(row["events"]):
            alignment.extend(dict(id=qid,event=index,**a,correct_merge=None,merge_opportunity=None) for a in event.get("alignment",[]))
            if event["type"] == "prune":
                pruning.append(dict(id=qid,event=index,**event,removes_unsupported=None,removes_support=None))
    if len(alignment) < entity_count:
        raise ValueError("Insufficient entity decisions")
    return dict(extraction=extraction,alignment=rng.sample(alignment,entity_count),pruning=pruning)


def pruning_rates(annotations):
    if not annotations or any(not isinstance(r.get(k),bool) for r in annotations for k in ["removes_unsupported","removes_support"]):
        raise ValueError("Complete all event-level Boolean annotations")
    return {k:dict(count=sum(r[k] for r in annotations),total=len(annotations),percent=100*sum(r[k] for r in annotations)/len(annotations))
            for k in ["removes_unsupported","removes_support"]}
