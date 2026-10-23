import json
from functools import lru_cache
from pathlib import Path
from ._official import hotpot, twowiki
from ._official.musique.answer import AnswerMetric


normalize = hotpot.normalize_answer


@lru_cache(maxsize=None)
def scorer_id(dataset):
    paths = {"hotpotqa": "hotpot.py", "2wiki": "twowiki.py", "musique": "musique/answer.py"}
    if dataset not in paths:
        raise ValueError(f"Unsupported dataset: {dataset}")
    sources = json.loads((Path(__file__).parent / "_official/SOURCES.json").read_text(encoding="utf-8"))
    source = next(s for s in sources if s["package_path"] == paths[dataset])
    return f"{dataset}:{source['commit']}:{source['upstream_path']}"


def score(answer, references, dataset="hotpotqa"):
    """Compute answer-only percentages with the dataset's official functions."""
    scorer_id(dataset)
    if not isinstance(answer, str):
        raise TypeError("The returned answer field must be a string")
    if not isinstance(references, (list, tuple)) or not references or not all(isinstance(r, str) for r in references):
        raise ValueError("At least one string reference answer is required")
    if dataset == "musique":
        metric = AnswerMetric()
        metric(answer, references)
        em, f1 = metric.get_metric()
    else:
        metrics = dict(em=0.0, f1=0.0, prec=0.0, recall=0.0)
        if dataset == "hotpotqa":
            if len(references) != 1:
                raise ValueError("Official HotpotQA evaluation uses one gold answer per question")
            hotpot.update_answer(metrics, answer, references[0])
        else:
            twowiki.update_answer(metrics, answer, references)
        em, f1 = metrics["em"], metrics["f1"]
    return {"em": float(em) * 100, "f1": float(f1) * 100}


def calibrate(rows, max_loss=1.0):
    """Rows contain one fixed gate candidate and mean always-retrieve F1 per question."""
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Calibration needs unique question records")
    candidates = sorted({r["confidence"] for r in rows if r["confidence"] is not None})
    for threshold in candidates:
        bypass = [r for r in rows if r["confidence"] is not None and r["confidence"] >= threshold]
        loss = sum(r["always_f1"]-r["gate_f1"] for r in bypass)/len(rows)
        if bypass and loss <= max_loss + 1e-12:
            return dict(threshold=threshold, bypass_count=len(bypass), mean_f1_loss=loss, questions=len(rows))
    return dict(threshold=None, bypass_count=0, mean_f1_loss=0, questions=len(rows))
