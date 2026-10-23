import re
import string
from collections import Counter


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score(answer, references, dataset="hotpotqa"):
    pred = normalize(answer)
    scores = []
    for gold in references:
        gold = normalize(gold)
        em = float(pred == gold)
        overlap = sum((Counter(pred.split()) & Counter(gold.split())).values())
        if dataset in {"hotpotqa", "2wiki"} and pred != gold and (pred in {"yes", "no", "noanswer"} or gold in {"yes", "no", "noanswer"}):
            overlap = 0
        f1 = 2 * overlap / (len(pred.split()) + len(gold.split())) if overlap else 0.0
        scores.append((em*100, f1*100))
    return {"em":max((x[0] for x in scores), default=0), "f1":max((x[1] for x in scores), default=0)}


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
