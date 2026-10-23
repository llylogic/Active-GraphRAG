import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from active_graphrag import evaluation
from active_graphrag.data import convert, load_aliases, load_examples, prepare, read_json
from active_graphrag.evaluation import score, scorer_id
from active_graphrag.scoring import evaluate_saved


@pytest.mark.parametrize("dataset", ["hotpotqa", "2wiki", "musique"])
@pytest.mark.parametrize("prediction,gold,em,f1", [
    ("The Belfast!", "belfast", 100, 100),
    ("Northern", "Northern Ireland", 0, 200 / 3),
    ("red red", "red blue", 0, 50),
    ("", "Belfast", 0, 0),
    ("wrong", "Belfast", 0, 0),
])
def test_answer_metrics(dataset, prediction, gold, em, f1):
    result = score(prediction, [gold], dataset)
    assert result == pytest.approx(dict(em=em, f1=f1))


def test_dataset_specific_empty_and_yes_no_rules():
    for dataset in ("hotpotqa", "2wiki"):
        assert score("yes indeed", ["yes"], dataset)["f1"] == 0
        assert score("", ["the"], dataset) == dict(em=100, f1=0)
    assert score("yes indeed", ["yes"], "musique")["f1"] == pytest.approx(200 / 3)
    assert score("", ["the"], "musique") == dict(em=100, f1=100)
    assert score("NYC", ["New York City", "NYC"], "musique") == dict(em=100, f1=100)
    with pytest.raises(ValueError):
        score("x", ["x"], "unknown")
    with pytest.raises(ValueError):
        score("x", [])
    with pytest.raises(TypeError):
        score(None, ["x"])


def test_pinned_sources_and_licenses_are_packaged():
    root = Path(evaluation.__file__).parent / "_official"
    sources = json.loads((root / "SOURCES.json").read_text())
    for source in sources:
        assert hashlib.sha256((root / source["package_path"]).read_bytes()).hexdigest() == source["packaged_sha256"]
        assert len(source["commit"]) == 40
    assert len(list(root.glob("LICENSE-*.txt"))) == 3


def test_2wiki_aliases_are_scoring_only(tmp_path):
    row = dict(_id="q", question="Where?", answer="United Kingdom", answer_id="Q145",
               context=[["Place", ["Belfast is a city."]]])
    alias_file = tmp_path / "aliases.json"
    alias_file.write_text(json.dumps(dict(Q_id="Q145", aliases=["UK"], demonyms=["British"])) + "\n")
    aliases = load_aliases(alias_file)
    example = convert(row, "2wiki", aliases)
    assert score("British", example.answers, "2wiki")["em"] == 100
    assert "British" not in str(example.passages) and "UK" not in example.question
    with pytest.raises(ValueError, match="requires --aliases"):
        convert(row, "2wiki")
    with pytest.raises(ValueError, match="answer_id"):
        convert({k: v for k, v in row.items() if k != "answer_id"}, "2wiki", aliases)
    assert convert({k: v for k, v in row.items() if k != "answer_id"}, "2wiki").answer_reference_policy == "canonical_only"


def make_files(tmp_path, dataset):
    rows = []
    for i, answer in enumerate(["Belfast", "Northern Ireland", "London"]):
        if dataset == "musique":
            rows.append(dict(id=str(i), question="q?", answer=answer, answerable=True,
                             answer_aliases=[], paragraphs=[]))
        else:
            rows.append(dict(_id=str(i), question="q?", answer=answer, context=[]))
    source = tmp_path / "gold.json"
    source.write_text(json.dumps(rows))
    prepared = tmp_path / "prepared"
    prepare(source, dataset, prepared, calibration=1, evaluation=2)
    examples = load_examples(prepared / "evaluation.jsonl")
    # Deliberately omit one question in each seed: the manifest must retain it.
    predictions = tmp_path / "predictions.jsonl"
    records = [dict(id=examples[0].id, seed=s, answer=examples[0].answers[0], em=0, f1=0) for s in [42, 43]]
    predictions.write_text("\n".join(json.dumps(r) for r in records))
    return source, prepared / "split_manifest.json", predictions, examples


@pytest.mark.parametrize("dataset", ["hotpotqa", "2wiki", "musique"])
def test_saved_predictions_keep_missing_answers_and_seeds(tmp_path, dataset):
    source, manifest, predictions, examples = make_files(tmp_path, dataset)
    output = tmp_path / "scored"
    result = evaluate_saved(predictions, source, dataset, manifest, output, seeds=[42, 43])
    assert result["questions"] == 2 and result["records"] == 4
    assert result["missing_predictions"] == 2
    assert result["em"] == result["f1"] == 50
    assert result["per_seed"] == {"42": {"em": 50, "f1": 50}, "43": {"em": 50, "f1": 50}}
    scored = read_json(output / "predictions.jsonl")
    assert all(r["answer"] == "" for r in scored if r["missing_prediction"])
    assert all(r["scorer"] == scorer_id(dataset) for r in scored)
    assert all(r["previous_f1"] == 0 for r in scored if not r["missing_prediction"])


def test_scoring_rejects_duplicates_wrong_source_and_extra_ids(tmp_path):
    source, manifest, predictions, _ = make_files(tmp_path, "hotpotqa")
    rows = read_json(predictions)
    predictions.write_text("\n".join(json.dumps(r) for r in rows + [rows[0]]))
    with pytest.raises(ValueError, match="Duplicate prediction"):
        evaluate_saved(predictions, source, "hotpotqa", manifest, tmp_path / "dup", seeds=[42, 43])
    rows[0]["id"] = "outside"
    predictions.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="outside"):
        evaluate_saved(predictions, source, "hotpotqa", manifest, tmp_path / "extra", seeds=[42, 43])
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_saved(predictions, source, "hotpotqa", manifest, tmp_path / "wrong", seeds=[42, 43])


def test_score_cli_does_not_load_models(tmp_path):
    source, manifest, predictions, _ = make_files(tmp_path, "hotpotqa")
    command = [sys.executable, "-m", "active_graphrag.cli", "score", "--source", str(source),
               "--predictions", str(predictions), "--manifest", str(manifest), "--dataset", "hotpotqa",
               "--seeds", "42", "43", "--output", str(tmp_path / "cli")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["em"] == 50


def test_2wiki_prepare_and_rescore_use_same_alias_file(tmp_path):
    rows = [dict(_id=str(i), question="q?", answer="United Kingdom", answer_id="Q145", context=[]) for i in range(3)]
    source = tmp_path / "gold.json"
    source.write_text(json.dumps(rows))
    aliases = tmp_path / "id_aliases.json"
    aliases.write_text(json.dumps(dict(Q_id="Q145", aliases=["UK"], demonyms=["British"])) + "\n")
    prepared = tmp_path / "prepared"
    prepare(source, "2wiki", prepared, calibration=1, evaluation=2, aliases_path=aliases)
    examples = load_examples(prepared / "evaluation.jsonl")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("\n".join(json.dumps(dict(id=e.id, seed=42, answer="British")) for e in examples))
    result = evaluate_saved(predictions, source, "2wiki", prepared / "split_manifest.json", tmp_path / "scores",
                            seeds=[42], aliases_path=aliases)
    assert result["em"] == result["f1"] == 100
    assert result["answer_reference_policy"] == "canonical_aliases_and_demonyms"
    with pytest.raises(ValueError, match="Aliases SHA-256"):
        evaluate_saved(predictions, source, "2wiki", prepared / "split_manifest.json", tmp_path / "bad", seeds=[42])


def test_calibration_rejects_unversioned_scores(tmp_path):
    records = tmp_path / "old.jsonl"
    records.write_text(json.dumps(dict(id="q", seed=42, dataset="hotpotqa", answer="a", f1=100)))
    result = subprocess.run([sys.executable, "-m", "active_graphrag.cli", "calibrate", "--always", str(records),
                             "--closed", str(records), "--output", str(tmp_path / "threshold.json")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "official scoring functions first" in result.stderr


def test_official_calibration_and_uneven_seeds(tmp_path):
    records = []
    for qid in ("q1", "q2"):
        for seed in (42, 43, 44):
            records.append(dict(id=qid, seed=seed, dataset="hotpotqa", answer="Belfast",
                                gate=dict(text="Belfast", logprobs=[-0.1]),
                                scorer=scorer_id("hotpotqa"), answer_reference_policy="canonical_only",
                                **score("Belfast", ["Belfast"], "hotpotqa")))
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "threshold.json"
    predictions.write_text("\n".join(json.dumps(r) for r in records))
    command = [sys.executable, "-m", "active_graphrag.cli", "calibrate", "--always", str(predictions),
               "--closed", str(predictions), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    threshold = json.loads(output.read_text())
    assert threshold["scorer"] == scorer_id("hotpotqa")
    assert threshold["bypass_count"] == 2 and threshold["mean_f1_loss"] == 0
    assert threshold["threshold"] == pytest.approx(0.9048374180359595)
    predictions.write_text("\n".join(json.dumps(r) for r in records[:-1]))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Calibration seed sets differ" in result.stderr
