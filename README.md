# Active-GraphRAG

Training-free multi-hop question answering with confidence routing, on-demand triple extraction, persistent graph memory, and graph-guided exploration and pruning.

## Installation

Python 3.10 or newer is required. Install a PyTorch build appropriate for your CUDA runtime, then:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -e ".[models,experiments,test]"
pytest
```

The default configuration places Llama-3-8B-Instruct on GPU 0, Qwen2.5-1.5B-Instruct on GPU 1, and BGE-M3 on CPU. Model access and licenses are managed by their publishers. Authenticate with Hugging Face before loading gated Llama weights. Models can also be specified as local directories in `configs/default.yaml`. No training is performed.

## Data and evaluation

Obtain the official development data from [HotpotQA](https://hotpotqa.github.io/), [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop), and [MuSiQue](https://github.com/StonyBrookNLP/musique). Dataset files and model weights are not bundled. Preserve each dataset's terms of use.

```bash
active-graphrag prepare --source data/hotpot_dev_distractor_v1.json --dataset hotpotqa --output data/hotpotqa
active-graphrag prepare --source data/2wiki/dev.json --dataset 2wiki --output data/2wiki
active-graphrag prepare --source data/musique_ans_v1.0_dev.jsonl --dataset musique --output data/musique
```

Preparation saves disjoint sets of 200 calibration and 1,000 evaluation questions, together with the selected identifiers and source SHA-256. Split selection is deterministic. Gold answers are used only for scoring. Supporting annotations are never passed to the retriever or models.

Run both calibration conditions with the same three seeds, then select the routing threshold:

```bash
active-graphrag run --config configs/default.yaml --data data/hotpotqa/calibration.jsonl --mode always --output runs/cal-always
active-graphrag run --config configs/default.yaml --data data/hotpotqa/calibration.jsonl --mode closed --output runs/cal-closed
active-graphrag calibrate --always runs/cal-always/predictions.jsonl --closed runs/cal-closed/predictions.jsonl --output runs/hotpotqa-threshold.json
active-graphrag run --config configs/default.yaml --data data/hotpotqa/evaluation.jsonl --threshold-file runs/hotpotqa-threshold.json --output runs/eval
```

Repeat calibration separately for each dataset. A null threshold disables bypassing. Thresholds maximize the number of bypassed questions subject to a mean loss of at most one F1 percentage point across the entire calibration set. Calibration averages always-retrieve scores over the decoding seeds. `--mode always` and `--mode closed` provide routing controls. Default generation seeds are 42, 43, and 44.

Outputs include one JSONL record per question and seed, answer-level EM/F1, terminal graphs, extraction records, entity-alignment decisions, pruning events, input/output token counts, and stage timings. Empty responses and context overflow remain in the denominator. `summary.json` reports means across runs. The internal answer scorer implements normalized EM/token F1;Answer-level EM and token-level F1 are computed from the returned answer field using the normalization and scoring functions in `evaluation.py`.

## Method

* Gate: greedy answer-only decoding, 64-token limit. Confidence is the geometric mean of answer-token probabilities from raw full-vocabulary logits, excluding prompt and EOS tokens. Empty, UNKNOWN and truncated responses cannot bypass retrieval.
* Retrieval: independent 500-token paragraph windows with stride 50, normalized BGE-M3 dense embeddings and FAISS cosine search, up to three passages per call. Repeated retrievals count toward the budget.
* Extraction: passage-by-passage open triples, string-field validation, whitespace trimming, lowercase relation labels and relation whitespace normalization.
* Fusion: NetworkX directed multigraph, cosine alignment against the fixed pre-update node set, thresholds 0.85/0.65, exact mapped-triple deduplication, separate soft-alias metadata. Identifier order resolves ties.
* Navigation: the initial action must explore; at most four retrievals; at most one prune between retrievals. Pruning removes a node, incident edges and references from alias metadata. Invalid actions or exhausted budgets invoke evidence-based fallback. All Llama prompts are checked after chat templating against 7,168 input tokens.

Prompts are in `src/active_graphrag/prompts.py`. All graph operations occur in the backend. YAML is the navigator's serialized view and is never parsed to perform graph mutation.

## Experiment interfaces

`experiments.py` provides paired YAML/JSON/triple-list inputs, matched-token retrieved text, nested TF-IDF distractor pools and human-audit sampling. Representation comparisons retain node identifiers, relation direction and aliases. Token matching excludes questions failing the text-length condition under any seed. Audit annotations must be completed before computing empirical audit rates.

For horizon sensitivity, create configurations with `max_rounds` from 1 to 6 and reuse the threshold calibrated at four rounds. For component switches: set `extractor` equal to `navigator` for symmetric extraction, `graceful: false` for forced completion, `stateful: false` to retain only the current local graph, or `reflection: false` to select exploration from the question and action history without graph-based reflection or pruning. These operational definitions are explicit in the configuration and controller.

This repository contains Active-GraphRAG, its confidence-routing controls, and configuration files for component ablations and retrieval-budget experiments. Baseline systems are maintained in their respective repositories.

## Development

```bash
pytest -q
python -m compileall -q src
```

The unit tests use deterministic fixtures to verify controller constraints, graph updates, and scoring. Model inference is implemented through Transformers, with BGE-M3 providing dense embeddings.
