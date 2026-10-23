# Official answer evaluation

The answer-evaluation adapter calls benchmark authors' scoring functions bundled in `src/active_graphrag/_official`. It does not implement a separate normalization or token-overlap algorithm.

| Dataset | Official source | Pinned revision | Entry point |
| --- | --- | --- | --- |
| HotpotQA | [hotpot_evaluate_v1.py](https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py) | `3635853403a8735609ee997664e1528f4480762a` | `update_answer` |
| 2WikiMultiHopQA | [2wikimultihop_evaluate_v1.1.py](https://github.com/Alab-NII/2wikimultihop/blob/13800e5be57df1b4040b9b1588c6c811779e69e9/2wikimultihop_evaluate_v1.1.py) | `13800e5be57df1b4040b9b1588c6c811779e69e9` | `update_answer` |
| MuSiQue-Ans | [metrics/answer.py](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/answer.py) | `922ac98f19a201998dbdae6d7f2887a5258dbdeb` | `AnswerMetric` |

HotpotQA and 2Wiki files are copied unchanged under package-safe filenames. MuSiQue's `metrics/answer.py` and `metrics/metric.py` are bundled; the import of `Metric` is changed to a package-relative import. Their scoring logic is unchanged. Original and packaged SHA-256 digests are recorded in `_official/SOURCES.json`.

HotpotQA and 2Wiki source is distributed under Apache License 2.0; MuSiQue source is distributed under Creative Commons Attribution 4.0 International as provided in the respective repositories. Their license texts are retained alongside the modules. Attribution belongs to the authors and contributors of those benchmark repositories. MuSiQue's answer module also acknowledges the AllenNLP SQuAD tools from which it derives.

The adapter reports answer EM/F1 in percent and averages unrounded per-question values across the fixed subsets and seeds. It does not apply the presentation rounding used by the standalone 2Wiki and MuSiQue scripts before averaging. HotpotQA uses one canonical reference. 2Wiki's `data_ids` release uses canonical answers plus official aliases and demonyms, while its legacy data uses canonical answers. MuSiQue-Ans uses the supplied canonical answers and answer aliases. Reference data are used only for evaluation, never as retrieval or model input.
