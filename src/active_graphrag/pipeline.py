import json
import re
import time
from dataclasses import asdict
from . import prompts
from .graph import GraphMemory, parse_triples
from .schema import ContextOverflow


def parse_action(text):
    match = re.fullmatch(r"\s*\[(Missing_Link|Dead_End|Path_Complete)\]\s*<(Explore|Prune|Answer):\s*(.*?)>\s*", text, re.S)
    if not match:
        return None
    assessment, action, value = match.groups()
    if dict(Missing_Link="Explore", Dead_End="Prune", Path_Complete="Answer")[assessment] != action or not value.strip():
        return None
    return assessment, action, value.strip()


def answer_field(text):
    try:
        obj = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))
        answer = obj.get("answer", "") if isinstance(obj, dict) else ""
        return answer.strip() if isinstance(answer, str) and answer.strip().upper() != "UNKNOWN" else ""
    except ValueError:
        return ""


class ActiveGraphRAG:
    def __init__(self, navigator, extractor, embedder, config):
        self.navigator, self.extractor, self.embedder, self.config = navigator, extractor, embedder, config

    def run(self, question, retriever, threshold=None, mode="full", gate=None):
        if mode not in {"full", "always", "closed"}:
            raise ValueError(mode)
        memory = GraphMemory(self.embedder, self.config.strong, self.config.weak)
        history, events, calls = [], [], []
        rounds, last_prune = 0, False
        start = time.perf_counter()
        def generate(model, instruction, content, stage, greedy=False):
            begin = time.perf_counter()
            output = model.generate(prompts.messages(instruction, content), gate=greedy)
            calls.append(dict(stage=stage, seconds=time.perf_counter()-begin, input_tokens=output.input_tokens,
                              output_tokens=output.output_tokens, text=output.text, truncated=output.truncated))
            return output
        def finish(answer, reason):
            return dict(answer=answer, reason=reason, rounds=rounds, graph=memory.snapshot(), history=history,
                        events=events, calls=calls, seconds=time.perf_counter()-start,
                        gate=asdict(gate) if gate is not None else None)
        def fallback(reason):
            output = generate(self.navigator, prompts.FALLBACK if self.config.graceful else prompts.FORCED,
                              json.dumps({"question":question,"graph":memory.serialize()}, ensure_ascii=False), "fallback")
            return finish(answer_field(output.text), reason)
        try:
            if mode != "always":
                if gate is None:
                    gate = generate(self.navigator, prompts.GATE, question, "gate", True)
                if mode == "closed":
                    return finish(gate.text if gate.valid_gate else "", "closed_book")
                if gate.valid_gate and threshold is not None and gate.confidence >= threshold:
                    return finish(gate.text, "bypass")
            while True:
                instruction = prompts.NAVIGATE
                if not self.config.reflection:
                    instruction = "Select a retrieval query for unanswered aspects of the question using the action history. Output [Missing_Link] <Explore: query>, or [Path_Complete] <Answer: answer>. Do not prune."
                state = {"question":question, "history":history, "retrieval_rounds":rounds,
                         "retrieval_limit":self.config.max_rounds, "must_explore":rounds == 0 or last_prune}
                if self.config.reflection:
                    state["graph"] = memory.serialize()
                output = generate(self.navigator, instruction, json.dumps(state, ensure_ascii=False), "navigation")
                parsed = parse_action(output.text) if not output.truncated else None
                if parsed is None:
                    return fallback("invalid_action")
                assessment, action, value = parsed
                if rounds == 0 and action != "Explore":
                    return fallback("initial_action_not_explore")
                if action == "Answer":
                    return finish("" if value.upper() == "UNKNOWN" else value, "answer")
                if rounds >= self.config.max_rounds:
                    return fallback("budget_exhausted")
                if action == "Prune":
                    if last_prune or value not in memory.graph or not self.config.reflection:
                        return fallback("inadmissible_prune")
                    before = memory.snapshot()
                    removed = memory.prune(value)
                    events.append(dict(type="prune", target=value, removed=removed, before=before))
                    last_prune = True
                else:
                    begin = time.perf_counter()
                    passages = retriever.search(value, self.config.top_k)
                    retrieval_seconds = time.perf_counter()-begin
                    rounds += 1
                    triples, extraction = [], []
                    for passage in passages:
                        output = generate(self.extractor, prompts.EXTRACT, json.dumps({"title":passage.title,"passage":passage.text}, ensure_ascii=False), "extraction")
                        local = parse_triples(output.text) if not output.truncated else []
                        extraction.append(dict(passage=asdict(passage), triples=local))
                        triples.extend(t for t in local if t not in triples)
                    if not self.config.stateful:
                        memory = GraphMemory(self.embedder, self.config.strong, self.config.weak)
                    begin = time.perf_counter()
                    decisions = memory.fuse(triples)
                    fusion_seconds = time.perf_counter()-begin
                    events.append(dict(type="explore", query=value, extraction=extraction, alignment=decisions,
                                       retrieval_seconds=retrieval_seconds, fusion_seconds=fusion_seconds))
                    last_prune = False
                history.append(dict(assessment=assessment, action=action, value=value))
        except ContextOverflow:
            return finish("", "context_overflow")
