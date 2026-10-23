from dataclasses import dataclass, field
import math


@dataclass
class Passage:
    id: str
    title: str
    text: str
    parent_id: str = ""


@dataclass
class Example:
    id: str
    question: str
    passages: list[Passage]
    answers: list[str]
    dataset: str


@dataclass
class Generation:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    logprobs: list[float] = field(default_factory=list)
    truncated: bool = False

    @property
    def valid_gate(self):
        return bool(self.text.strip()) and self.text.strip().upper() != "UNKNOWN" and not self.truncated and bool(self.logprobs)

    @property
    def confidence(self):
        return math.exp(sum(self.logprobs) / len(self.logprobs)) if self.valid_gate else None


@dataclass
class Config:
    navigator: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    extractor: str = "Qwen/Qwen2.5-1.5B-Instruct"
    embedding: str = "BAAI/bge-m3"
    navigator_device: str = "cuda:0"
    extractor_device: str = "cuda:1"
    embedding_device: str = "cpu"
    max_rounds: int = 4
    top_k: int = 3
    window: int = 500
    stride: int = 50
    strong: float = 0.85
    weak: float = 0.65
    input_limit: int = 7168
    output_limit: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9
    stateful: bool = True
    reflection: bool = True
    graceful: bool = True

    def __post_init__(self):
        if not 0 <= self.weak <= self.strong <= 1:
            raise ValueError("Require 0 <= weak <= strong <= 1")
        if min(self.max_rounds, self.top_k, self.window, self.stride) < 1:
            raise ValueError("Budgets and window settings must be positive")
        if self.stride > self.window:
            raise ValueError("Stride cannot leave gaps between windows")


class ContextOverflow(RuntimeError):
    pass
