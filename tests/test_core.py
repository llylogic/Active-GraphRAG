import math
import json
import numpy as np
import pytest
from active_graphrag.schema import Config, Generation, Passage, ContextOverflow
from active_graphrag.graph import GraphMemory, parse_triples, serialize
from active_graphrag.pipeline import ActiveGraphRAG, parse_action
from active_graphrag.evaluation import score, calibrate
from active_graphrag.retrieval import chunk_passages


class Embedder:
    def encode(self, texts):
        vectors = {"A":[1,0,0],"A2":[0.99,0.1,0],"Alias":[0.75,0.66,0],"B":[0,0,1],"C":[0,1,0]}
        a = np.array([vectors[t] for t in texts],dtype=float)
        return a / np.linalg.norm(a,axis=1,keepdims=True)


class Model:
    def __init__(self, outputs):
        self.outputs, self.inputs = iter(outputs), []
    def generate(self, messages, gate=False):
        self.inputs.append(messages)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output if isinstance(output,Generation) else Generation(output)


class Retriever:
    def __init__(self): self.queries=[]
    def search(self,q,k):
        self.queries.append(q)
        return [Passage("p","title","A relates to B")]


def test_normalize_triples():
    assert parse_triples('[{"head":" A ","relation":" NOT   IN ","tail":" B "},[1,"x","Y"]]') == [("A","not in","B")]
    assert parse_triples('not json') == []


def test_fixed_snapshot_alignment_and_dedup():
    m=GraphMemory(Embedder())
    m.fuse([("A","r","A2")])
    assert len(m.graph)==2  # No merging against nodes introduced in the same update.
    m.fuse([("A","r","B"),("A","other","B")])
    count=m.graph.number_of_edges()
    m.fuse([("A","r","B")])
    assert m.graph.number_of_edges()==count


def test_alias_and_prune():
    m=GraphMemory(Embedder()); m.fuse([("A","r","B")])
    m.fuse([("Alias","r","B")])
    alias=next(n for n,d in m.graph.nodes(data=True) if d["label"]=="Alias")
    assert m.graph.nodes[alias]["alias_candidates"]==["n00000000"]
    m.prune("n00000000")
    assert m.graph.nodes[alias]["alias_candidates"]==[]
    assert len(m.graph.edges)==1


def test_gate_and_budget():
    gate=Generation("secret",logprobs=[math.log(.1)])
    nav=Model([gate,"[Missing_Link] <Explore: query>","[Missing_Link] <Explore: again>",'{"answer":"partial"}'])
    ext=Model(['[["A","r","B"]]']); retriever=Retriever()
    result=ActiveGraphRAG(nav,ext,Embedder(),Config(max_rounds=1)).run("question",retriever,.9)
    assert result["answer"]=="partial" and result["rounds"]==1
    assert len(retriever.queries)==1
    assert all("secret" not in json.dumps(messages) for messages in nav.inputs[1:])


def test_gate_bypass_no_retrieval():
    nav=Model([Generation("B",logprobs=[math.log(.9)])])
    result=ActiveGraphRAG(nav,Model([]),Embedder(),Config()).run("q",None,.8)
    assert result["answer"]=="B" and result["rounds"]==0


def test_no_consecutive_pruning():
    nav=Model(["[Missing_Link] <Explore: query>","[Dead_End] <Prune: n00000000>","[Dead_End] <Prune: n00000001>",'{"answer":""}'])
    result=ActiveGraphRAG(nav,Model(['[["A","r","B"]]']),Embedder(),Config()).run("q",Retriever(),mode="always")
    assert result["reason"]=="inadmissible_prune"
    assert len(result["graph"]["nodes"])==1


def test_overflow_included_empty():
    result=ActiveGraphRAG(Model([ContextOverflow()]),Model([]),Embedder(),Config()).run("q",None,mode="always")
    assert result["answer"]=="" and result["reason"]=="context_overflow"


def test_parse_constraints():
    assert parse_action("[Dead_End] <Explore: a>") is None
    assert not Generation("UNKNOWN",logprobs=[0]).valid_gate
    assert not Generation("answer",logprobs=[0],truncated=True).valid_gate


def test_calibration_full_subset_loss():
    rows=[dict(id=str(i),confidence=.8 if i==0 else None,gate_f1=99,always_f1=100) for i in range(100)]
    assert calibrate(rows)["mean_f1_loss"]==.01
    rows=[dict(id="0",confidence=.8,gate_f1=0,always_f1=100)]
    assert calibrate(rows)["threshold"] is None


def test_scores():
    assert score("The Belfast",["Belfast"])["em"]==100
    assert score("yes indeed",["yes"])["f1"]==0
    assert score("Northern",["Northern Ireland"])["f1"]==pytest.approx(200/3)


def test_chunk_stop_at_end():
    class Tokenizer:
        def encode(self,text,**kwargs): return text.split()
        def decode(self,ids,**kwargs): return " ".join(ids)
    p=Passage("p","t"," ".join(str(i) for i in range(550)))
    chunks=chunk_passages([p],Tokenizer())
    assert len(chunks)==2 and chunks[1].id=="p:50:550"
