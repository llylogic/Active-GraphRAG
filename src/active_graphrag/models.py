import math
from .schema import Generation, ContextOverflow


class HuggingFaceModel:
    def __init__(self, name, device, input_limit=7168, output_limit=1024, temperature=0.1, top_p=0.9):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        self.model.requires_grad_(False)
        self.input_limit, self.output_limit = input_limit, output_limit
        self.temperature, self.top_p = temperature, top_p

    def generate(self, messages, gate=False):
        torch = self.torch
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        if ids.shape[1] > self.input_limit:
            raise ContextOverflow(f"Templated input has {ids.shape[1]} tokens (limit {self.input_limit})")
        limit = 64 if gate else self.output_limit
        options = dict(max_new_tokens=limit, do_sample=not gate, return_dict_in_generate=True,
                       output_logits=gate, pad_token_id=self.tokenizer.eos_token_id)
        if not gate:
            options.update(temperature=self.temperature, top_p=self.top_p)
        with torch.inference_mode():
            result = self.model.generate(ids, attention_mask=torch.ones_like(ids), **options)
        output = result.sequences[0, ids.shape[1]:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        answer_ids, logprobs = [], []
        for i, token in enumerate(output):
            if token in eos:
                break
            answer_ids.append(token)
            if gate:
                logprobs.append(float(torch.log_softmax(result.logits[i][0].float(), dim=-1)[token].item()))
        return Generation(self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip(), ids.shape[1], len(output),
                          logprobs, len(output) >= limit and (not output or output[-1] not in eos))


class BGEEmbedder:
    def __init__(self, name="BAAI/bge-m3", device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=device)
        self.tokenizer = self.model.tokenizer

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
