from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import BatchEncoding

from models import AbstractModelWrapper
from models.utils import pool


class GemmaWrapper(AbstractModelWrapper):

    def __init__(
        self,
        pretrained_model_name="google/embeddinggemma-300m",
    ):
        super().__init__(pretrained_model_name, torch.bfloat16, max_tokens=2048)
        self.query_instruct = "task: search result | query:"
        self.corpus_instruct = "title: 'none' | text: {content}"
        self.chunk_size = 16

    def _get_query_instruct(self, query: str) -> str:
        return f"{self.query_instruct} {query}"

    @torch.no_grad()
    def _do_encode(self, corpus: List[str], apply_pooling: bool = True):
        inputs = self.tokenizer(corpus, padding=True, truncation=True, return_tensors='pt', max_length=2048)
        iterator = range(0, len(corpus), self.chunk_size)
        embeds, model_outputs = [], []

        for start_idx in tqdm(iterator, desc="Generating embeddings", total=len(iterator)):
            sub_inputs = BatchEncoding()
            sub_inputs['input_ids'] = inputs['input_ids'][start_idx:start_idx + self.chunk_size]
            sub_inputs['attention_mask'] = inputs['attention_mask'][start_idx:start_idx + self.chunk_size]
            sub_inputs.to('cuda', non_blocking=True)
            outputs = self.encoder(**sub_inputs)

            if apply_pooling:
                embeds.append(self.pool_and_normalize(
                    outputs.last_hidden_state, sub_inputs['attention_mask']
                ).detach().cpu())
            else:
                model_outputs.append({
                    "last_hidden_state": outputs.last_hidden_state.detach().cpu(),
                    "attention_mask": sub_inputs["attention_mask"].detach().cpu()
                })

        if embeds:
            embeds = torch.vstack(embeds).detach().cpu().float()

        return embeds, model_outputs

    def pool_and_normalize(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.tensor:
        embeds = pool(last_hidden_states, attention_mask, "avg")
        return torch.nn.functional.normalize(embeds, p=2, dim=-1).float()

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        sentences = [self._get_query_instruct(query) for query in queries]
        embeds, _ = self._do_encode(sentences)
        return embeds

    def encode_corpus(self, corpus: List[str]) -> torch.Tensor:
        sentences = [self.corpus_instruct.format(content=text) for text in corpus]
        embeds, _ = self._do_encode(sentences)
        return embeds

    def encode_corpus_no_pooling(self, corpus: List[str]) -> List[Dict[str, torch.Tensor]]:
        sentences = [self.corpus_instruct.format(content=text) for text in corpus]
        _, outputs = self._do_encode(sentences, False)
        return outputs
